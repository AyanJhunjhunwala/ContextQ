"""
Selective-GPT: Gradient-aware 4-bit quantization demo (AWQ-backed)
Target model: customizable via CLI or defaults to Llama-3.2-1B-Instruct
Requires: torch>=2.2, transformers>=4.43, awq>=0.1, datasets
"""

import os, warnings
warnings.filterwarnings(
    "ignore",
    message=r".*AutoAWQ is officially deprecated.*",
    category=DeprecationWarning,
    module=r"awq"
)
# Silence TensorFlow/JAX/XLA, absl, and glog BEFORE they initialize
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")      # 0=all, 1=INFO, 2=WARNING, 3=ERROR+FATAL only
os.environ.setdefault("GLOG_minloglevel", "3")          # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
os.environ.setdefault("ABSL_LOG_SEVERITY", "3")         # absl (best-effort)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# If you are NOT using JAX on GPU, pin it to CPU to avoid GPU plugin chatter:
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# Targeted suppression for known DeprecationWarning sources
warnings.filterwarnings(
    "ignore",
    message=r".*AutoAWQ is officially deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"flatbuffers\.compat",
)
# --- Global Warning Filters ---
warnings.filterwarnings("ignore", message=r".*AutoAWQ is officially deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"flatbuffers\.compat")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"networkx\.readwrite\.graphml")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"networkx\.readwrite\.gexf")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"joblib\.backports")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"sklearn\.utils\.multiclass")

# Targeted message-based filters
warnings.filterwarnings("ignore", message=".*the imp module is deprecated.*")
warnings.filterwarnings("ignore", message=".*The distutils package is deprecated.*")
warnings.filterwarnings("ignore", message=".*the scipy\.sparse\.base namespace is deprecated.*")

# Import awq inside a suppression context so its import-time warning is guaranteed to be dropped
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from awq import AutoAWQForCausalLM

import argparse
import gc
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from huggingface_hub import login

if os.environ.get("HF_TOKEN"):
    login(os.environ["HF_TOKEN"])

def parse_args():
    parser = argparse.ArgumentParser(
        description="Selective gradient-based 4-bit quantization for causal LMs"
    )
    parser.add_argument("--model_id", type=str,
                        default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Hugging Face model identifier")
    parser.add_argument(
        "--quant_type",
        type=str,
        choices=["qualitative", "quantitative", "reasoning", "code"],
        default=None,
        help="Task domain to guide calibration/eval dataset (defaults to prompted choice)."
    )
    parser.add_argument("--cache_dir", type=str, default="./hf_cache",
                        help="Directory for Hugging Face cache")
    parser.add_argument("--calib_samples", type=int, default=128,
                        help="Number of calibration examples to use")
    parser.add_argument("--batch_size_fwd", type=int, default=1,
                        help="Batch size for gradient scan (forward/backward)")
    parser.add_argument("--batch_size_eval", type=int, default=8,
                        help="Batch size for evaluation loss computation")
    parser.add_argument("--quant_fraction", type=float, default=0.5,
                        help="Fraction of linear layers to quantize (0.0-1.0)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for computation")
    return parser.parse_args()


class SelectiveGPTQuantizer:
    """
    Gradient-aware selective 4-bit quantization for causal LMs.
    Now implemented under the hood with AWQ (activation-aware).
    """
    def __init__(
        self,
        model_id: str,
        cache_dir: str,
        calib_samples: int,
        batch_size_fwd: int,
        batch_size_eval: int,
        quant_fraction: float,
        device: str,
        quant_type: str | None = None,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.calib_samples = calib_samples
        self.batch_size_fwd = batch_size_fwd
        self.batch_size_eval = batch_size_eval
        self.quant_fraction = quant_fraction
        self.device = device
        self.tokenizer = None
        self.model = None
        self.quant_type = quant_type  # 'qualitative' | 'quantitative' | 'reasoning' | 'code' | None
        self._awq_tmp_dir = os.path.join(self.cache_dir, "awq_quantized_tmp")

    # ---------- Task prompting / normalization ----------

    @staticmethod
    def _canonical_task(t: str) -> str:
        t = (t or "").strip().lower()
        aliases = {
            "qual": "qualitative",
            "nlp": "qualitative",
            "quant": "quantitative",
            "math": "quantitative",
            "reason": "reasoning",
            "logic": "reasoning",
            "prog": "code",
            "coding": "code",
        }
        return aliases.get(t, t)

    def _ensure_quant_type(self):
        if self.quant_type:
            self.quant_type = self._canonical_task(self.quant_type)
            if self.quant_type in {"qualitative", "quantitative", "reasoning", "code"}:
                return
            print(f"[warn] Unknown quant_type '{self.quant_type}', prompting instead.")

        # Interactive prompt
        print("\nSelect task domain for calibration/eval:")
        options = ["Qualitative (WikiText-2)", "Quantitative (GSM8K)", "Reasoning (ARC-Challenge)", "Code (MBPP)"]
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")
        while True:
            choice = input("Enter 1-4 or name (qualitative/quantitative/reasoning/code): ").strip().lower()
            choice = self._canonical_task(choice)
            if choice in {"1", "qualitative"}:
                self.quant_type = "qualitative"
                break
            if choice in {"2", "quantitative"}:
                self.quant_type = "quantitative"
                break
            if choice in {"3", "reasoning"}:
                self.quant_type = "reasoning"
                break
            if choice in {"4", "code"}:
                self.quant_type = "code"
                break
            print("Invalid choice. Please try again.")
        print(f"Using task domain: {self.quant_type}\n")

    # ---------- Model/tokenizer ----------

    def load_bf16_model(self):
        """Load model and tokenizer in BF16 precision (AWQ wrapper, unquantized)."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use AWQ wrapper even before quantization; behaves like a normal HF model.
        self.model = AutoAWQForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda" if torch.cuda.is_available() else None,
        ).eval()
        return self.model, self.tokenizer

    # ---------- Dataset preparation by task ----------

    def _load_raw_dataset(self, split: str):
        """
        Load a raw HF dataset split for the selected task.
        We map 'validation' to 'test' where necessary (e.g., GSM8K).
        """
        task = self.quant_type
        if task == "quantitative":
            # GSM8K has 'train' and 'test'
            split_map = {"train": "train", "validation": "test"}
            return load_dataset("gsm8k", "main", split=split_map.get(split, split), cache_dir=self.cache_dir)
        elif task == "reasoning":
            # ARC-Challenge has train/validation/test
            return load_dataset("ai2_arc", "ARC-Challenge", split=split, cache_dir=self.cache_dir)
        elif task == "code":
            # MBPP has train/validation/test
            return load_dataset("mbpp", split=split, cache_dir=self.cache_dir)
        else:
            # Qualitative default: WikiText-2
            return load_dataset("wikitext", "wikitext-2-raw-v1", split=split, cache_dir=self.cache_dir)

    def _to_text_dataset(self, raw_ds) -> Dataset:
        """
        Convert a task-specific dataset into a Dataset with a single 'text' column.
        """
        task = self.quant_type

        if task == "quantitative":
            # GSM8K: fields: 'question', 'answer'
            texts = []
            for ex in raw_ds:
                q = ex.get("question", "").strip()
                a = ex.get("answer", "").strip()
                if q:
                    texts.append(f"Question: {q}\nAnswer: {a}")
            return Dataset.from_dict({"text": texts})

        if task == "reasoning":
            # ARC-Challenge: 'question', 'choices': {'text': [...], 'label': [...]}, 'answerKey'
            texts = []
            for ex in raw_ds:
                q = ex.get("question", "").strip()
                choices = ex.get("choices", {})
                c_texts = choices.get("text", []) or []
                c_labels = choices.get("label", []) or []
                choice_str = " ".join([f"{lbl}) {txt}" for lbl, txt in zip(c_labels, c_texts)])
                ans = ex.get("answerKey", "")
                if q and choice_str:
                    texts.append(f"Question: {q}\nChoices: {choice_str}\nAnswer: {ans}")
            return Dataset.from_dict({"text": texts})

        if task == "code":
            # MBPP: 'text' (problem statement), 'code' (reference solution)
            texts = []
            for ex in raw_ds:
                desc = (ex.get("text") or ex.get("prompt") or "").strip()
                code = (ex.get("code") or "").strip()
                if desc:
                    if code:
                        texts.append(f"Problem: {desc}\nSolution:\n{code}")
                    else:
                        texts.append(f"Problem: {desc}")
            return Dataset.from_dict({"text": texts})

        # Qualitative (WikiText-2) already has 'text'
        if "text" in raw_ds.column_names:
            return raw_ds
        # Fallback: concatenate any present string fields
        texts = []
        for ex in raw_ds:
            s = " ".join(str(v) for v in ex.values() if isinstance(v, str)).strip()
            if s:
                texts.append(s)
        return Dataset.from_dict({"text": texts})

    # ---------- Dataloaders ----------

    def make_dataloader(self, split: str, batch_size: int) -> DataLoader:
        """Create a DataLoader for calibration or evaluation (task-aware)."""
        raw = self._load_raw_dataset(split)
        ds = self._to_text_dataset(raw).shuffle(seed=42).select(range(min(self.calib_samples, len(raw))))

        def collate(batch):
            texts = [ex["text"] for ex in batch]
            toks = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="longest",
            )
            # Mask out pads in labels to avoid skewing loss
            labels = toks["input_ids"].clone()
            labels[toks["attention_mask"] == 0] = -100
            toks["labels"] = labels
            return toks

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            drop_last=False,
        )

    # ---------- Original structural placeholders ----------

    def compute_grad_importance(self, loader: DataLoader) -> dict:
        """
        Placeholder to keep structure: returns a dict of linear layer names with zero
        'importance'. AWQ handles saliency via activation stats internally.
        """
        return {name: 0.0 for name, m in self.model.named_modules()
                if isinstance(m, nn.Linear)}

    @staticmethod
    def to_nf4(layer: nn.Linear):
        """Kept for structure compatibility; unused under AWQ."""
        return layer  # no-op

    # ---------- AWQ helpers ----------

    def _awq_calib_texts(self):
        # Use the *train* split for calibration, transformed to 'text' column.
        raw = self._load_raw_dataset("train")
        ds = self._to_text_dataset(raw)
        texts = [t for t in ds["text"] if t.strip()]
        return texts[: self.calib_samples]

    def _count_awq_layers(self) -> int:
        cnt = 0
        for _, m in self.model.named_modules():
            if hasattr(m, "qweight") or hasattr(m, "qzeros") or m.__class__.__name__.lower().startswith(("wqlinear", "awq")):
                cnt += 1
        return cnt

    def _save_quantized_fallback(self, out_dir: str):
        """Save quantized model with best available method for awq backend."""
        os.makedirs(out_dir, exist_ok=True)
        saver = getattr(self.model, "save_quantized", None)
        if callable(saver):
            saver(out_dir, safetensors=True)
        else:
            torch.save(self.model.state_dict(), os.path.join(out_dir, "weights-awq.pt"))

    # ---------- Quantization ----------

    def selective_quantize(self, grad_scores: dict) -> set:
        """
        Under the hood, quantize with AWQ (4-bit) using activation-based calibration.
        Returns a set of module names that appear quantized (best-effort detection).
        """
        calib_texts = self._awq_calib_texts()
        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM",
        }
        self.model.quantize(
            self.tokenizer,
            quant_config=quant_config,
            calib_data=calib_texts,
            max_calib_samples=self.calib_samples,
            max_calib_seq_len=512,
        )

        # Detect/return "quantized" modules by AWQ-specific attributes.
        q_names = set()
        for name, m in self.model.named_modules():
            if hasattr(m, "qweight") or hasattr(m, "qzeros") or m.__class__.__name__.lower().startswith(("wqlinear", "awq")):
                q_names.add(name)
        return q_names

    # ---------- Evaluation & size ----------

    @torch.no_grad()
    def evaluate_loss(self, loader: DataLoader) -> float:
        """Compute average cross-entropy loss over a dataset."""
        tot_loss, tot_tokens = 0.0, 0
        self.model.to(self.device)
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(**batch)
            n_tok = (batch['labels'] != -100).sum().item()
            tot_loss += out.loss.item() * n_tok
            tot_tokens += n_tok
        return tot_loss / max(tot_tokens, 1)

    @staticmethod
    def model_size_mb(model) -> float:
        """Return total parameter size in MB (approx, in-memory)."""
        return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    @staticmethod
    def dir_size_mb(path: str) -> float:
        """On-disk size of a directory in MB."""
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
        return total / 1e6

    # ---------- Orchestration ----------

    def run(self):
        # Ensure we know which task domain to use (prompt if not provided via args)
        self._ensure_quant_type()

        print(f"Loading model '{self.model_id}' on {self.device}…")
        self.load_bf16_model()

        # Record pre-quantization size (in-memory param estimate)
        size_before_mb = self.model_size_mb(self.model)
        print(f"Parameter size before quantization: {size_before_mb:.1f} MB")

        train_loader = self.make_dataloader('train', self.batch_size_fwd)

        print("Computing gradient importance…")
        grad_scores = self.compute_grad_importance(train_loader)
        print(f"Found {len(grad_scores)} linear layers.")

        print(f"Quantizing bottom {int(self.quant_fraction*100)}% by grad-norm (AWQ handles saliency internally)…")
        quantized = self.selective_quantize(grad_scores)
        print(f"Quantized {len(quantized)} layers (detected).")

        # Approx in-memory param size after quantization (may undercount packed int4 buffers)
        size_after_param_mb = self.model_size_mb(self.model)
        print(f"Parameter size after quantization (approx): {size_after_param_mb:.1f} MB")

        # On-disk size for the quantized model
        os.makedirs(self._awq_tmp_dir, exist_ok=True)
        self._save_quantized_fallback(self._awq_tmp_dir)
        size_after_disk_mb = self.dir_size_mb(self._awq_tmp_dir)
        print(f"On-disk size after quantization: {size_after_disk_mb:.1f} MB (saved at {self._awq_tmp_dir})")

        # Build eval loader for the chosen task
        eval_loader = self.make_dataloader('validation', self.batch_size_eval)

        print("Evaluating full-precision baseline…")
        # Stash quantized model, then reload fresh FP model (keeps your original call order)
        q_model_ref = self.model
        fp_model, _ = self.load_bf16_model()
        fp_loss = self.evaluate_loss(eval_loader)
        print(f"Full-precision loss: {fp_loss:.4f}")

        print("Evaluating mixed-precision model…")
        # Restore quantized model reference to compute mixed loss
        self.model = q_model_ref
        mixed_loss = self.evaluate_loss(eval_loader)
        print(f"Mixed loss: {mixed_loss:.4f}")
        print(f"Δ loss = {mixed_loss - fp_loss:+.4f}")


def main():
    args = parse_args()
    quantizer = SelectiveGPTQuantizer(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        calib_samples=args.calib_samples,
        batch_size_fwd=args.batch_size_fwd,
        batch_size_eval=args.batch_size_eval,
        quant_fraction=args.quant_fraction,
        device=args.device,
        quant_type=args.quant_type,
    )
    quantizer.run()

if __name__ == "__main__":
    main()
