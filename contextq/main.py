"""
Selective-GPT: Gradient-aware 4-bit quantization demo
Target model: customizable via CLI or defaults to Llama-3.2-1B-Instruct
Requires: torch>=2.2, transformers>=4.43, bitsandbytes>=0.43
"""

import os
import argparse
import gc
import torch
import bitsandbytes as bnb
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Selective gradient-based 4-bit quantization for causal LMs"
    )
    parser.add_argument("--model_id", type=str,
                        default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Hugging Face model identifier")
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
    """
    def __init__(
        self,
        model_id: str,
        cache_dir: str,
        calib_samples: int,
        batch_size_fwd: int,
        batch_size_eval: int,
        quant_fraction: float,
        device: str
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

    def load_bf16_model(self):
        """Load model and tokenizer in BF16 precision."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=self.cache_dir
        ).eval()
        return self.model, self.tokenizer

    def make_dataloader(self, split: str, batch_size: int) -> DataLoader:
        """Create a DataLoader for calibration or evaluation."""
        ds = (
            load_dataset(
                "wikitext", "wikitext-2-raw-v1",
                split=split,
                cache_dir=self.cache_dir
            )
            .shuffle(seed=42)
            .select(range(self.calib_samples))
        )
        def collate(batch):
            texts = [ex["text"] for ex in batch]
            toks = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="longest",
            )
            toks["labels"] = toks["input_ids"].clone()
            return toks
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            drop_last=False,
        )

    def compute_grad_importance(self, loader: DataLoader) -> dict:
        """Accumulate gradient-norm per linear layer over the calibration set."""
        grad_norm = {name: 0.0 for name, m in self.model.named_modules()
                     if isinstance(m, nn.Linear)}
        self.model.to(self.device)
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.model.zero_grad(set_to_none=True)
            out = self.model(**batch)
            out.loss.backward()
            for name, mod in self.model.named_modules():
                if isinstance(mod, nn.Linear) and mod.weight.grad is not None:
                    grad_norm[name] += mod.weight.grad.pow(2).sum().item()
        return grad_norm

    @staticmethod
    def to_nf4(layer: nn.Linear) -> bnb.nn.Linear4bit:
        """Convert a linear layer to 4-bit NF4 and copy weights/bias."""
        q = bnb.nn.Linear4bit(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            quant_type="nf4",
            compute_dtype=torch.bfloat16,
        )
        q.weight.data = layer.weight.data.clone()
        if layer.bias is not None:
            q.bias.data = layer.bias.data.clone()
        return q.to(layer.weight.device)

    def selective_quantize(self, grad_scores: dict) -> set:
        """Quantize the bottom fraction of layers by gradient importance."""
        sorted_layers = sorted(grad_scores.items(), key=lambda kv: kv[1])
        k = int(len(sorted_layers) * self.quant_fraction)
        to_quant = {name for name, _ in sorted_layers[:k]}
        for name, module in self.model.named_modules():
            if name in to_quant and isinstance(module, nn.Linear):
                parent_name, child = name.rsplit('.', 1)
                parent = dict(self.model.named_modules())[parent_name]
                setattr(parent, child, self.to_nf4(module))
        return to_quant

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
        return tot_loss / tot_tokens

    @staticmethod
    def model_size_mb(model: nn.Module) -> float:
        """Return total parameter size in MB."""
        return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    def run(self):
        print(f"Loading model '{self.model_id}' on {self.device}…")
        self.load_bf16_model()
        train_loader = self.make_dataloader('train', self.batch_size_fwd)

        print("Computing gradient importance…")
        grad_scores = self.compute_grad_importance(train_loader)
        print(f"Found {len(grad_scores)} linear layers.")

        print(f"Quantizing bottom {int(self.quant_fraction*100)}% by grad-norm…")
        quantized = self.selective_quantize(grad_scores)
        print(f"Quantized {len(quantized)} layers.")

        size_mb = self.model_size_mb(self.model)
        print(f"Parameter size after quantization: {size_mb:.1f} MB")

        eval_loader = self.make_dataloader('validation', self.batch_size_eval)
        print("Evaluating full-precision baseline…")
        fp_model, _ = self.load_bf16_model()
        fp_loss = self.evaluate_loss(eval_loader)
        print(f"Full-precision loss: {fp_loss:.4f}")

        print("Evaluating mixed-precision model…")
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
        device=args.device
    )
    quantizer.run()

if __name__ == "__main__":
    main()
