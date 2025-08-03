"""
Selective-GPT: Gradient-aware 4-bit quantization demo
Target model: meta-llama/Llama-3.1-8B-Instruct  (24-layer decoder)
Hardware: single ≥24 GB GPU (RTX 4090 / A6000 / H100-80G…)
Requires: torch>=2.2, transformers>=4.43, bitsandbytes>=0.43
─────────────────────────────────────────────────────────────
"""

import os, math, gc, torch, bitsandbytes as bnb
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ─────────────────────────────────────────────────────────────
# CONFIG
MODEL_ID       = "meta-llama/Llama-3.2-1B"
CACHE_DIR      = "./hf_cache"
BATCH_SIZE_FWD = 4          # forward pass during grad-scan
BATCH_SIZE_EVAL= 8          # evaluation
CALIB_SAMPLES  = 128        # how many lines of Wikitext-2 to scan
QUANT_FRACTION = 0.50       # quantize bottom-50 % by grad-norm
DEVICE         = "cuda"

# ─────────────────────────────────────────────────────────────
def load_bf16_model(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    ).eval()
    return model, tok

# ─────────────────────────────────────────────────────────────
def make_dataloader(tokenizer, split, n_lines, batch_size):
    ds = (load_dataset("wikitext", "wikitext-2-raw-v1", split=split,
                       cache_dir=CACHE_DIR)
          .shuffle(seed=0)
          .select(range(n_lines)))
    def collate(batch):
        # batch is a list of dicts; extract the texts into a list of strings
        texts = [example["text"] for example in batch]

        toks = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="longest",
        )
        # set up labels for causal LM loss
        toks["labels"] = toks["input_ids"].clone()
        return toks
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
    )


# ─────────────────────────────────────────────────────────────
def grad_importance(model: nn.Module, loader: DataLoader):
    """
    Return dict[layer_name] = accumulated L2-grad-norm
    """
    grad_norm = {n: 0.0 for n, m in model.named_modules()
                 if isinstance(m, nn.Linear)}
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        out = model(**batch)
        # HF causal-LM returns loss already computed, but we want grads
        loss = out.loss
        loss.backward()
        # accumulate per-linear grad norms
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and mod.weight.grad is not None:
                grad_norm[name] += mod.weight.grad.pow(2).sum().item()
    return grad_norm

# ─────────────────────────────────────────────────────────────
def to_nf4(layer: nn.Linear) -> bnb.nn.Linear4bit:
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

def selective_quantize(model: nn.Module, grad_scores: dict,
                       fraction: float):
    # pick bottom-fraction by grad importance
    sorted_layers = sorted(grad_scores.items(),
                           key=lambda kv: kv[1])
    k = int(len(sorted_layers) * fraction)
    to_quant = {name for name, _ in sorted_layers[:k]}
    for name, module in model.named_modules():
        if name in to_quant and isinstance(module, nn.Linear):
            parent_name, child = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, child, to_nf4(module))
    return to_quant

# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader):
    tot_loss, tot_tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch)
        # multiply by number of target tokens (all except pad)
        n_tok = (batch["labels"] != -100).sum().item()
        tot_loss   += out.loss.item() * n_tok
        tot_tokens += n_tok
    return tot_loss / tot_tokens

def model_size_mb(model: nn.Module):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


if __name__ == "__main__":
    print("Loading model …")
    model, tok = load_bf16_model(MODEL_ID)
    dataloader = make_dataloader(tok, "train", CALIB_SAMPLES, BATCH_SIZE_FWD)

    print("⏳  Gradient scan …")
    grad_scores = grad_importance(model, dataloader)
    print("Collected grad norms for", len(grad_scores), "linear layers.")

    print(f"📉  Quantizing bottom {int(QUANT_FRACTION*100)} % by grad-norm …")
    quantized_layers = selective_quantize(model, grad_scores, QUANT_FRACTION)
    print("Quantized", len(quantized_layers), "layers to 4-bit NF4.")

    # size stats
    full_mb   = model_size_mb(model)  # after quant some params are NF4 but
    # we can estimate bf16-equivalent vs real memory using bitsandbytes attr
    real_mb   = sum(p.storage().nbytes() for p in model.parameters()) / 1e6
    print(f"Model parameter size: {full_mb:,.1f} MB (bf16)—"
          f" actually {real_mb:,.1f} MB in VRAM after mixed quant.")

    # evaluate
    eval_loader = make_dataloader(tok, "validation", CALIB_SAMPLES,
                                  BATCH_SIZE_EVAL)
    print("🔄  Evaluating full-precision baseline …")
    # need a fresh fp copy for fair compare
    fp_model, _ = load_bf16_model(MODEL_ID)
    fp_loss = evaluate_loss(fp_model, eval_loader)
    print(f"   full-precision loss: {fp_loss:.4f}")

    print("🔄  Evaluating mixed-precision model …")
    mixed_loss = evaluate_loss(model, eval_loader)
    print(f"   mixed loss        : {mixed_loss:.4f}")

    print(f"Δ loss = {mixed_loss-fp_loss:+.4f}")
