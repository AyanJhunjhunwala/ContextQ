# Context-Aware Model Library for LLM Quantization

*Selective quantization using AWQ, context manager for specific models

## Installation

```bash
pip install contextq
```
```python
import contextq
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

quantizer = contextq.SelectiveGPTQuantizer("meta-llama/Llama-3.1-8B-Instruct", "code")
quant_dir = "./hf_cache/awq_quantized_tmp"
tok = AutoTokenizer.from_pretrained(quant_dir, trust_remote_code=True)

if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoAWQForCausalLM.from_quantized(
    quant_dir,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map="cuda" if torch.cuda.is_available() else "auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
```
## VLLM Support

```python
from vllm import LLM, SamplingParams

model_path = "./hf_cache/awq_quantized_tmp"

llm = LLM(
    model=model_path,
    quantization="awq",   
    dtype="auto",        
    trust_remote_code=True
)

sampler = SamplingParams(max_tokens=128, temperature=0.8, top_p=0.95)
outputs = llm.generate(["Explain attention like I'm five."], sampler)
print(outputs[0].outputs[0].text)

```

## Current Status

* **Release:** Available on PyPI – give it a try.
* **Last update:** **Sep 13, 2025**
* This runs perfectly now on AWQ. Just give the wrapper your model name and it will ask you options from Quantitative, Qualitative, Reasoning, Code for task specific quantization.

## Key Features

* Gradient‑aware (importance‑driven) **4‑bit** quantization workflow for any Hugging Face causal‑LM (defaults to `Llama‑3.2‑1B‑Instruct`).
* Calibration on **WikiText** (custom HF dataset support coming soon).
* Supports bf16 → 4‑bit with at least 50 % of layers quantized.
* Experimental mixed‑precision pipeline (32 / 16 / 8 / 4 bit); 3‑2 bit quantization planned.
* Adds Quantization profiles for different types of context quant. Less than 1% loss and super easy

## Quick Start

```bash
selective_gpt --help
```

## Development Notes

I wrapped this up on a late Saturday, so some code is still messy – a cleanup pass is on the way.

## Journal / Changelog

### July 21 2025

* Roadmap drafted: multi‑AWQ managed models for more aggressive quantization.
* Plan to add a lightweight classifier for pre‑defined categories (later).

### July 17 2025

* Ran *Llama 3.1‑8B‑Instruct* on ARC and SVAMP to separate qualitative/quantitative tasks.
* Investigating pattern modification and quantization for improved SVAMP accuracy (without SFT).
* Began shaping the PyPI release; extrapolation remains the focus.

### July 13 2025

* Divergent attention/gradient patterns observed between quant/qual on `dialo-small`.
* Tested `dialo` with ARC and GSM8k (results below).
* The wrapper is officially out – at this stage it focuses on quantization only.

---

Benchmarks and additional details will be added soon.

