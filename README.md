# Context-Aware Model Library for Layerwise Quantization

*Selective quantization / pruning & attention pattern editing for causal LMs*

## Installation

```bash
pip install contextq
```

## Current Status

* **Release:** Available on PyPI – give it a try.
* **Last update:** **August 2, 2025**

## Key Features

* Gradient‑aware (importance‑driven) **4‑bit** quantization workflow for any Hugging Face causal‑LM (defaults to `Llama‑3.2‑1B‑Instruct`).
* Calibration on **WikiText** (custom HF dataset support coming soon).
* Supports bf16 → 4‑bit with at least 50 % of layers quantized.
* Experimental mixed‑precision pipeline (32 / 16 / 8 / 4 bit); 3‑2 bit quantization planned.

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

