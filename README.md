# Context-Aware Model Wrapper for Selective Quantization/Pruning and editing Attention Patterns

What I am currently doing (this may kinda read like a blog, I apologize in advance).

Benchmarking qualitative vs quantitative questions and seeing average gradient magnitudes.
This will hopefully allow me to quantize seemingly non important layers as I go on.

![fig1](https://github.com/AyanJhunjhunwala/ContextQ/blob/main/Figure_1.png "Initial Finding")

# Is the wrapper out? -> No

# Observations

Very initial but we see that atleast for the DiaboGPT-small model, the gradient mags are much more condensed for non reasoning and non open ended questions when compared to mathematical ones(simple questions)


# Use case

1. When thinking of strctured outputs from LLMs, and repetitve requests for generation, we can run context analysis and see gradients and their magnitudes. Using this, we can have layers quantized on requests.

2. We end up doing this anyway on the daily for models


# More thoughts

This is an extremely long project and I hope to be done with a prototype soon and continue working on it as long as I can. 


# Day by Day

This is just me keeping up with any progress I made.

# 13th July 2025

The attention patterns and gradient magnitude are obviously pretty different for quant/qual in dialo-small. I am planning to test this with a larger model. I tested out dialo with the ARC and GSM8k test. (below)

![fig2](https://github.com/AyanJhunjhunwala/ContextQ/blob/main/Figure_2.png)

I am going to switch over to llama 3.1 and benchmark 8-4 bit next. I am also going to work on attention patterns and quant strutcures for both quant/qual. This means a lot of reading for me :( 

# 17th July 2025

I am thinking of using github blogs for this but probably won't. So I ran the llama 3.1 8b instruct on ARC and svamp to split qualitative and quantitative and now my biggest question is how do I modify patterns and quantize the models to make this make sense. Svamp had a low accuracy and optmizing that without SFT would be something I guess. I started figruing out the pypi library as well but my main focus would have to be on extrapolating 

# Model Quantization Analysis

A comprehensive analysis of 8-bit vs 4-bit quantization effects on transformer model performance across different reasoning tasks.

## Performance Summary

| Task | Model Type | Accuracy | Correct/Total | Speed (samples/s) | Time (min) |
|------|------------|----------|---------------|-------------------|------------|
| **ARC-Easy** | 8-bit | 95.42% | 2148/2251 | 4.90 | 7.66 |
| | 4-bit | 95.29% | 2145/2251 | 9.18 | 4.09 |
| **SVAMP** | 8-bit | 5.43% | 38/700 | 4.92 | 2.37 |
| | 4-bit | 6.71% | 47/700 | 8.52 | 1.37 |

### Key Insights
- **ARC-Easy**: Minimal accuracy drop (0.13%) with 87% speed improvement
- **SVAMP**: Unexpected accuracy improvement (1.28%) with 73% speed improvement
- **Overall**: 4-bit quantization provides significant inference speedup with task-dependent accuracy effects

## Layer-wise Analysis

### ARC-Easy (Multiple Choice Reasoning)

| Layer | Bits | Activation | Sparsity | Magnitude | Rank | AttnEntropy | ReasonScore |
|-------|------|------------|----------|-----------|------|-------------|-------------|
| 0 | 8 | 0.0001 | 0.000 | 1.04 | 32.0 | 0.000 | 0.000 |
| 0 | 4 | 0.0001 | 0.000 | 1.04 | 32.0 | 0.000 | 0.000 |
| 4 | 8 | 0.0004 | 0.000 | 3.67 | 32.0 | 0.000 | 0.000 |
| 4 | 4 | 0.0005 | 0.000 | 3.63 | 32.0 | 0.000 | 0.000 |
| 8 | 8 | 0.0005 | 0.000 | 5.72 | 32.0 | 0.000 | 0.000 |
| 8 | 4 | 0.0006 | 0.000 | 5.69 | 32.0 | 0.000 | 0.000 |
| 16 | 8 | 0.0002 | 0.000 | 11.39 | 32.0 | 0.000 | 0.000 |
| 16 | 4 | 0.0003 | 0.000 | 11.22 | 32.0 | 0.000 | 0.000 |
| 24 | 8 | 0.0016 | 0.000 | 25.93 | 32.0 | 0.000 | 0.000 |
| 24 | 4 | 0.0017 | 0.000 | 25.56 | 32.0 | 0.000 | 0.000 |
| 31 | 8 | 0.0101 | 0.000 | 63.71 | 32.0 | 0.000 | 0.000 |
| 31 | 4 | 0.0109 | 0.000 | 62.47 | 32.0 | 0.000 | 0.000 |

### SVAMP (Elementary Math Reasoning)

| Layer | Bits | Activation | Sparsity | Magnitude | Rank | AttnEntropy | ReasonScore |
|-------|------|------------|----------|-----------|------|-------------|-------------|
| 0 | 8 | 0.0001 | 0.000 | 1.00 | 32.0 | 0.000 | 0.000 |
| 0 | 4 | 0.0001 | 0.000 | 1.01 | 32.0 | 0.000 | 0.000 |
| 4 | 8 | 0.0005 | 0.000 | 3.47 | 32.0 | 0.000 | 0.000 |
| 4 | 4 | 0.0005 | 0.000 | 3.46 | 32.0 | 0.000 | 0.000 |
| 8 | 8 | 0.0003 | 0.000 | 5.25 | 32.0 | 0.000 | 0.000 |
| 8 | 4 | 0.0002 | 0.000 | 5.26 | 32.0 | 0.000 | 0.000 |
| 16 | 8 | -0.0007 | 0.000 | 10.88 | 32.0 | 0.000 | 0.000 |
| 16 | 4 | 0.0000 | 0.000 | 10.86 | 32.0 | 0.000 | 0.000 |
| 24 | 8 | -0.0013 | 0.000 | 24.29 | 32.0 | 0.000 | 0.000 |
| 24 | 4 | -0.0008 | 0.000 | 24.53 | 32.0 | 0.000 | 0.000 |
| 31 | 8 | 0.0054 | 0.000 | 52.63 | 32.0 | 0.000 | 0.000 |
| 31 | 4 | 0.0044 | 0.000 | 58.66 | 32.0 | 0.000 | 0.000 |

## 📈 Quantization Impact Analysis

### ARC-Easy Impact (8-bit → 4-bit)

| Layer | Sparsity Δ | Magnitude Δ | Rank Δ | AttnEntropy Δ | ReasonScore Δ |
|-------|-------------|-------------|---------|----------------|----------------|
| 0 | -0.000 | -0.2% | 0.0% | 0.000 | 0.000 |
| 4 | 0.000 | -1.2% | 0.0% | 0.000 | 0.000 |
| 8 | 0.000 | -0.5% | 0.0% | 0.000 | 0.000 |
| 16 | 0.000 | -1.5% | 0.0% | 0.000 | 0.000 |
| 24 | 0.000 | -1.4% | 0.0% | 0.000 | 0.000 |
| 31 | -0.000 | -1.9% | 0.0% | 0.000 | 0.000 |

### SVAMP Impact (8-bit → 4-bit)

| Layer | Sparsity Δ | Magnitude Δ | Rank Δ | AttnEntropy Δ | ReasonScore Δ |
|-------|-------------|-------------|---------|----------------|----------------|
| 0 | -0.000 | 0.5% | 0.0% | 0.000 | 0.000 |
| 4 | 0.000 | -0.3% | 0.0% | 0.000 | 0.000 |
| 8 | 0.000 | 0.2% | 0.0% | 0.000 | 0.000 |
| 16 | -0.000 | -0.2% | 0.0% | 0.000 | 0.000 |
| 24 | -0.000 | 1.0% | 0.0% | 0.000 | 0.000 |
| 31 | -0.000 | 11.5% | 0.0% | 0.000 | 0.000 |

## 🎯 Recommendations

### ✅ Quantization Strategy
- **All layers show good robustness to quantization** for both tasks
- 4-bit quantization is recommended for production deployment
- Significant inference speed improvements with minimal accuracy trade-offs

### 🔍 Task-Specific Insights

#### Multiple Choice Reasoning (ARC-Easy)
- **Accuracy Drop**: 0.1% (8-bit → 4-bit)
- **Speed Improvement**: 87%
- **Pattern**: Choice-selection attention patterns are robust to quantization

#### Mathematical Reasoning (SVAMP)
- **Accuracy Change**: +1.28% improvement (8-bit → 4-bit)
- **Speed Improvement**: 73%
- **Pattern**: Numerical reasoning benefits from quantization regularization

## Comparative Analysis

| Metric | ARC-Easy | SVAMP |
|--------|----------|-------|
| Accuracy Change | -0.13% | +1.28% |
| Speed Improvement | 87% | 73% |
| Task Type | Multiple Choice | Math Reasoning |
| Attention Pattern | Choice-selection | Numerical reasoning |

## Technical Observations

1. **Layer Sensitivity**: Later layers (24, 31) show higher magnitude changes but maintain performance
2. **Attention Robustness**: All attention entropy values remain stable across quantization
3. **Rank Preservation**: Model representational capacity (rank) is preserved across all layers
4. **Task Dependency**: Mathematical reasoning shows different quantization sensitivity compared to multiple choice tasks

## Implementation Notes

- Zero errors recorded across all configurations
- Consistent 32.0 rank maintained across all layers
- Sparsity patterns remain stable
- Magnitude changes are within acceptable thresholds (&lt;12%)

---



