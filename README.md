# Python Library Wrapper for Intelligent Context Aware Quantization

What I am currently doing (this may kinda read like a blog, I apologize in advance).

Benchmarking qualitative vs quantitative questions and seeing average gradient magnitudes.
This will hopefully allow me to quantize seemingly non important layers as I go on.

![fig1](https://github.com/AyanJhunjhunwala/ContextQ/blob/main/Figure_1.png "Initial Finding")


# Observations

Very initial but we see that atleast for the DiaboGPT-small model, the gradient mags are much more condensed for non reasoning and non open ended questions when compared to mathematical ones(simple questions)


# Use case

1. When thinking of strctured outputs from LLMs, and repetitve requests for generation, we can run context analysis and see gradients and their magnitudes. Using this, we can have layers quantized on requests.

2. We end up doing this anyway on the daily for models


# More thoughts

This is an extremely long project and I hope to be done with a prototype soon and continue working on it as long as I can. 
