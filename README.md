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

ARC-Easy (Multiple Choice Reasoning) Summary:
==========================================================================================    
Bits   Accuracy   Correct  Total   Errors  Time(m)  Samples/s
------------------------------------------------------------------------------------------    
8      0.9542     2148     2251    0       7.66     4.90      
4      0.9529     2145     2251    0       4.09     9.18

SVAMP (Elementary Math Reasoning) Summary:
==========================================================================================    
Bits   Accuracy   Correct  Total   Errors  Time(m)  Samples/s
------------------------------------------------------------------------------------------    
8      0.0543     38       700     0       2.37     4.92
4      0.0671     47       700     0       1.37     8.52

ARC-Easy Comprehensive Analysis:
========================================================================================================================
Layer    Bits   Activation   Sparsity   Magnitude    Rank     AttnEntropy  ReasonScore        
------------------------------------------------------------------------------------------------------------------------
0        8      0.0001       0.000      1.04         32.0     0.000        0.000
0        4      0.0001       0.000      1.04         32.0     0.000        0.000

4        8      0.0004       0.000      3.67         32.0     0.000        0.000
4        4      0.0005       0.000      3.63         32.0     0.000        0.000

8        8      0.0005       0.000      5.72         32.0     0.000        0.000
8        4      0.0006       0.000      5.69         32.0     0.000        0.000

16       8      0.0002       0.000      11.39        32.0     0.000        0.000
16       4      0.0003       0.000      11.22        32.0     0.000        0.000

24       8      0.0016       0.000      25.93        32.0     0.000        0.000
24       4      0.0017       0.000      25.56        32.0     0.000        0.000

31       8      0.0101       0.000      63.71        32.0     0.000        0.000
31       4      0.0109       0.000      62.47        32.0     0.000        0.000


Quantization Impact Analysis:
----------------------------------------------------------------------------------------------------
Layer    Sparsity Δ   Magnitude Δ    Rank Δ       AttnEntropy Δ    ReasonScore Δ
----------------------------------------------------------------------------------------------------
0        -0.000++++++ -0.2++++++++++% 0.0+++++++++% 0.000+++++++++++ 0.000+++++++++++
4        0.000+++++++ -1.2++++++++++% 0.0+++++++++% 0.000+++++++++++ 0.000+++++++++++
8        0.000+++++++ -0.5++++++++++% 0.0+++++++++% 0.000+++++++++++ 0.000+++++++++++
16       0.000+++++++ -1.5++++++++++% 0.0+++++++++% 0.000+++++++++++ 0.000+++++++++++
24       0.000+++++++ -1.4++++++++++% 0.0+++++++++% 0.000+++++++++++ 0.000+++++++++++
31       -0.000++++++ -1.9++++++++++% 0.0+++++++++% 0.000+++++++++++ 0.000+++++++++++

ARC-Easy Quantization Strategy Recommendations:
--------------------------------------------------------------------------------
✅ All layers show good robustness to quantization

SVAMP Comprehensive Analysis:
========================================================================================================================
Layer    Bits   Activation   Sparsity   Magnitude    Rank     AttnEntropy  ReasonScore        
------------------------------------------------------------------------------------------------------------------------
0        8      0.0001       0.000      1.00         32.0     0.000        0.000
0        4      0.0001       0.000      1.01         32.0     0.000        0.000

4        8      0.0005       0.000      3.47         32.0     0.000        0.000
4        4      0.0005       0.000      3.46         32.0     0.000        0.000

8        8      0.0003       0.000      5.25         32.0     0.000        0.000
8        4      0.0002       0.000      5.26         32.0     0.000        0.000

16       8      -0.0007      0.000      10.88        32.0     0.000        0.000
16       4      0.0000       0.000      10.86        32.0     0.000        0.000

24       8      -0.0013      0.000      24.29        32.0     0.000        0.000
24       4      -0.0008      0.000      24.53        32.0     0.000        0.000       

31       8      0.0054       0.000      52.63        32.0     0.000        0.000
31       4      0.0044       0.000      58.66        32.0     0.000        0.000


Quantization Impact Analysis:
----------------------------------------------------------------------------------------------------
Layer    Sparsity Δ   Magnitude Δ    Rank Δ       AttnEntropy Δ    ReasonScore Δ
----------------------------------------------------------------------------------------------------
0        -0.000++++++ 0.5+++++++++++% 0.0+++++++++% 0.000+++++++++++ 0.000+++++++++++
4        0.000+++++++ -0.3++++++++++% 0.0+++++++++% 0.000+++++++++++ 0.000+++++++++++
8        0.000+++++++ 0.2+++++++++++% 0.0+++++++++% 0.000+++++++++++ 0.000+++++++++++
16       -0.000++++++ -0.2++++++++++% 0.0+++++++++% 0.000+++++++++++ 0.000+++++++++++
24       -0.000++++++ 1.0+++++++++++% 0.0+++++++++% 0.000+++++++++++ 0.000+++++++++++
31       -0.000++++++ 11.5++++++++++% 0.0+++++++++% 0.000+++++++++++ 0.000+++++++++++

SVAMP Quantization Strategy Recommendations:
--------------------------------------------------------------------------------
✅ All layers show good robustness to quantization

================================================================================
COMPARATIVE QUANTIZATION IMPACT ANALYSIS
================================================================================
ARC-Easy Accuracy Drop (8-bit → 4-bit): 0.1%
SVAMP Accuracy Drop (8-bit → 4-bit): -23.7%

Task Difficulty Analysis:
- Multiple Choice (ARC) vs Math (SVAMP) reasoning patterns differ significantly
- ARC requires choice-selection attention, SVAMP needs numerical reasoning
