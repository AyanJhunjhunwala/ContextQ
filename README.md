# Context-Aware Model Wrapper for Selective Quantization/Pruning and editing Attention Patterns

# Is the wrapper out? -> Yes!
# pip install contextq
give it a try
# Update : August 2nd
The wrapper is out!
Gradient-aware (importance-driven) 4-bit quantization workflow for any causal-LM on Hugging Face, defaulting to Llama-3.2-1B-Instruct.

I am finishing this up on a late Saturday so I do have some messy code in there, but I will clean it up.
Currently, it calibrates on wikitexts only, I plan to allow any custom HGF dataset soon. bf16->4 with atleast 50% of layers being quantized and from full precision to multi precision(32,16,8,4) and I will try 3-2 bit quant as well in the future.

run selective_gpt --help for help!


# 13th July 2025

The attention patterns and gradient magnitude are obviously pretty different for quant/qual in dialo-small. I am planning to test this with a larger model. I tested out dialo with the ARC and GSM8k test. (below)



The wrapper is officially out, all it does right now is quantize 
# 17th July 2025

I am thinking of using github blogs for this but probably won't. So I ran the llama 3.1 8b instruct on ARC and svamp to split qualitative and quantitative and now my biggest question is how do I modify patterns and quantize the models to make this make sense. Svamp had a low accuracy and optmizing that without SFT would be something I guess. I started figruing out the pypi library as well but my main focus would have to be on extrapolating 

Benchmarks were ok...

# 21 July 2025

I have a roadmap! I will be implementing multi AWQ managed models and they will be able to be quantized more agressively. I will then implement a lightweight classifier that will figure out pre determined categories(much later obv)
