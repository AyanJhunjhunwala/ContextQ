from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.gptq import GPTQQuantizer
import torch
import os

model_name = "facebook/opt-125m"

if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if os.path.exists("opt-125m-gptq"):
        print("Loading quantized model...")
        model = AutoModelForCausalLM.from_pretrained(
            "opt-125m-gptq",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("opt-125m-gptq")
        
        input_text = "The future of AI is"
        inputs = tokenizer(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {response}")
        
    else:
        print("Loading original model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        calibration_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
            "Neural networks are inspired by biological neural networks.",
            "Deep learning has revolutionized computer vision and NLP."
        ]
        
        print("Creating quantizer...")
        quantizer = GPTQQuantizer(
            bits=4,
            dataset=calibration_texts,
            block_name_to_quantize="model.decoder.layers", 
            model_seqlen=512
        )
        
        print("Quantizing model...")
        quantized_model = quantizer.quantize_model(model, tokenizer)
        
        print("Saving quantized model...")
        quantized_model.save_pretrained("opt-125m-gptq")
        tokenizer.save_pretrained("opt-125m-gptq")
        
        print("Quantization complete!")
        
        input_text = "The future of AI is"
        inputs = tokenizer(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        outputs = quantized_model.generate(**inputs, max_new_tokens=50, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {response}")
