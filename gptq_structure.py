from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import os

model_id = "meta-llama/Llama-3.1-8B"
quant_path = "Quantized_Llama-3.1-8B"

if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(quant_path):
        print(f"Quantizing '{model_id}'...")

        print("Loading calibration dataset 'allenai/c4'...")
        calibration_dataset = load_dataset(
            "allenai/c4",
            data_files="en/c4-train.00001-of-01024.json.gz",
            split="train"
        ).select(range(1024))["text"]

        quant_config = QuantizeConfig(bits=4, group_size=128)

        print("Loading base model...")
        model = GPTQModel.load(model_id, quant_config,trust_remote_code=True)

        print("Starting quantization... this may take a while.")
        model.quantize(calibration_dataset, batch_size=1) 
        print("Quantization complete.")

        model.save(quant_path)
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(quant_path)
        print(f"Model and tokenizer saved to '{quant_path}'")

    print(f"\nLoading quantized model from '{quant_path}' for inference...")
    tokenizer = AutoTokenizer.from_pretrained(quant_path)
    
    model = GPTQModel.load(quant_path, device_map="auto", trust_remote_code=True)

    print("Running example inference... 🤖")
    prompt = "The capital of France is"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
    
    print(f"\nPrompt: {prompt}")
    print(f"Response: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
