from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "meta-llama/Llama-3.1-8B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",             # Use NormalFloat 4-bit data type
    bnb_4bit_compute_dtype=torch.float16,   # Compute dtype for 4-bit base models
    bnb_4bit_use_double_quant=True,        # Use double quantization (nested)
    bnb_4bit_quant_storage=torch.uint8,   
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    
# Set pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with appropriate settings
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,

    torch_dtype=torch.float16,  # Use float16 for memory efficiency
    device_map="auto",          # Automatically distribute across available GPUs
    trust_remote_code=True,
    use_auth_token=True,
    low_cpu_mem_usage=True     # Reduce CPU memory usage during loading
)

def generate_response(prompt, max_length=512,temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            quantization_config=bnb_config,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Test prompt
    test_prompt = "What is the capital of France?"
    
    print(f"\nPrompt: {test_prompt}")
    print("Generating response...")
    
    response = generate_response(test_prompt)
    print(f"Response: {response}")
