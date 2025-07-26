import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import re
import warnings

warnings.filterwarnings("ignore")

def extract_answer_from_text(text):
    patterns = [
        r'(?:the answer is|answer:|=|equals)\s*([+-]?\d+(?:\.\d+)?)',
        r'([+-]?\d+(?:\.\d+)?)\s*(?:is the answer|$)',
        r'(?:therefore|so|thus),?\s*([+-]?\d+(?:\.\d+)?)',
        r'answer\s*:\s*([+-]?\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*$'  # Number at end of text
    ]
    text = text.lower().strip()
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue
    numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    return None

def evaluate_svamp_answer(generated_answer, correct_answer):
    if not generated_answer or not correct_answer:
        return False
    try:
        correct_num = float(correct_answer.strip())
    except (ValueError, TypeError):
        return False
    generated_num = extract_answer_from_text(generated_answer)
    if generated_num is None:
        return False
    return abs(generated_num - correct_num) < 0.01

def main():
    print("Loading SVAMP dataset...")
    dataset = load_dataset("ChilleD/SVAMP", split="train")
    questions = [item['Body'] + " " + item['Question'] for item in dataset][:10]
    answers = [str(item['Answer']) for item in dataset][:10]

    print("\nLoading 4-bit Llama 3.1 Instruct model...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_auth_token=True,
        low_cpu_mem_usage=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    correct = 0
    print("\nTesting on 10 SVAMP samples:")
    system_prompt = "You are a helpful assistant. Only give me the answer, not the reasoning."
    for i, (q, gold) in enumerate(zip(questions, answers)):
        prompt = (
            f"<s>[INST] <<SYS>> {system_prompt} <</SYS>>\n"
            f"Solve this math problem step by step:\n\nQuestion: {q}\n\nAnswer: [/INST]"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=50,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):].strip()
        is_correct = evaluate_svamp_answer(generated, gold)
        print(f"\nSample {i+1}:")
        print(f"Q: {q}")
        print(f"Gold: {gold}")
        print(f"Model: {generated}")
        print(f"Correct: {is_correct}")
        if is_correct:
            correct += 1
    print(f"\nAccuracy: {correct}/10 = {correct/10:.2f}")

if __name__ == "__main__":
    main() 