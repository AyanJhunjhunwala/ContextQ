from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import BitsAndBytesConfig
import re
import warnings
import time
import gc
warnings.filterwarnings("ignore")

# Load datasets
print("Loading datasets...")
dataset = load_dataset("openai/gsm8k", 'main', split="train")
questions = [item['question'] for item in dataset]
answers = [item['answer'] for item in dataset]
print(f"Loaded GSM8K dataset: {len(questions)} samples")

dataset2 = load_dataset("allenai/ai2_arc", 'ARC-Easy', split="train")
questions2 = [item['question'] for item in dataset2]
answers2 = [item['answerKey'] for item in dataset2]
print(f"Loaded ARC dataset: {len(questions2)} samples")

print("\nSample GSM8K questions:")
for i, q in enumerate(questions[:3]):
    print(f"{i+1}. {q}")

print("\nSample ARC questions:")
for i, q in enumerate(questions2[:3]):
    print(f"{i+1}. {q}")

def extract_answer_from_text(text):
    """Extract numerical answer from generated text"""
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
    
    # Fallback: find all numbers and take the last one
    numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None

def evaluate_gsm8k_answer(generated_answer, correct_answer):
    """Evaluate GSM8K answer by extracting final numerical value"""
    # Extract correct answer
    correct_num = re.findall(r'#### ([+-]?\d+(?:\.\d+)?)', correct_answer)
    if not correct_num:
        return False
    
    try:
        correct_num = float(correct_num[0])
    except (ValueError, IndexError):
        return False
    
    # Extract generated answer
    generated_num = extract_answer_from_text(generated_answer)
    
    if generated_num is None:
        return False
    
    # Allow small floating point differences
    return abs(generated_num - correct_num) < 0.01

def evaluate_arc_answer(generated_answer, correct_answer):
    """Evaluate ARC answer by looking for the correct choice"""
    if not generated_answer or not correct_answer:
        return False
    
    generated_lower = generated_answer.lower().strip()
    correct_lower = correct_answer.lower().strip()
    
    # Direct match
    if correct_lower in generated_lower:
        return True
    
    # Look for choice indicators
    choice_patterns = [
        f"answer is {correct_lower}",
        f"answer: {correct_lower}",
        f"({correct_lower})",
        f"choice {correct_lower}",
        f"option {correct_lower}"
    ]
    
    for pattern in choice_patterns:
        if pattern in generated_lower:
            return True
    
    return False

def run_benchmark_with_quantization(questions_list, answers_list, num_samples, dataset_name, bits):
    """Run benchmark with specified quantization level"""
    print(f"\nRunning {dataset_name} benchmark with {bits}-bit precision...")
    start_time = time.time()
    
    model_name = "microsoft/DialoGPT-small"
    
    try:
        # Configure quantization based on bits
        print(f"Loading model with {bits}-bit quantization...")
        if bits == 32:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto"
            )
        elif bits == 16:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        elif bits == 8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        elif bits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # DOUBLE QUANTIZATION ENABLED
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            raise ValueError(f"Unsupported bit size: {bits}")
        
        print(f"Model loaded successfully")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize counters
        correct = 0
        total = min(num_samples, len(questions_list)) if num_samples > 0 else len(questions_list)
        errors = 0
        
        print(f"Processing {total} samples...")
        
        # Process samples
        for i in range(total):
            try:
                question = questions_list[i]
                correct_answer = answers_list[i] if i < len(answers_list) else ""
                
                # Create appropriate prompt
                if dataset_name == "GSM8K":
                    prompt = f"Solve this math problem step by step:\n\nQuestion: {question}\n\nAnswer:"
                else:
                    prompt = f"Answer this question by choosing the best option:\n\nQuestion: {question}\n\nAnswer:"
                
                # Tokenize
                inputs = tokenizer(
                    prompt, 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=400,
                    padding=False
                )
                
                # Move to device
                if torch.cuda.is_available() and hasattr(model, 'device'):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate response
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs.get('attention_mask'),
                        max_new_tokens=75,
                        num_return_sequences=1, 
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    
                    # Decode response
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    original_prompt = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
                    generated_answer = response[len(original_prompt):].strip()
                    
                    # Evaluate answer
                    if dataset_name == "GSM8K":
                        is_correct = evaluate_gsm8k_answer(generated_answer, correct_answer)
                    else:
                        is_correct = evaluate_arc_answer(generated_answer, correct_answer)
                    
                    if is_correct:
                        correct += 1
                        
            except Exception as gen_error:
                errors += 1
                if errors <= 5:  # Only print first few errors
                    print(f"Error processing sample {i}: {gen_error}")
                continue
            
            # Progress updates
            if (i + 1) % 100 == 0 or (i + 1) == total:
                current_acc = correct / (i + 1)
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                eta = avg_time * (total - i - 1)
                print(f"Progress: {i + 1}/{total} ({(i+1)/total*100:.1f}%) | "
                      f"Accuracy: {current_acc:.3f} | "
                      f"Errors: {errors} | "
                      f"ETA: {eta/60:.1f}m")
        
        # Calculate final results
        accuracy = correct / total if total > 0 else 0.0
        elapsed_time = time.time() - start_time
        
        print(f"\n{dataset_name} - {bits}-bit Results:")
        print(f"  Total Samples: {total}")
        print(f"  Correct: {correct}")
        print(f"  Errors: {errors}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Time: {elapsed_time/60:.2f} minutes")
        print(f"  Avg Time/Sample: {elapsed_time/total:.2f}s")
        
        # Memory cleanup
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"  GPU Memory Cleared")
            
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'errors': errors,
            'time': elapsed_time
        }
        
    except Exception as e:
        print(f"Fatal error in {bits}-bit {dataset_name}: {e}")
        # Clean up on error
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            'accuracy': 0.0,
            'correct': 0,
            'total': 0,
            'errors': 1,
            'time': 0
        }

def print_summary_table(results_dict, dataset_name):
    """Print a formatted summary table"""
    print(f"\n{dataset_name} Summary:")
    print("="*80)
    print(f"{'Bits':<6} {'Accuracy':<10} {'Correct':<8} {'Total':<7} {'Errors':<7} {'Time(m)':<8} {'Samples/s':<10}")
    print("-"*80)
    
    for bits in [32, 16, 8, 4]:
        if bits in results_dict:
            r = results_dict[bits]
            samples_per_sec = r['total'] / r['time'] if r['time'] > 0 else 0
            print(f"{bits:<6} {r['accuracy']:<10.4f} {r['correct']:<8} {r['total']:<7} "
                  f"{r['errors']:<7} {r['time']/60:<8.2f} {samples_per_sec:<10.2f}")

# Main execution
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configuration - set num_samples to 0 to run ALL data
    bit_sizes = [32, 16, 8, 4]
    num_samples = 0  # 0 = run all data, or set to specific number for testing
    
    print(f"\nConfiguration:")
    print(f"- Bit sizes: {bit_sizes}")
    print(f"- Samples per test: {'ALL' if num_samples == 0 else num_samples}")
    print(f"- Total GSM8K samples: {len(questions)}")
    print(f"- Total ARC samples: {len(questions2)}")
    
    # GSM8K Benchmarks
    print("\n" + "="*80)
    print("GSM8K BENCHMARKS (Mathematical Reasoning)")
    print("="*80)
    gsm8k_results = {}
    
    for bits in bit_sizes:
        print(f"\n[{bits}-bit] Starting GSM8K benchmark...")
        result = run_benchmark_with_quantization(questions, answers, num_samples, "GSM8K", bits)
        gsm8k_results[bits] = result
        
        # Short break between runs
        time.sleep(2)
    
    # ARC Benchmarks
    print("\n" + "="*80)
    print("AI2 ARC BENCHMARKS (Multiple Choice Questions)")
    print("="*80)
    arc_results = {}
    
    for bits in bit_sizes:
        print(f"\n[{bits}-bit] Starting ARC benchmark...")
        result = run_benchmark_with_quantization(questions2, answers2, num_samples, "ARC", bits)
        arc_results[bits] = result
        
        # Short break between runs
        time.sleep(2)
    
    # Final Summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    print_summary_table(gsm8k_results, "GSM8K (Math Reasoning)")
    print_summary_table(arc_results, "ARC (Multiple Choice)")
    
    # Quantization Impact Analysis
    print("\n" + "="*80)
    print("QUANTIZATION IMPACT ANALYSIS")
    print("="*80)
    
    if 32 in gsm8k_results and 4 in gsm8k_results:
        gsm8k_32_acc = gsm8k_results[32]['accuracy']
        gsm8k_4_acc = gsm8k_results[4]['accuracy']
        gsm8k_degradation = (gsm8k_32_acc - gsm8k_4_acc) / gsm8k_32_acc * 100 if gsm8k_32_acc > 0 else 0
        
        print(f"GSM8K Accuracy Drop (32-bit → 4-bit): {gsm8k_degradation:.1f}%")
    
    if 32 in arc_results and 4 in arc_results:
        arc_32_acc = arc_results[32]['accuracy']
        arc_4_acc = arc_results[4]['accuracy']
        arc_degradation = (arc_32_acc - arc_4_acc) / arc_32_acc * 100 if arc_32_acc > 0 else 0
        
        print(f"ARC Accuracy Drop (32-bit → 4-bit): {arc_degradation:.1f}%")
    
    print("\nDouble Quantization Notes:")
    print("- 4-bit mode uses bnb_4bit_use_double_quant=True")
    print("- This quantizes both weights AND quantization parameters")
    print("- Provides maximum memory compression with minimal accuracy loss")
    print("- DialoGPT is not optimized for math/reasoning tasks")
    print("- For better math performance, consider models like Flan-T5 or Llama-2")
    
    print(f"\nBenchmark completed! Total runtime: {sum(r['time'] for r in {**gsm8k_results, **arc_results}.values())/60:.1f} minutes")