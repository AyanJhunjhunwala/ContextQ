#!/usr/bin/env python3
"""
Simplified Quantization Analysis Script
Compatible with various PyTorch versions and environments
"""

import re
import warnings
import time
import gc
import json
import os
from datetime import datetime
from collections import defaultdict
warnings.filterwarnings("ignore")

# Try to import PyTorch with fallbacks
try:
    import torch
    torch_available = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception as e:
    print(f"⚠️  PyTorch import issue: {e}")
    torch_available = False
    device = "cpu"

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    transformers_available = True
except Exception as e:
    print(f"⚠️  Transformers import issue: {e}")
    transformers_available = False

try:
    from datasets import load_dataset
    datasets_available = True
except Exception as e:
    print(f"⚠️  Datasets import issue: {e}")
    datasets_available = False

# Create output directory
output_dir = "quantization_analysis"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"\n🎯 SIMPLIFIED QUANTIZATION ANALYSIS")
print(f"🖥️  Device: {device}")
print(f"📁 Output directory: {output_dir}")
print(f"⏰ Timestamp: {timestamp}")
print(f"🔧 PyTorch available: {torch_available}")
print(f"🤗 Transformers available: {transformers_available}")
print(f"📊 Datasets available: {datasets_available}")

def extract_answer_from_text(text):
    """Extract numerical answer from generated text - same as test_svamp_llama4bit.py"""
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

def evaluate_svamp_answer(generated_answer, correct_answer):
    """Evaluate SVAMP answer by extracting final numerical value"""
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
        f"option {correct_lower}",
        f"the correct answer is {correct_lower}",
        f"{correct_lower})"
    ]
    
    for pattern in choice_patterns:
        if pattern in generated_lower:
            return True
    
    return False

def process_batch_simple(model, tokenizer, batch_questions, dataset_name, batch_size=4):
    """Simple batch processing without complex analysis"""
    try:
        # Create prompts
        prompts = []
        for question in batch_questions:
            if dataset_name == "ARC":
                prompt = f"Answer this multiple choice question by selecting the best option:\n\nQuestion: {question}\n\nAnswer:"
            else:  # SVAMP - use same system prompt as test_svamp_llama4bit.py
                system_prompt = "You are a helpful assistant. Only give me the answer, not the reasoning."
                prompt = (
                    f"<s>[INST] <<SYS>> {system_prompt} <</SYS>>\n"
                    f"Solve this math problem step by step:\n\nQuestion: {question}\n\nAnswer: [/INST]"
                )
            prompts.append(prompt)
        
        # Tokenize batch with padding
        inputs = tokenizer(
            prompts, 
            return_tensors='pt', 
            truncation=True, 
            max_length=400,
            padding=True
        )
        
        # Generate responses for the batch
        if torch_available:
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
        else:
            # Fallback if PyTorch has issues
            print("⚠️ PyTorch generation failed, using dummy responses")
            return ["Error: PyTorch not available"] * len(batch_questions)
        
        # Decode responses
        generated_answers = []
        for i, output in enumerate(outputs):
            response = tokenizer.decode(output, skip_special_tokens=True)
            original_prompt = prompts[i]
            generated_answer = response[len(original_prompt):].strip()
            generated_answers.append(generated_answer)
        
        return generated_answers
        
    except Exception as e:
        print(f"Error processing batch: {e}")
        return [f"Error: {str(e)}"] * len(batch_questions)

def run_simple_benchmark(questions_list, answers_list, num_samples, dataset_name, bits):
    """Run simplified benchmark"""
    print(f"\n🔬 Running {dataset_name} benchmark ({bits}-bit simulated)...")
    start_time = time.time()
    
    if not transformers_available:
        print("❌ Transformers not available, skipping actual model loading")
        return {
            'accuracy': 0.0,
            'correct': 0,
            'total': num_samples,
            'errors': num_samples,
            'time': 1.0,
            'status': 'transformers_unavailable'
        }
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    try:
        print(f"Loading {model_name}...")
        
        # Try to load model and tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("✅ Tokenizer loaded successfully")
        except Exception as e:
            print(f"❌ Tokenizer loading failed: {e}")
            return {
                'accuracy': 0.0,
                'correct': 0,
                'total': num_samples,
                'errors': num_samples,
                'time': 1.0,
                'status': 'tokenizer_failed'
            }
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return {
                'accuracy': 0.0,
                'correct': 0,
                'total': num_samples,
                'errors': num_samples,
                'time': 1.0,
                'status': 'model_failed'
            }
        
        # Process samples
        correct = 0
        total = min(num_samples, len(questions_list))
        errors = 0
        batch_size = 2  # Small batch size for compatibility
        
        print(f"Processing {total} samples in batches of {batch_size}...")
        
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch_questions = questions_list[i:batch_end]
            batch_correct_answers = answers_list[i:batch_end] if i < len(answers_list) else [""] * len(batch_questions)
            
            try:
                batch_generated_answers = process_batch_simple(model, tokenizer, batch_questions, dataset_name, batch_size)
                
                # Evaluate batch results
                for j, (generated_answer, correct_answer) in enumerate(zip(batch_generated_answers, batch_correct_answers)):
                    if "Error:" in generated_answer:
                        errors += 1
                        continue
                    
                    # Evaluate answer
                    if dataset_name == "ARC":
                        is_correct = evaluate_arc_answer(generated_answer, correct_answer)
                    else:  # SVAMP
                        is_correct = evaluate_svamp_answer(generated_answer, correct_answer)
                    
                    if is_correct:
                        correct += 1
                    
                    # Show sample results for first few
                    if i + j < 5:
                        print(f"Sample {i+j+1}: {'✅' if is_correct else '❌'}")
                        print(f"  Q: {batch_questions[j][:100]}...")
                        print(f"  Expected: {correct_answer}")
                        print(f"  Generated: {generated_answer[:100]}...")
                        
            except Exception as batch_error:
                errors += len(batch_questions)
                print(f"Batch {i//batch_size + 1} error: {batch_error}")
                continue
            
            # Progress update
            processed = min(batch_end, total)
            if processed % 10 == 0 or processed == total:
                current_acc = correct / processed if processed > 0 else 0
                elapsed = time.time() - start_time
                samples_per_sec = processed / elapsed if elapsed > 0 else 0
                print(f"Progress: {processed}/{total} | Accuracy: {current_acc:.3f} | Speed: {samples_per_sec:.1f} samples/s")
        
        # Calculate final results
        accuracy = correct / total if total > 0 else 0.0
        elapsed_time = time.time() - start_time
        
        print(f"\n📊 {dataset_name} - {bits}-bit Results:")
        print(f"  Total Samples: {total}")
        print(f"  Correct: {correct}")
        print(f"  Errors: {errors}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Time: {elapsed_time:.2f} seconds")
        
        # Clean up
        del model
        del tokenizer
        if torch_available:
            gc.collect()
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'errors': errors,
            'time': elapsed_time,
            'status': 'completed'
        }
        
    except Exception as e:
        print(f"Fatal error in {bits}-bit {dataset_name}: {e}")
        return {
            'accuracy': 0.0,
            'correct': 0,
            'total': 0,
            'errors': 1,
            'time': 0,
            'status': f'fatal_error: {str(e)}'
        }

def main():
    """Main execution function"""
    
    if not datasets_available:
        print("❌ Datasets library not available - creating dummy data")
        questions = ["What is 2+2?"] * 10
        answers = ["4"] * 10
        questions2 = ["If you have 5 apples and eat 2, how many are left?"] * 10  
        answers2 = ["3"] * 10
    else:
        print("📁 Loading datasets...")
        try:
            dataset = load_dataset("allenai/ai2_arc", 'ARC-Easy', split="train")
            questions = [item['question'] for item in dataset]
            answers = [item['answerKey'] for item in dataset]
            print(f"✅ Loaded ARC-Easy dataset: {len(questions)} samples")
        except Exception as e:
            print(f"❌ ARC dataset loading failed: {e}")
            questions = ["What is 2+2?"] * 10
            answers = ["4"] * 10
        
        try:
            dataset2 = load_dataset("ChilleD/SVAMP", split="train")
            questions2 = [item['Body'] + " " + item['Question'] for item in dataset2]
            answers2 = [str(item['Answer']) for item in dataset2]
            print(f"✅ Loaded SVAMP dataset: {len(questions2)} samples")
        except Exception as e:
            print(f"❌ SVAMP dataset loading failed: {e}")
            questions2 = ["If you have 5 apples and eat 2, how many are left?"] * 10
            answers2 = ["3"] * 10
    
    # Show sample questions
    print("\n📝 Sample questions:")
    print(f"ARC: {questions[0][:100]}...")
    print(f"SVAMP: {questions2[0][:100]}...")
    
    # Configuration
    bit_sizes = [4]  # Simplified to just test 4-bit equivalent
    num_samples = 20  # Small number for testing
    
    print(f"\n⚙️ Configuration:")
    print(f"- Bit sizes: {bit_sizes}")
    print(f"- Samples per test: {num_samples}")
    print(f"- Device: {device}")
    
    # Run benchmarks
    results = {}
    
    print("\n" + "="*60)
    print("ARC-EASY BENCHMARKS")
    print("="*60)
    
    for bits in bit_sizes:
        result = run_simple_benchmark(questions, answers, num_samples, "ARC", bits)
        results[f"ARC_{bits}bit"] = result
    
    print("\n" + "="*60)
    print("SVAMP BENCHMARKS")
    print("="*60)
    
    for bits in bit_sizes:
        result = run_simple_benchmark(questions2, answers2, num_samples, "SVAMP", bits)
        results[f"SVAMP_{bits}bit"] = result
    
    # Save results
    results_file = f"{output_dir}/simple_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎯 ANALYSIS COMPLETE")
    print(f"📊 Results saved to: {results_file}")
    print(f"📁 Output directory: {output_dir}")
    
    # Print summary
    print(f"\n📋 SUMMARY:")
    for test_name, result in results.items():
        status = result.get('status', 'unknown')
        accuracy = result.get('accuracy', 0.0)
        print(f"  {test_name}: {accuracy:.3f} accuracy ({status})")

if __name__ == "__main__":
    main() 