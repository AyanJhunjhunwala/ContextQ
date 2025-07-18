def evaluate_svamp_answer(generated_answer, correct_answer):
    """Evaluate SVAMP answer by extracting final numerical value"""
    if not generated_answer or not correct_answer:
        return False
    
    try:
        correct_num = float(correct_answer.strip())
    except (ValueError, TypeError):
        return False
    
    # Extract generated answer using same logic as GSM8K
    generated_num = extract_answer_from_text(generated_answer)
    
    if generated_num is None:
        return False
    
    # Allow small floating point differences
    return abs(generated_num - correct_num) < 0.01
def print_summary_table(results_dict, dataset_name):
    """Print a formatted summary table"""
    print(f"\n{dataset_name} Summary:")
    print("="*90)
    print(f"{'Bits':<6} {'Accuracy':<10} {'Correct':<8} {'Total':<7} {'Errors':<7} {'Time(m)':<8} {'Samples/s':<10}")
    print("-"*90)

    for bits in [8, 4]:
        if bits in results_dict:
            r = results_dict[bits]
            samples_per_sec = r['total'] / r['time'] if r['time'] > 0 else 0
            print(f"{bits:<6} {r['accuracy']:<10.4f} {r['correct']:<8} {r['total']:<7} "
                  f"{r['errors']:<7} {r['time']/60:<8.2f} {samples_per_sec:<10.2f}")
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import BitsAndBytesConfig
import re
import warnings
import time
import gc
import numpy as np
from collections import defaultdict
warnings.filterwarnings("ignore")

# Load datasets
print("Loading datasets...")
dataset = load_dataset("allenai/ai2_arc", 'ARC-Easy', split="train")
questions = [item['question'] for item in dataset]
answers = [item['answerKey'] for item in dataset]
print(f"Loaded ARC-Easy dataset: {len(questions)} samples")

dataset2 = load_dataset("ChilleD/SVAMP", split="train")
questions2 = [item['Body'] + " " + item['Question'] for item in dataset2]
answers2 = [str(item['Answer']) for item in dataset2]
print(f"Loaded SVAMP dataset: {len(questions2)} samples")

print("\nSample ARC-Easy questions:")
for i, q in enumerate(questions[:3]):
    print(f"{i+1}. {q}")

print("\nSample SVAMP questions:")
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
        f"option {correct_lower}",
        f"the correct answer is {correct_lower}",
        f"{correct_lower})"
    ]
    
    for pattern in choice_patterns:
        if pattern in generated_lower:
            return True
    
    return False

def setup_attention_hooks(model, layer_indices=None, head_indices=None):
    """Setup hooks to capture attention patterns and layer outputs"""
    if layer_indices is None:
        total_layers = len(model.model.layers)
        layer_indices = [
            0,                              # Input processing
            total_layers // 8,              # Early reasoning 
            total_layers // 4,              # Question understanding
            total_layers // 2,              # Core reasoning
            3 * total_layers // 4,          # Answer synthesis
            total_layers - 1                # Output generation
        ]
    
    # Default to analyzing key attention heads (often heads 0, 4, 8, 12 are important)
    if head_indices is None:
        head_indices = [0, 4, 8, 12, 16, 20]  # Strategic head selection
    
    attention_data = {}
    layer_outputs = {}
    hooks = []
    
    def create_attention_hook(layer_idx):
        def hook_fn(module, input, output):
            # Capture attention weights from self-attention
            if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'attention_weights'):
                attention_weights = module.self_attn.attention_weights
                if attention_weights is not None:
                    attention_data[f'layer_{layer_idx}_attention'] = attention_weights.detach().cpu()
            
            # Capture hidden states
            if isinstance(output, tuple):
                layer_outputs[layer_idx] = output[0].detach().cpu()
            else:
                layer_outputs[layer_idx] = output.detach().cpu()
        return hook_fn
    
    def create_self_attn_hook(layer_idx):
        def hook_fn(module, input, output):
            # Capture attention patterns from self-attention module
            if len(output) >= 2 and output[1] is not None:  # attention_weights exist
                attention_weights = output[1].detach().cpu()
                attention_data[f'layer_{layer_idx}_attention'] = attention_weights
        return hook_fn
    
    # Register hooks on transformer layers and attention modules
    for idx in layer_indices:
        if idx < len(model.model.layers):
            # Hook the transformer layer
            layer_hook = model.model.layers[idx].register_forward_hook(create_attention_hook(idx))
            hooks.append(layer_hook)
            
            # Hook the self-attention module directly
            if hasattr(model.model.layers[idx], 'self_attn'):
                attn_hook = model.model.layers[idx].self_attn.register_forward_hook(create_self_attn_hook(idx))
                hooks.append(attn_hook)
    
    return attention_data, layer_outputs, hooks, layer_indices, head_indices

def analyze_attention_patterns(attention_data, layer_outputs, tokenizer, input_ids, layer_indices, head_indices, dataset_name):
    """Analyze attention patterns and their relationship to reasoning quality"""
    analysis = {}
    
    for layer_idx in layer_indices:
        layer_analysis = {}
        
        # Analyze layer representations
        if layer_idx in layer_outputs:
            hidden_states = layer_outputs[layer_idx]
            batch_size, seq_len, hidden_size = hidden_states.shape
            last_token_repr = hidden_states[:, -1, :]
            
            layer_analysis.update({
                'mean_activation': float(last_token_repr.mean()),
                'std_activation': float(last_token_repr.std()),
                'sparsity': float((last_token_repr == 0).float().mean()),
                'magnitude': float(torch.norm(last_token_repr, dim=1).mean()),
                'representation_rank': estimate_rank(last_token_repr),
            })
        
        # Analyze attention patterns
        attention_key = f'layer_{layer_idx}_attention'
        if attention_key in attention_data:
            attention_weights = attention_data[attention_key]  # [batch, heads, seq_len, seq_len]
            
            if attention_weights.dim() == 4:
                batch_size, num_heads, seq_len, _ = attention_weights.shape
                
                # Focus on specific heads that are important for reasoning
                target_heads = [h for h in head_indices if h < num_heads]
                
                head_analyses = {}
                for head_idx in target_heads:
                    head_attn = attention_weights[:, head_idx, :, :]  # [batch, seq_len, seq_len]
                    
                    # Calculate attention metrics for this head
                    head_analyses[head_idx] = {
                        # Attention concentration (how focused vs diffuse)
                        'entropy': float(-torch.sum(head_attn * torch.log(head_attn + 1e-12), dim=-1).mean()),
                        
                        # Self-attention vs cross-attention patterns
                        'self_attention_ratio': float(torch.diagonal(head_attn, dim1=-2, dim2=-1).mean()),
                        
                        # Last token attention (important for generation)
                        'last_token_attention': float(head_attn[:, -1, :].mean()),
                        
                        # Attention span (how far back the model looks)
                        'attention_span': calculate_attention_span(head_attn),
                        
                        # Question-to-answer attention flow
                        'qa_attention_flow': calculate_qa_attention_flow(head_attn, seq_len, dataset_name),
                        
                        # Attention sharpness (peakiness of attention distribution)
                        'attention_sharpness': float(torch.max(head_attn, dim=-1)[0].mean()),
                    }
                
                layer_analysis['attention_heads'] = head_analyses
                
                # Layer-level attention aggregates
                layer_analysis.update({
                    'avg_attention_entropy': np.mean([h['entropy'] for h in head_analyses.values()]),
                    'avg_attention_sharpness': np.mean([h['attention_sharpness'] for h in head_analyses.values()]),
                    'avg_attention_span': np.mean([h['attention_span'] for h in head_analyses.values()]),
                    'reasoning_attention_score': calculate_reasoning_attention_score(head_analyses, dataset_name)
                })
        
        analysis[layer_idx] = layer_analysis
    
    return analysis

def calculate_attention_span(attention_weights):
    """Calculate the effective span of attention (how far back the model looks)"""
    batch_size, seq_len, _ = attention_weights.shape
    positions = torch.arange(seq_len).float().unsqueeze(0).unsqueeze(0)
    
    # Calculate weighted average position for each query
    weighted_positions = torch.sum(attention_weights * positions, dim=-1)
    query_positions = torch.arange(seq_len).float().unsqueeze(0)
    
    # Attention span as difference between query position and attended position
    spans = query_positions - weighted_positions
    return float(spans.mean())

def calculate_qa_attention_flow(attention_weights, seq_len, dataset_name):
    """Calculate how well attention flows from question to answer components"""
    if dataset_name == "ARC":
        # For ARC: Look for attention to choice tokens and reasoning words
        # Assume question is in first 70% of sequence, choices in last 30%
        question_end = int(seq_len * 0.7)
        choice_start = question_end
        
        # Attention from choice region back to question region
        qa_flow = attention_weights[:, choice_start:, :question_end].mean()
        
    else:  # SVAMP
        # For SVAMP: Look for attention to numbers and mathematical operations
        # Focus on last token attention to mathematical content (assumed early in sequence)
        math_region_end = int(seq_len * 0.6)
        qa_flow = attention_weights[:, -1, :math_region_end].mean()
    
    return float(qa_flow)

def calculate_reasoning_attention_score(head_analyses, dataset_name):
    """Calculate a composite score indicating reasoning quality of attention patterns"""
    if not head_analyses:
        return 0.0
    
    if dataset_name == "ARC":
        # For ARC: Reward focused attention, good QA flow, moderate entropy
        score = 0
        for head_data in head_analyses.values():
            # Balanced entropy (not too high, not too low)
            entropy_score = 1.0 - abs(head_data['entropy'] - 2.5) / 2.5
            # High QA attention flow
            qa_score = head_data['qa_attention_flow']
            # Sharp attention for choices
            sharpness_score = min(head_data['attention_sharpness'], 1.0)
            
            score += (entropy_score + qa_score + sharpness_score) / 3
            
    else:  # SVAMP
        # For SVAMP: Reward mathematical reasoning patterns
        score = 0
        for head_data in head_analyses.values():
            # Lower entropy for focused mathematical attention
            entropy_score = max(0, 1.0 - head_data['entropy'] / 3.0)
            # Good mathematical attention flow
            qa_score = head_data['qa_attention_flow']
            # Moderate attention span for multi-step problems
            span_score = 1.0 - abs(head_data['attention_span'] - 5.0) / 10.0
            
            score += (entropy_score + qa_score + span_score) / 3
    
    return score / len(head_analyses)

def estimate_rank(tensor, threshold=1e-6):
    """Estimate the effective rank of tensor representations"""
    try:
        # Use SVD to estimate rank
        U, S, V = torch.svd(tensor.float())
        rank = (S > threshold).sum().item()
        return rank
    except:
        return -1  # Return -1 if SVD fails

def remove_hooks(hooks):
    """Remove all registered hooks"""
    for hook in hooks:
        hook.remove()

def generate_with_analysis(model, tokenizer, prompts, layer_indices, head_indices, dataset_name):
    """Generate responses while capturing comprehensive layer and attention analysis"""
    try:
        # Setup hooks for both layers and attention
        attention_data, layer_outputs, hooks, actual_layer_indices, actual_head_indices = setup_attention_hooks(
            model, layer_indices, head_indices
        )
        
        # Tokenize batch
        inputs = tokenizer(
            prompts, 
            return_tensors='pt', 
            truncation=True, 
            max_length=400,
            padding=True,
            pad_to_multiple_of=8
        )
        
        # Move to device
        if torch.cuda.is_available() and hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Clear previous data
        attention_data.clear()
        layer_outputs.clear()
        
        # Generate with comprehensive capture
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=50,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                output_attentions=True,  # Enable attention output
                return_dict_in_generate=True
            )
        
        # Analyze patterns
        analysis = analyze_attention_patterns(
            attention_data, layer_outputs, tokenizer, inputs['input_ids'], 
            actual_layer_indices, actual_head_indices, dataset_name
        )
        
        # Decode responses
        generated_answers = []
        for i, output in enumerate(outputs.sequences):
            response = tokenizer.decode(output, skip_special_tokens=True)
            original_prompt = prompts[i]
            generated_answer = response[len(original_prompt):].strip()
            generated_answers.append(generated_answer)
        
        # Remove hooks
        remove_hooks(hooks)
        
        return generated_answers, analysis
        
    except Exception as e:
        print(f"Error in comprehensive analysis generation: {e}")
        if 'hooks' in locals():
            remove_hooks(hooks)
        return [None] * len(prompts), {}

def process_batch_with_layers(model, tokenizer, batch_questions, dataset_name, layer_indices, head_indices, batch_size=8):
    """Process batch with comprehensive layer and attention analysis"""
    try:
        # Create prompts
        prompts = []
        for question in batch_questions:
            if dataset_name == "ARC":
                prompt = f"Answer this multiple choice question by selecting the best option:\n\nQuestion: {question}\n\nAnswer:"
            else:  # SVAMP
                prompt = f"Solve this math problem step by step:\n\nQuestion: {question}\n\nAnswer:"
            prompts.append(prompt)
        
        # Generate with comprehensive analysis
        generated_answers, analysis = generate_with_analysis(
            model, tokenizer, prompts, layer_indices, head_indices, dataset_name
        )
        
        return generated_answers, analysis
        
    except Exception as e:
        print(f"Error processing batch with comprehensive analysis: {e}")
        return [None] * len(batch_questions), {}

def run_benchmark_with_quantization(questions_list, answers_list, num_samples, dataset_name, bits, batch_size=8, enable_layer_analysis=True):
    """Run benchmark with specified quantization level, batch processing, and layer analysis"""
    print(f"\nRunning {dataset_name} benchmark with {bits}-bit precision (batch size: {batch_size})...")
    if enable_layer_analysis:
        print("Layer-wise analysis: ENABLED")
    start_time = time.time()
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    try:
        # Configure quantization based on bits
        if bits == 8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,  # Use float16 for speed
            )
        elif bits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            raise ValueError(f"Unsupported bit size: {bits}")
        
        print(f"Model loaded successfully")
        
        # Determine layer indices for analysis (strategic selection)
        total_layers = len(model.model.layers)
        layer_indices = [
            0,                              # Input processing
            total_layers // 8,              # Early reasoning (12.5%)
            total_layers // 4,              # Question understanding (25%)
            total_layers // 2,              # Core reasoning (50%)
            3 * total_layers // 4,          # Answer synthesis (75%)
            total_layers - 1                # Output generation (100%)
        ]
        
        # Strategic attention head selection (often these heads specialize)
        head_indices = [0, 4, 8, 12, 16, 20, 24, 28]  # Key reasoning heads
        
        print(f"Total layers: {total_layers}, analyzing layers: {layer_indices}")
        print(f"Analyzing attention heads: {head_indices}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize counters and layer tracking
        correct = 0
        total = min(num_samples, len(questions_list)) if num_samples > 0 else len(questions_list)
        errors = 0
        layer_stats = defaultdict(list)  # Collect stats for each layer
        
        print(f"Processing {total} samples in batches of {batch_size}...")
        
        # Process samples in batches
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch_questions = questions_list[i:batch_end]
            batch_correct_answers = answers_list[i:batch_end] if i < len(answers_list) else [""] * len(batch_questions)
            
            try:
                if enable_layer_analysis and i < 500:  # Only analyze first 500 samples for speed
                    # Process batch with comprehensive analysis
                    batch_generated_answers, comprehensive_analysis = process_batch_with_layers(
                        model, tokenizer, batch_questions, dataset_name, layer_indices, head_indices, batch_size
                    )
                    
                    # Collect layer and attention statistics
                    for layer_idx, stats in comprehensive_analysis.items():
                        layer_stats[layer_idx].append(stats)
                        
                else:
                    # Process batch normally (faster)
                    batch_generated_answers = process_batch_standard(model, tokenizer, batch_questions, dataset_name, batch_size)
                
                # Evaluate batch results
                for j, (generated_answer, correct_answer) in enumerate(zip(batch_generated_answers, batch_correct_answers)):
                    if generated_answer is None:
                        errors += 1
                        continue
                    
                    # Evaluate answer
                    if dataset_name == "ARC":
                        is_correct = evaluate_arc_answer(generated_answer, correct_answer)
                    else:  # SVAMP
                        is_correct = evaluate_svamp_answer(generated_answer, correct_answer)
                    
                    if is_correct:
                        correct += 1
                        
            except Exception as batch_error:
                errors += len(batch_questions)
                if errors <= 20:  # Print first few batch errors
                    print(f"Error processing batch {i//batch_size + 1}: {batch_error}")
                continue
            
            # Progress updates
            processed = min(batch_end, total)
            if processed % (batch_size * 10) == 0 or processed == total:
                current_acc = correct / processed if processed > 0 else 0
                elapsed = time.time() - start_time
                avg_time = elapsed / processed
                eta = avg_time * (total - processed)
                samples_per_sec = processed / elapsed if elapsed > 0 else 0
                
                print(f"Progress: {processed}/{total} ({processed/total*100:.1f}%) | "
                      f"Accuracy: {current_acc:.3f} | "
                      f"Errors: {errors} | "
                      f"Speed: {samples_per_sec:.1f} samples/s | "
                      f"ETA: {eta/60:.1f}m")
        
        # Calculate final results
        accuracy = correct / total if total > 0 else 0.0
        elapsed_time = time.time() - start_time
        
        # Aggregate comprehensive statistics
        aggregated_layer_stats = {}
        for layer_idx, stats_list in layer_stats.items():
            if stats_list:
                # Basic layer stats
                layer_data = {
                    'mean_activation': np.mean([s.get('mean_activation', 0) for s in stats_list]),
                    'std_activation': np.mean([s.get('std_activation', 0) for s in stats_list]),
                    'sparsity': np.mean([s.get('sparsity', 0) for s in stats_list]),
                    'magnitude': np.mean([s.get('magnitude', 0) for s in stats_list]),
                    'representation_rank': np.mean([s.get('representation_rank', 0) for s in stats_list if s.get('representation_rank', 0) > 0]),
                    'samples_analyzed': len(stats_list)
                }
                
                # Attention-specific stats
                attention_entropies = [s.get('avg_attention_entropy', 0) for s in stats_list if 'avg_attention_entropy' in s]
                attention_sharpnesses = [s.get('avg_attention_sharpness', 0) for s in stats_list if 'avg_attention_sharpness' in s]
                attention_spans = [s.get('avg_attention_span', 0) for s in stats_list if 'avg_attention_span' in s]
                reasoning_scores = [s.get('reasoning_attention_score', 0) for s in stats_list if 'reasoning_attention_score' in s]
                
                if attention_entropies:
                    layer_data.update({
                        'avg_attention_entropy': np.mean(attention_entropies),
                        'avg_attention_sharpness': np.mean(attention_sharpnesses),
                        'avg_attention_span': np.mean(attention_spans),
                        'reasoning_attention_score': np.mean(reasoning_scores),
                        'attention_analyzed': True
                    })
                else:
                    layer_data['attention_analyzed'] = False
                
                aggregated_layer_stats[layer_idx] = layer_data
        
        print(f"\n{dataset_name} - {bits}-bit Results:")
        print(f"  Total Samples: {total}")
        print(f"  Correct: {correct}")
        print(f"  Errors: {errors}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Time: {elapsed_time/60:.2f} minutes")
        print(f"  Avg Time/Sample: {elapsed_time/total:.2f}s")
        print(f"  Samples/Second: {total/elapsed_time:.2f}")
        
        # Print comprehensive analysis summary
        if aggregated_layer_stats and enable_layer_analysis:
            print(f"\n  Comprehensive Analysis Summary:")
            print(f"  {'Layer':<8} {'Activation':<12} {'Sparsity':<10} {'Magnitude':<12} {'Rank':<8} {'AttnEntropy':<12} {'ReasonScore':<12}")
            print(f"  {'-'*85}")
            for layer_idx in sorted(aggregated_layer_stats.keys()):
                stats = aggregated_layer_stats[layer_idx]
                attn_entropy = stats.get('avg_attention_entropy', 0)
                reason_score = stats.get('reasoning_attention_score', 0)
                print(f"  {layer_idx:<8} {stats['mean_activation']:<12.4f} {stats['sparsity']:<10.3f} "
                      f"{stats['magnitude']:<12.2f} {stats['representation_rank']:<8.1f} "
                      f"{attn_entropy:<12.3f} {reason_score:<12.3f}")
        
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
            'time': elapsed_time,
            'layer_stats': aggregated_layer_stats
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
            'time': 0,
            'layer_stats': {}
        }

def process_batch_standard(model, tokenizer, batch_questions, dataset_name, batch_size=8):
    """Standard batch processing without layer analysis (faster)"""
    try:
        # Create prompts for the batch
        prompts = []
        for question in batch_questions:
            if dataset_name == "ARC":
                prompt = f"Answer this multiple choice question by selecting the best option:\n\nQuestion: {question}\n\nAnswer:"
            else:  # SVAMP
                prompt = f"Solve this math problem step by step:\n\nQuestion: {question}\n\nAnswer:"
            prompts.append(prompt)
        
        # Tokenize batch with padding
        inputs = tokenizer(
            prompts, 
            return_tensors='pt', 
            truncation=True, 
            max_length=400,
            padding=True,
            pad_to_multiple_of=8
        )
        
        # Move to device
        if torch.cuda.is_available() and hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate responses for the batch
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
        return [None] * len(batch_questions)

def print_layer_analysis(results_dict, dataset_name):
    """Print detailed layer and attention analysis comparison"""
    print(f"\n{dataset_name} Comprehensive Analysis:")
    print("="*120)
    
    # Check if we have layer stats for both bit sizes
    bit_sizes = [8, 4]
    layer_data = {}
    
    for bits in bit_sizes:
        if bits in results_dict and 'layer_stats' in results_dict[bits]:
            layer_data[bits] = results_dict[bits]['layer_stats']
    
    if not layer_data:
        print("No analysis data available.")
        return
    
    # Get all layer indices
    all_layers = set()
    for bits in layer_data:
        all_layers.update(layer_data[bits].keys())
    all_layers = sorted(all_layers)
    
    if not all_layers:
        print("No layer data found.")
        return
    
    # Print comprehensive comparison table
    print(f"{'Layer':<8} {'Bits':<6} {'Activation':<12} {'Sparsity':<10} {'Magnitude':<12} {'Rank':<8} {'AttnEntropy':<12} {'ReasonScore':<12}")
    print("-"*120)
    
    for layer_idx in all_layers:
        for bits in bit_sizes:
            if bits in layer_data and layer_idx in layer_data[bits]:
                stats = layer_data[bits][layer_idx]
                attn_entropy = stats.get('avg_attention_entropy', 0)
                reason_score = stats.get('reasoning_attention_score', 0)
                print(f"{layer_idx:<8} {bits:<6} {stats['mean_activation']:<12.4f} "
                      f"{stats['sparsity']:<10.3f} {stats['magnitude']:<12.2f} "
                      f"{stats['representation_rank']:<8.1f} {attn_entropy:<12.3f} {reason_score:<12.3f}")
        print()  # Empty line between layers
    
    # Calculate quantization impact metrics
    print("\nQuantization Impact Analysis:")
    print("-"*100)
    print(f"{'Layer':<8} {'Sparsity Δ':<12} {'Magnitude Δ':<14} {'Rank Δ':<12} {'AttnEntropy Δ':<16} {'ReasonScore Δ':<16}")
    print("-"*100)
    
    critical_layers = []
    
    for layer_idx in all_layers:
        if (8 in layer_data and layer_idx in layer_data[8] and 
            4 in layer_data and layer_idx in layer_data[4]):
            
            stats_8bit = layer_data[8][layer_idx]
            stats_4bit = layer_data[4][layer_idx]
            
            sparsity_delta = stats_4bit['sparsity'] - stats_8bit['sparsity']
            magnitude_delta = (stats_4bit['magnitude'] - stats_8bit['magnitude']) / stats_8bit['magnitude'] * 100 if stats_8bit['magnitude'] > 0 else 0
            rank_delta = (stats_4bit['representation_rank'] - stats_8bit['representation_rank']) / stats_8bit['representation_rank'] * 100 if stats_8bit['representation_rank'] > 0 else 0
            
            # Attention deltas
            attn_entropy_delta = stats_4bit.get('avg_attention_entropy', 0) - stats_8bit.get('avg_attention_entropy', 0)
            reason_score_delta = stats_4bit.get('reasoning_attention_score', 0) - stats_8bit.get('reasoning_attention_score', 0)
            
            print(f"{layer_idx:<8} {sparsity_delta:+<12.3f} {magnitude_delta:+<14.1f}% {rank_delta:+<12.1f}% "
                  f"{attn_entropy_delta:+<16.3f} {reason_score_delta:+<16.3f}")
            
            # Identify critical layers for quantization strategy
            if (abs(magnitude_delta) > 15 or abs(rank_delta) > 20 or 
                abs(reason_score_delta) > 0.1 or abs(attn_entropy_delta) > 0.5):
                critical_layers.append((layer_idx, abs(magnitude_delta) + abs(rank_delta) + abs(reason_score_delta)*100))
    
    # Quantization strategy recommendations
    print(f"\n{dataset_name} Quantization Strategy Recommendations:")
    print("-"*80)
    
    if critical_layers:
        critical_layers.sort(key=lambda x: x[1], reverse=True)
        most_critical = [layer for layer, _ in critical_layers[:3]]
        
        print(f"🔴 CRITICAL LAYERS (preserve in higher precision): {most_critical}")
        print(f"   These layers show severe degradation under 4-bit quantization")
        
        safe_layers = [layer for layer in all_layers if layer not in [l for l, _ in critical_layers]]
        if safe_layers:
            print(f"🟢 SAFE LAYERS (can use 4-bit): {safe_layers}")
            print(f"   These layers maintain performance under quantization")
        
        # Attention-specific recommendations
        if dataset_name == "ARC":
            print(f"\n📋 ARC-Specific Recommendations:")
            print(f"   - Focus on preserving choice-selection attention patterns")
            print(f"   - Monitor question-to-answer attention flow in critical layers")
            print(f"   - Consider mixed precision: 8-bit for layers {most_critical}, 4-bit for others")
        else:
            print(f"\n🔢 SVAMP-Specific Recommendations:")
            print(f"   - Preserve mathematical reasoning attention in critical layers")
            print(f"   - Maintain number-to-operation attention patterns")
            print(f"   - Mathematical reasoning most sensitive in layers {most_critical}")
        
    else:
        print("✅ All layers show good robustness to quantization")
        
    return critical_layers

# Main execution
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configuration
    bit_sizes = [8, 4]
    num_samples = 3000  # Process 3000 samples per dataset
    batch_size = 8      # Default batch size
    enable_layer_analysis = True  # Enable detailed layer analysis
    
    # Auto-adjust batch size based on GPU memory (optimized for 80GB)
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb >= 75:  # 80GB GPUs (H100, A100 80GB)
            batch_size = 32  # Reduced from 64 due to layer analysis overhead
        elif gpu_memory_gb >= 40:  # 48GB GPUs (A6000, RTX 6000 Ada)
            batch_size = 16
        elif gpu_memory_gb >= 24:  # 24GB GPUs (RTX 4090, A5000)
            batch_size = 12
        elif gpu_memory_gb >= 16:  # 16GB GPUs (RTX 4080, V100)
            batch_size = 8
        elif gpu_memory_gb >= 12:  # 12GB GPUs (RTX 4070 Ti)
            batch_size = 6
        else:                      # <12GB GPUs
            batch_size = 4
    
    print(f"\nConfiguration:")
    print(f"- Bit sizes: {bit_sizes}")
    print(f"- Samples per test: {num_samples}")
    print(f"- Batch size: {batch_size} (optimized for {gpu_memory_gb:.1f}GB GPU)")
    print(f"- Layer analysis: {'ENABLED' if enable_layer_analysis else 'DISABLED'}")
    print(f"- Total ARC-Easy samples available: {len(questions)}")
    print(f"- Total SVAMP samples available: {len(questions2)}")
    print(f"- Estimated batches per test: {(num_samples + batch_size - 1) // batch_size}")
    print(f"- Estimated total runtime: {(num_samples * 4) / (batch_size * 5):.1f} minutes")  # Adjusted for layer analysis
    
    # ARC Benchmarks
    print("\n" + "="*80)
    print("ARC-EASY BENCHMARKS (Multiple Choice Reasoning)")
    print("="*80)
    arc_results = {}
    
    for bits in bit_sizes:
        print(f"\n[{bits}-bit] Starting ARC benchmark...")
        result = run_benchmark_with_quantization(questions, answers, num_samples, "ARC", bits, batch_size, enable_layer_analysis)
        arc_results[bits] = result
        
        # Short break between runs
        time.sleep(2)
    
    # SVAMP Benchmarks
    print("\n" + "="*80)
    print("SVAMP BENCHMARKS (Elementary Mathematical Reasoning)")
    print("="*80)
    svamp_results = {}
    
    for bits in bit_sizes:
        print(f"\n[{bits}-bit] Starting SVAMP benchmark...")
        result = run_benchmark_with_quantization(questions2, answers2, num_samples, "SVAMP", bits, batch_size, enable_layer_analysis)
        svamp_results[bits] = result
        
        # Short break between runs
        time.sleep(2)
    
    # Final Summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    print_summary_table(arc_results, "ARC-Easy (Multiple Choice Reasoning)")
    print_summary_table(svamp_results, "SVAMP (Elementary Math Reasoning)")
    
    # Print detailed layer and attention analysis
    arc_critical_layers = print_layer_analysis(arc_results, "ARC-Easy")
    svamp_critical_layers = print_layer_analysis(svamp_results, "SVAMP")
    
    # Quantization Impact Analysis
    print("\n" + "="*80)
    print("COMPARATIVE QUANTIZATION IMPACT ANALYSIS")
    print("="*80)
    
    if 8 in arc_results and 4 in arc_results:
        arc_8_acc = arc_results[8]['accuracy']
        arc_4_acc = arc_results[4]['accuracy']
        arc_degradation = (arc_8_acc - arc_4_acc) / arc_8_acc * 100 if arc_8_acc > 0 else 0
        
        print(f"ARC-Easy Accuracy Drop (8-bit → 4-bit): {arc_degradation:.1f}%")

    if 8 in svamp_results and 4 in svamp_results:
        svamp_8_acc = svamp_results[8]['accuracy']
        svamp_4_acc = svamp_results[4]['accuracy']
        svamp_degradation = (svamp_8_acc - svamp_4_acc) / svamp_8_acc * 100 if svamp_8_acc > 0 else 0

        print(f"SVAMP Accuracy Drop (8-bit → 4-bit): {svamp_degradation:.1f}%")
    
    # Task-specific insights
    if 8 in arc_results and 8 in svamp_results:
        print(f"\nTask Difficulty Analysis:")
        print(f"- Multiple Choice (ARC) vs Math (SVAMP) reasoning patterns differ significantly")
        print(f"- ARC requires choice-selection attention, SVAMP needs numerical reasoning")
        
        # Cross-task critical layer analysis
        if arc_critical_layers and svamp_critical_layers:
            arc_critical_set = set([layer for layer, _ in arc_critical_layers])
            svamp_critical_set = set([layer for layer, _ in svamp_critical_layers])
            
            shared_critical = arc_critical_set.intersection(svamp_critical_set)
            arc_specific = arc_critical_set - svamp_critical_set
            svamp_specific = svamp_critical_set - arc_critical_set
            
            print(f"\nCross-Task Critical Layer Analysis:")
            print(f"🔴 Universally Critical Layers: {sorted(shared_critical)} (affect both tasks)")
            print(f"📋 ARC-Specific Critical Layers: {sorted(arc_specific)} (choice reasoning)")
            print(f"🔢 SVAMP-Specific Critical Layers: {sorted(svamp_specific)} (math reasoning)")
            
            print(f"\n💡 Optimal Quantization Strategy:")
            if shared_critical:
                print(f"   - Keep layers {sorted(shared_critical)} in 8-bit (universal importance)")
            if arc_specific:
                print(f"   - For ARC tasks: Also preserve layers {sorted(arc_specific)} in 8-bit")
            if svamp_specific:
                print(f"   - For Math tasks: Also preserve layers {sorted(svamp_specific)} in 8-bit")
            
            safe_layers = set(range(32)) - arc_critical_set - svamp_critical_set  # Assuming 32 layers
            if safe_layers:
                print(f"   - Safe for 4-bit quantization: {sorted(list(safe_layers)[:10])}... (others)")
    
    # Performance improvement summary
    total_time = sum(r['time'] for r in {**arc_results, **svamp_results}.values())
    total_samples = sum(r['total'] for r in {**arc_results, **svamp_results}.values())
    overall_speed = total_samples / total_time if total_time > 0 else 0
    
    print(f"\nPerformance Summary:")
    print(f"- Total samples processed: {total_samples}")
    print(f"- Total runtime: {total_time/60:.1f} minutes")
    print(f"- Overall speed: {overall_speed:.1f} samples/second")
    print(f"- Estimated speedup vs single processing: {batch_size * 0.5:.1f}x")  # Conservative estimate with analysis overhead