def evaluate_svamp_answer(generated_answer, correct_answer):
    """Evaluate SVAMP answer by extracting final numerical value"""
    if not generated_answer or not correct_answer:
        return False
    
    try:
        correct_num = float(correct_answer.strip())
    except (ValueError, TypeError):
        return False
    
    # Extract generated answer using same logic as test_svamp_llama4bit.py
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

    for bits in [4, 2]:
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
import json
import os
import pickle
from datetime import datetime
warnings.filterwarnings("ignore")

# Create output directory for analysis data
output_dir = "quantization_analysis"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load datasets with SVAMP input filtering from test_svamp_llama4bit.py
print("Loading datasets...")
dataset = load_dataset("allenai/ai2_arc", 'ARC-Easy', split="train")
questions = [item['question'] for item in dataset]
answers = [item['answerKey'] for item in dataset]
print(f"Loaded ARC-Easy dataset: {len(questions)} samples")

# Apply SVAMP input filtering like in test_svamp_llama4bit.py
dataset2 = load_dataset("ChilleD/SVAMP", split="train")
# Combine Body and Question fields as done in test_svamp_llama4bit.py
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

class QuantizationAnalyzer:
    """Comprehensive quantization analysis with data recording"""
    
    def __init__(self, model_name, output_dir, timestamp):
        self.model_name = model_name
        self.output_dir = output_dir
        self.timestamp = timestamp
        self.attention_data = {}
        self.gradient_data = {}
        self.layer_activations = {}
        self.quantization_effects = {}
        self.hooks = []
        
    def setup_comprehensive_hooks(self, model, layer_indices=None, head_indices=None, record_gradients=True):
        """Setup hooks for attention, activations, and gradients"""
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
        
        if head_indices is None:
            head_indices = [0, 4, 8, 12, 16, 20, 24, 28]
        
        self.layer_indices = layer_indices
        self.head_indices = head_indices
        
        def create_forward_hook(layer_idx):
            def hook_fn(module, input, output):
                try:
                    # Record layer activations
                    if isinstance(output, tuple):
                        hidden_states = output[0].detach().cpu()
                    else:
                        hidden_states = output.detach().cpu()
                    
                    self.layer_activations[layer_idx] = {
                        'hidden_states': hidden_states,
                        'input_norm': torch.norm(input[0]).item() if input and len(input) > 0 else 0,
                        'output_norm': torch.norm(hidden_states).item(),
                        'mean_activation': hidden_states.mean().item(),
                        'std_activation': hidden_states.std().item(),
                        'sparsity': (hidden_states == 0).float().mean().item(),
                        'magnitude': torch.norm(hidden_states, dim=-1).mean().item(),
                    }
                    
                    # Record gradients if available
                    if record_gradients and hidden_states.requires_grad:
                        def grad_hook(grad):
                            self.gradient_data[f'layer_{layer_idx}_grad'] = {
                                'grad_norm': torch.norm(grad).item(),
                                'grad_mean': grad.mean().item(),
                                'grad_std': grad.std().item(),
                                'grad_sparsity': (grad == 0).float().mean().item()
                            }
                        hidden_states.register_hook(grad_hook)
                        
                except Exception as e:
                    print(f"Error in forward hook for layer {layer_idx}: {e}")
            return hook_fn
        
        def create_attention_hook(layer_idx):
            def hook_fn(module, input, output):
                try:
                    # Capture attention weights if available
                    if len(output) >= 2 and output[1] is not None:
                        attention_weights = output[1].detach().cpu()
                        
                        if attention_weights.dim() == 4:  # [batch, heads, seq_len, seq_len]
                            batch_size, num_heads, seq_len, _ = attention_weights.shape
                            
                            # Analyze attention patterns
                            attention_analysis = self.analyze_attention_comprehensive(
                                attention_weights, layer_idx, self.head_indices
                            )
                            
                            self.attention_data[f'layer_{layer_idx}'] = {
                                'attention_weights': attention_weights,
                                'analysis': attention_analysis,
                                'shape': attention_weights.shape,
                                'entropy': self.calculate_attention_entropy(attention_weights),
                                'concentration': self.calculate_attention_concentration(attention_weights),
                                'head_specialization': self.calculate_head_specialization(attention_weights)
                            }
                            
                except Exception as e:
                    print(f"Error in attention hook for layer {layer_idx}: {e}")
            return hook_fn
        
        # Register hooks
        for idx in layer_indices:
            if idx < len(model.model.layers):
                # Forward hook for layer
                layer_hook = model.model.layers[idx].register_forward_hook(create_forward_hook(idx))
                self.hooks.append(layer_hook)
                
                # Attention hook
                if hasattr(model.model.layers[idx], 'self_attn'):
                    attn_hook = model.model.layers[idx].self_attn.register_forward_hook(create_attention_hook(idx))
                    self.hooks.append(attn_hook)
        
        return self.layer_indices, self.head_indices
    
    def analyze_attention_comprehensive(self, attention_weights, layer_idx, head_indices):
        """Comprehensive attention pattern analysis"""
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        analysis = {}
        
        for head_idx in head_indices:
            if head_idx < num_heads:
                head_attn = attention_weights[:, head_idx, :, :]
                
                analysis[head_idx] = {
                    'entropy': float(-torch.sum(head_attn * torch.log(head_attn + 1e-12), dim=-1).mean()),
                    'self_attention_ratio': float(torch.diagonal(head_attn, dim1=-2, dim2=-1).mean()),
                    'last_token_attention': float(head_attn[:, -1, :].mean()),
                    'attention_span': self.calculate_attention_span(head_attn),
                    'attention_sharpness': float(torch.max(head_attn, dim=-1)[0].mean()),
                    'local_vs_global': self.calculate_local_global_ratio(head_attn),
                    'pattern_consistency': self.calculate_pattern_consistency(head_attn)
                }
        
        return analysis
    
    def calculate_attention_entropy(self, attention_weights):
        """Calculate entropy across all attention heads"""
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-12), dim=-1)
        return {
            'mean': float(entropy.mean()),
            'std': float(entropy.std()),
            'per_head_mean': entropy.mean(dim=(0, 2)).tolist(),
            'per_layer_mean': entropy.mean(dim=(1, 2)).tolist()
        }
    
    def calculate_attention_concentration(self, attention_weights):
        """Calculate how concentrated attention is"""
        max_attn = torch.max(attention_weights, dim=-1)[0]
        return {
            'mean_max_attention': float(max_attn.mean()),
            'std_max_attention': float(max_attn.std()),
            'concentration_per_head': max_attn.mean(dim=(0, 2)).tolist()
        }
    
    def calculate_head_specialization(self, attention_weights):
        """Calculate how specialized each attention head is"""
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Calculate variance across sequence positions for each head
        head_variance = torch.var(attention_weights, dim=-1).mean(dim=(0, 2))
        
        # Calculate correlation between heads
        head_correlations = []
        for i in range(num_heads):
            for j in range(i+1, num_heads):
                corr = torch.corrcoef(torch.stack([
                    attention_weights[:, i, :, :].flatten(),
                    attention_weights[:, j, :, :].flatten()
                ]))[0, 1]
                head_correlations.append(float(corr) if not torch.isnan(corr) else 0.0)
        
        return {
            'head_variance': head_variance.tolist(),
            'mean_head_correlation': np.mean(head_correlations) if head_correlations else 0.0,
            'specialization_score': float(head_variance.std())  # Higher std = more specialized heads
        }
    
    def calculate_attention_span(self, attention_weights):
        """Calculate effective attention span"""
        batch_size, seq_len, _ = attention_weights.shape
        positions = torch.arange(seq_len).float().unsqueeze(0).unsqueeze(0)
        
        weighted_positions = torch.sum(attention_weights * positions, dim=-1)
        query_positions = torch.arange(seq_len).float().unsqueeze(0)
        
        spans = query_positions - weighted_positions
        return float(spans.mean())
    
    def calculate_local_global_ratio(self, attention_weights):
        """Calculate ratio of local vs global attention"""
        seq_len = attention_weights.shape[-1]
        local_window = min(5, seq_len // 4)
        
        # Create local attention mask (within window)
        local_mask = torch.zeros_like(attention_weights)
        for i in range(seq_len):
            start = max(0, i - local_window)
            end = min(seq_len, i + local_window + 1)
            local_mask[:, i, start:end] = 1
        
        local_attention = (attention_weights * local_mask).sum(dim=-1)
        global_attention = attention_weights.sum(dim=-1) - local_attention
        
        ratio = local_attention / (global_attention + 1e-8)
        return float(ratio.mean())
    
    def calculate_pattern_consistency(self, attention_weights):
        """Calculate how consistent attention patterns are across batch"""
        if attention_weights.shape[0] > 1:
            # Calculate pairwise correlations across batch
            batch_correlations = []
            for i in range(attention_weights.shape[0]):
                for j in range(i+1, attention_weights.shape[0]):
                    corr = torch.corrcoef(torch.stack([
                        attention_weights[i].flatten(),
                        attention_weights[j].flatten()
                    ]))[0, 1]
                    if not torch.isnan(corr):
                        batch_correlations.append(float(corr))
            
            return np.mean(batch_correlations) if batch_correlations else 0.0
        return 1.0
    
    def save_analysis_data(self, bits, dataset_name, sample_idx, question, answer, generated_answer, is_correct):
        """Save comprehensive analysis data"""
        data = {
            'metadata': {
                'timestamp': self.timestamp,
                'bits': bits,
                'dataset': dataset_name,
                'sample_idx': sample_idx,
                'question': question,
                'correct_answer': answer,
                'generated_answer': generated_answer,
                'is_correct': is_correct
            },
            'attention_data': self.attention_data,
            'layer_activations': self.layer_activations,
            'gradient_data': self.gradient_data
        }
        
        # Save to file
        filename = f"{self.output_dir}/analysis_{dataset_name}_{bits}bit_sample_{sample_idx}_{self.timestamp}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        return filename
    
    def clear_data(self):
        """Clear stored data for next sample"""
        self.attention_data.clear()
        self.layer_activations.clear()
        self.gradient_data.clear()
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

def setup_attention_hooks(model, layer_indices=None, head_indices=None):
    """Legacy function - now uses QuantizationAnalyzer"""
    analyzer = QuantizationAnalyzer("legacy", output_dir, timestamp)
    layer_indices, head_indices = analyzer.setup_comprehensive_hooks(model, layer_indices, head_indices)
    return analyzer.attention_data, analyzer.layer_activations, analyzer.hooks, layer_indices, head_indices

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

def process_batch_with_layers(model, tokenizer, batch_questions, dataset_name, layer_indices, head_indices, batch_size=8, analyzer=None, sample_start_idx=0, bits=None):
    """Process batch with comprehensive layer and attention analysis"""
    try:
        # Create prompts - same system prompt as test_svamp_llama4bit.py for SVAMP
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
        
        # Use comprehensive analyzer if provided
        if analyzer:
            generated_answers = []
            analysis = {}
            
            for i, (prompt, question) in enumerate(zip(prompts, batch_questions)):
                # Setup hooks for this sample
                analyzer.clear_data()
                layer_indices, head_indices = analyzer.setup_comprehensive_hooks(model, layer_indices, head_indices)
                
                # Tokenize single prompt
                inputs = tokenizer(
                    [prompt], 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=400,
                    padding=True
                )
                
                if torch.cuda.is_available() and hasattr(model, 'device'):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate with hooks active
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
                
                # Decode response
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_answer = response[len(prompt):].strip()
                generated_answers.append(generated_answer)
                
                # Save comprehensive analysis data
                if bits is not None:
                    filename = analyzer.save_analysis_data(
                        bits, dataset_name, sample_start_idx + i, 
                        question, "", generated_answer, False  # We'll update correctness later
                    )
                    print(f"Saved analysis data to {filename}")
                
                # Remove hooks for this sample
                analyzer.remove_hooks()
                
                # Collect analysis for return
                analysis[sample_start_idx + i] = {
                    'attention_data': dict(analyzer.attention_data),
                    'layer_activations': dict(analyzer.layer_activations),
                    'gradient_data': dict(analyzer.gradient_data)
                }
            
            return generated_answers, analysis
        else:
            # Generate with comprehensive analysis (legacy mode)
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
    
    # Initialize comprehensive analyzer
    analyzer = QuantizationAnalyzer(model_name, output_dir, timestamp) if enable_layer_analysis else None
    
    try:
        # Configure quantization based on bits (changed from 8,4 to 4,2)
        if bits == 4:
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
                trust_remote_code=True,
                use_auth_token=True,
                low_cpu_mem_usage=True
            )
        elif bits == 2:
            # 2-bit quantization configuration
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,  # Use 4bit infrastructure for 2bit
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                use_auth_token=True,
                low_cpu_mem_usage=True
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
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize counters and layer tracking
        correct = 0
        total = min(num_samples, len(questions_list)) if num_samples > 0 else len(questions_list)
        errors = 0
        layer_stats = defaultdict(list)  # Collect stats for each layer
        
        # Limit comprehensive analysis to first 100 samples for detailed recording
        analysis_limit = min(100, total) if enable_layer_analysis else 0
        
        print(f"Processing {total} samples in batches of {batch_size}...")
        print(f"Comprehensive analysis will be saved for first {analysis_limit} samples")
        
        # Process samples in batches
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch_questions = questions_list[i:batch_end]
            batch_correct_answers = answers_list[i:batch_end] if i < len(answers_list) else [""] * len(batch_questions)
            
            try:
                if enable_layer_analysis and i < analysis_limit:
                    # Process batch with comprehensive analysis and data recording
                    batch_generated_answers, comprehensive_analysis = process_batch_with_layers(
                        model, tokenizer, batch_questions, dataset_name, layer_indices, head_indices, 
                        batch_size, analyzer, i, bits
                    )
                    
                    # Collect layer and attention statistics
                    for sample_idx, stats in comprehensive_analysis.items():
                        if 'layer_activations' in stats:
                            for layer_idx, layer_data in stats['layer_activations'].items():
                                layer_stats[layer_idx].append(layer_data)
                        
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
        
        # Save comprehensive experiment summary
        experiment_summary = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'bits': bits,
            'model_name': model_name,
            'total_samples': total,
            'correct': correct,
            'accuracy': accuracy,
            'errors': errors,
            'elapsed_time': elapsed_time,
            'layer_indices': layer_indices,
            'head_indices': head_indices,
            'analysis_limit': analysis_limit,
            'quantization_config': {
                'load_in_4bit': bits in [2, 4],
                'bnb_4bit_compute_dtype': 'float16',
                'bnb_4bit_use_double_quant': True,
                'bnb_4bit_quant_type': 'nf4'
            }
        }
        
        summary_file = f"{output_dir}/experiment_summary_{dataset_name}_{bits}bit_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(experiment_summary, f, indent=2)
        print(f"Saved experiment summary to {summary_file}")
        
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
                    'samples_analyzed': len(stats_list)
                }
                
                aggregated_layer_stats[layer_idx] = layer_data
        
        print(f"\n{dataset_name} - {bits}-bit Results:")
        print(f"  Total Samples: {total}")
        print(f"  Correct: {correct}")
        print(f"  Errors: {errors}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Time: {elapsed_time/60:.2f} minutes")
        print(f"  Avg Time/Sample: {elapsed_time/total:.2f}s")
        print(f"  Samples/Second: {total/elapsed_time:.2f}")
        print(f"  Comprehensive analysis saved for {analysis_limit} samples")
        
        # Cleanup analyzer
        if analyzer:
            analyzer.remove_hooks()
        
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
            'layer_stats': aggregated_layer_stats,
            'summary_file': summary_file,
            'analysis_samples': analysis_limit
        }
        
    except Exception as e:
        print(f"Fatal error in {bits}-bit {dataset_name}: {e}")
        # Clean up on error
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if analyzer:
            analyzer.remove_hooks()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            'accuracy': 0.0,
            'correct': 0,
            'total': 0,
            'errors': 1,
            'time': 0,
            'layer_stats': {},
            'summary_file': None,
            'analysis_samples': 0
        }

def process_batch_standard(model, tokenizer, batch_questions, dataset_name, batch_size=8):
    """Standard batch processing without layer analysis (faster)"""
    try:
        # Create prompts for the batch - same system prompt as test_svamp_llama4bit.py for SVAMP
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
    
    # Check if we have layer stats for both bit sizes (changed from 8,4 to 4,2)
    bit_sizes = [4, 2]
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
    print(f"{'Layer':<8} {'Bits':<6} {'Activation':<12} {'Sparsity':<10} {'Magnitude':<12} {'Samples':<8}")
    print("-"*70)
    
    for layer_idx in all_layers:
        for bits in bit_sizes:
            if bits in layer_data and layer_idx in layer_data[bits]:
                stats = layer_data[bits][layer_idx]
                print(f"{layer_idx:<8} {bits:<6} {stats['mean_activation']:<12.4f} "
                      f"{stats['sparsity']:<10.3f} {stats['magnitude']:<12.2f} "
                      f"{stats['samples_analyzed']:<8}")
        print()  # Empty line between layers
    
    # Calculate quantization impact metrics (changed from 8->4 to 4->2)
    print("\nQuantization Impact Analysis:")
    print("-"*80)
    print(f"{'Layer':<8} {'Sparsity Δ':<12} {'Magnitude Δ':<14} {'Activation Δ':<14}")
    print("-"*80)
    
    critical_layers = []
    
    for layer_idx in all_layers:
        if (4 in layer_data and layer_idx in layer_data[4] and 
            2 in layer_data and layer_idx in layer_data[2]):
            
            stats_4bit = layer_data[4][layer_idx]
            stats_2bit = layer_data[2][layer_idx]
            
            sparsity_delta = stats_2bit['sparsity'] - stats_4bit['sparsity']
            magnitude_delta = (stats_2bit['magnitude'] - stats_4bit['magnitude']) / stats_4bit['magnitude'] * 100 if stats_4bit['magnitude'] > 0 else 0
            activation_delta = (stats_2bit['mean_activation'] - stats_4bit['mean_activation']) / abs(stats_4bit['mean_activation']) * 100 if stats_4bit['mean_activation'] != 0 else 0
            
            print(f"{layer_idx:<8} {sparsity_delta:+<12.3f} {magnitude_delta:+<14.1f}% {activation_delta:+<14.1f}%")
            
            # Identify critical layers for quantization strategy
            if (abs(magnitude_delta) > 15 or abs(activation_delta) > 25 or abs(sparsity_delta) > 0.1):
                critical_layers.append((layer_idx, abs(magnitude_delta) + abs(activation_delta)))
    
    # Print analysis file locations
    print(f"\nDetailed Analysis Files:")
    for bits in bit_sizes:
        if bits in results_dict and 'summary_file' in results_dict[bits]:
            print(f"  {bits}-bit summary: {results_dict[bits]['summary_file']}")
            print(f"  {bits}-bit samples analyzed: {results_dict[bits].get('analysis_samples', 0)}")
    
    return critical_layers

# Main execution
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configuration - changed from [8, 4] to [4, 2]
    bit_sizes = [4, 2]
    num_samples = 1000  # Reduced for comprehensive analysis
    batch_size = 4      # Smaller batches for detailed recording
    enable_layer_analysis = True  # Enable detailed layer analysis
    
    print(f"\nConfiguration:")
    print(f"- Bit sizes: {bit_sizes}")
    print(f"- Samples per test: {num_samples}")
    print(f"- Batch size: {batch_size}")
    print(f"- Layer analysis: {'ENABLED' if enable_layer_analysis else 'DISABLED'}")
    print(f"- Output directory: {output_dir}")
    print(f"- Timestamp: {timestamp}")
    print(f"- Total ARC-Easy samples available: {len(questions)}")
    print(f"- Total SVAMP samples available: {len(questions2)}")
    print(f"- Comprehensive analysis will record first 100 samples per test")
    
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
    
    # Create final comprehensive summary
    final_summary = {
        'timestamp': timestamp,
        'configuration': {
            'bit_sizes': bit_sizes,
            'num_samples': num_samples,
            'batch_size': batch_size,
            'enable_layer_analysis': enable_layer_analysis
        },
        'arc_results': arc_results,
        'svamp_results': svamp_results,
        'critical_layers': {
            'arc': arc_critical_layers,
            'svamp': svamp_critical_layers
        },
        'output_directory': output_dir
    }
    
    final_summary_file = f"{output_dir}/final_comprehensive_summary_{timestamp}.json"
    with open(final_summary_file, 'w') as f:
        json.dump(final_summary, f, indent=2, default=str)
    
    print(f"\n🎯 COMPREHENSIVE ANALYSIS COMPLETE")
    print(f"📁 All analysis data saved in: {output_dir}")
    print(f"📊 Final summary: {final_summary_file}")
    print(f"🔍 Individual sample analyses available for detailed quantization study")
    print(f"💾 Total analysis files created: ~{len(bit_sizes) * 200} (100 samples × 2 datasets × 2 bit levels)")