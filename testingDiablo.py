import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

class GradientAnalyzer:
    def __init__(self, model_name="microsoft/DialoGPT-small", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self._load_model(model_name)
        
    def _load_model(self, model_name):
        print(f"Loading {model_name} on {self.device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if self.device.type == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {total_params:,} parameters")
        return model, tokenizer
    
    def compute_gradients(self, input_ids, target_token_pos):
        self.model.eval()
        input_ids = input_ids.to(self.device)
        
        layer_gradients = {}
        
        def gradient_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output is not None:
                    grad = grad_output[0] if isinstance(grad_output, tuple) else grad_output
                    if grad is not None:
                        layer_gradients[name] = grad.norm().item()
            return hook
        
        handles = []
        
        # Register hooks based on architecture
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Llama-style architecture
            for i, layer in enumerate(self.model.model.layers):
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
                    handles.append(layer.self_attn.o_proj.register_full_backward_hook(
                        gradient_hook(f'layer_{i}_attn')))
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'down_proj'):
                    handles.append(layer.mlp.down_proj.register_full_backward_hook(
                        gradient_hook(f'layer_{i}_ffn')))
                        
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-style architecture
            for i, layer in enumerate(self.model.transformer.h):
                if hasattr(layer, 'attn'):
                    handles.append(layer.attn.register_full_backward_hook(
                        gradient_hook(f'layer_{i}_attn')))
                if hasattr(layer, 'mlp'):
                    handles.append(layer.mlp.register_full_backward_hook(
                        gradient_hook(f'layer_{i}_mlp')))
        else:
            # Generic approach
            layer_count = 0
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear) and 'lm_head' not in name:
                    handles.append(module.register_full_backward_hook(
                        gradient_hook(f'linear_{layer_count}')))
                    layer_count += 1
        
        try:
            self.model.zero_grad()
            outputs = self.model(input_ids)
            logits = outputs.logits
            
            seq_len = logits.shape[1]
            if target_token_pos < 0:
                target_token_pos = seq_len + target_token_pos
            target_token_pos = min(max(0, target_token_pos), seq_len - 1)
            
            loss = logits[0, target_token_pos, :].sum()
            loss.backward()
            
        finally:
            for handle in handles:
                handle.remove()
        
        return layer_gradients
    
    def analyze_text(self, text, target_pos=-1):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                              truncation=True, max_length=128)
        input_ids = inputs['input_ids']
        return self.compute_gradients(input_ids, target_pos)
    
    def analyze_dataset(self, questions, dataset_name):
        print(f"\nAnalyzing {dataset_name} dataset ({len(questions)} questions)")
        
        all_gradients = {}
        
        for i, question in enumerate(questions):
            print(f"Processing {i+1}/{len(questions)}: {question[:50]}...")
            gradients = self.analyze_text(question)
            
            for layer_name, grad_val in gradients.items():
                if layer_name not in all_gradients:
                    all_gradients[layer_name] = []
                all_gradients[layer_name].append(grad_val)
        
        # Calculate averages
        avg_gradients = {layer: np.mean(grads) for layer, grads in all_gradients.items()}
        std_gradients = {layer: np.std(grads) for layer, grads in all_gradients.items()}
        
        return avg_gradients, std_gradients
    
    def visualize_comparison(self, quant_avg, quant_std, qual_avg, qual_std):
        # Get all unique layers
        all_layers = set(quant_avg.keys()) | set(qual_avg.keys())
        
        # Separate by layer type
        attn_layers = sorted([int(name.split('_')[1]) for name in all_layers if 'attn' in name])
        ffn_layers = sorted([int(name.split('_')[1]) for name in all_layers if 'ffn' in name or 'mlp' in name])
        
        if attn_layers and ffn_layers:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Attention layers comparison
            quant_attn = [quant_avg.get(f'layer_{i}_attn', 0) for i in attn_layers]
            qual_attn = [qual_avg.get(f'layer_{i}_attn', 0) for i in attn_layers]
            quant_attn_std = [quant_std.get(f'layer_{i}_attn', 0) for i in attn_layers]
            qual_attn_std = [qual_std.get(f'layer_{i}_attn', 0) for i in attn_layers]
            
            x = np.arange(len(attn_layers))
            width = 0.35
            
            ax1.bar(x - width/2, quant_attn, width, alpha=0.7, color='skyblue', 
                   label='Quantitative', yerr=quant_attn_std, capsize=3)
            ax1.bar(x + width/2, qual_attn, width, alpha=0.7, color='lightcoral', 
                   label='Qualitative', yerr=qual_attn_std, capsize=3)
            ax1.set_title('Attention Layers: Quantitative vs Qualitative')
            ax1.set_xlabel('Layer Index')
            ax1.set_ylabel('Average Gradient Magnitude')
            ax1.set_xticks(x)
            ax1.set_xticklabels(attn_layers)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # FFN layers comparison
            quant_ffn = [quant_avg.get(f'layer_{i}_ffn', 0) or quant_avg.get(f'layer_{i}_mlp', 0) for i in ffn_layers]
            qual_ffn = [qual_avg.get(f'layer_{i}_ffn', 0) or qual_avg.get(f'layer_{i}_mlp', 0) for i in ffn_layers]
            quant_ffn_std = [quant_std.get(f'layer_{i}_ffn', 0) or quant_std.get(f'layer_{i}_mlp', 0) for i in ffn_layers]
            qual_ffn_std = [qual_std.get(f'layer_{i}_ffn', 0) or qual_std.get(f'layer_{i}_mlp', 0) for i in ffn_layers]
            
            ax2.bar(x - width/2, quant_ffn, width, alpha=0.7, color='skyblue', 
                   label='Quantitative', yerr=quant_ffn_std, capsize=3)
            ax2.bar(x + width/2, qual_ffn, width, alpha=0.7, color='lightcoral', 
                   label='Qualitative', yerr=qual_ffn_std, capsize=3)
            ax2.set_title('FFN/MLP Layers: Quantitative vs Qualitative')
            ax2.set_xlabel('Layer Index')
            ax2.set_ylabel('Average Gradient Magnitude')
            ax2.set_xticks(x)
            ax2.set_xticklabels(ffn_layers)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Overall comparison - Quantitative
            all_quant = list(quant_avg.values())
            all_qual = list(qual_avg.values())
            
            ax3.hist(all_quant, bins=20, alpha=0.7, color='skyblue', label='Quantitative')
            ax3.set_title('Quantitative Questions - Gradient Distribution')
            ax3.set_xlabel('Gradient Magnitude')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
            
            # Overall comparison - Qualitative
            ax4.hist(all_qual, bins=20, alpha=0.7, color='lightcoral', label='Qualitative')
            ax4.set_title('Qualitative Questions - Gradient Distribution')
            ax4.set_xlabel('Gradient Magnitude')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
            
        else:
            # Generic visualization for other architectures
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Side by side comparison
            layer_names = list(quant_avg.keys())
            quant_vals = [quant_avg[name] for name in layer_names]
            qual_vals = [qual_avg[name] for name in layer_names]
            
            x = np.arange(len(layer_names))
            width = 0.35
            
            ax1.bar(x - width/2, quant_vals, width, alpha=0.7, color='skyblue', label='Quantitative')
            ax1.bar(x + width/2, qual_vals, width, alpha=0.7, color='lightcoral', label='Qualitative')
            ax1.set_title('Layer Comparison: Quantitative vs Qualitative')
            ax1.set_xlabel('Layer')
            ax1.set_ylabel('Average Gradient Magnitude')
            ax1.set_xticks(x)
            ax1.set_xticklabels([name.replace('_', '\n') for name in layer_names], rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Distribution comparison
            ax2.hist(quant_vals, bins=15, alpha=0.7, color='skyblue', label='Quantitative')
            ax2.hist(qual_vals, bins=15, alpha=0.7, color='lightcoral', label='Qualitative')
            ax2.set_title('Gradient Distribution Comparison')
            ax2.set_xlabel('Gradient Magnitude')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def print_summary(self, quant_avg, qual_avg, dataset_names):
        print(f"\nSummary Statistics:")
        print("=" * 60)
        
        # Overall averages
        quant_mean = np.mean(list(quant_avg.values()))
        qual_mean = np.mean(list(qual_avg.values()))
        
        print(f"Overall Average Gradient Magnitude:")
        print(f"  {dataset_names[0]}: {quant_mean:.6f}")
        print(f"  {dataset_names[1]}: {qual_mean:.6f}")
        print(f"  Difference: {abs(quant_mean - qual_mean):.6f}")
        
        # Top layers for each type
        print(f"\nTop 3 Most Active Layers:")
        quant_top = sorted(quant_avg.items(), key=lambda x: x[1], reverse=True)[:3]
        qual_top = sorted(qual_avg.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"  {dataset_names[0]}:")
        for name, val in quant_top:
            print(f"    {name}: {val:.6f}")
        
        print(f"  {dataset_names[1]}:")
        for name, val in qual_top:
            print(f"    {name}: {val:.6f}")


def main():
    model_options = [
        "microsoft/DialoGPT-small",
        "gpt2",
        "distilgpt2",
        "microsoft/DialoGPT-medium",
    ]
    
    analyzer = None
    for model_name in model_options:
        try:
            analyzer = GradientAnalyzer(model_name)
            break
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
    
    if analyzer is None:
        print("Could not load any model")
        return
    
    # Define datasets
    quantitative_questions = [
        "What is 15 + 27?",
        "Calculate 144 divided by 12",
        "What is 8 squared?",
        "Find the product of 7 and 9",
        "What is 50% of 200?",
        "How many minutes in 3 hours?",
        "What is the square root of 64?",
        "Calculate 2^5",
        "What is 25 x 4?",
        "Find 80% of 150"
    ]
    
    qualitative_questions = [
        "What is the meaning of life?",
        "How do you feel about friendship?",
        "What makes a good leader?",
        "Why is creativity important?",
        "What defines happiness?",
        "How do cultures differ?",
        "What is the nature of consciousness?",
        "Why do people dream?",
        "What is love?",
        "How do we find purpose?"
    ]
    
    # Analyze datasets
    quant_avg, quant_std = analyzer.analyze_dataset(quantitative_questions, "Quantitative")
    qual_avg, qual_std = analyzer.analyze_dataset(qualitative_questions, "Qualitative")
    
    # Visualize comparison
    analyzer.visualize_comparison(quant_avg, quant_std, qual_avg, qual_std)
    
    # Print summary
    analyzer.print_summary(quant_avg, qual_avg, ["Quantitative", "Qualitative"])


if __name__ == "__main__":
    main()