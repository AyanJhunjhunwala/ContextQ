from testingDialo import GradientAnalyzer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import BitsAndBytesConfig

dataset = load_dataset("openai/gsm8k", 'main', split="train")
questions = [item['question'] for item in dataset]
print(questions[:5])

dataset2= load_dataset("allenai/ai2_arc",'ARC-Easy', split="train")
questions2 = [item['question'] for item in dataset2]


# analyzer = GradientAnalyzer("microsoft/DialoGPT-small")


# quant_avg, quant_std = analyzer.analyze_dataset(questions, "Quantitative")
# qual_avg, qual_std = analyzer.analyze_dataset(questions2, "Qualitative")

# analyzer.visualize_comparison(quant_avg, quant_std, qual_avg, qual_std)


# analyzer.print_summary(quant_avg, qual_avg, ["Quantitative(GSM8K)", "Qualitative(AI2ARC)"])