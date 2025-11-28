# make_prompts.py
from datasets import load_dataset

# Load GSM8K, test split
ds = load_dataset("openai/gsm8k", "main", split="test")

# Take the first 25 questions as prompts
prompts = [ex["question"] for ex in ds.select(range(25))]

# Save them to a text file, separated by "\n\n---\n\n"
with open("prompts.txt", "w") as f:
    f.write("\n\n---\n\n".join(prompts))
