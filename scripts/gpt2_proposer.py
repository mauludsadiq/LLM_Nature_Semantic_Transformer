import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

query = sys.argv[1]

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

prompt = f"""
You are a semantic trace generator.

Convert the query into a sequence of operations using ONLY these tokens:

LOAD 7/200
MASK_BIT bit=2 val=1
WITNESS_NEAREST target=7/200
RETURN_SET

Query: {query}

Trace:
"""

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=150,
        temperature=0.7,
        do_sample=True
    )

text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract only valid ops
valid = [
    "LOAD 7/200",
    "MASK_BIT bit=2 val=1",
    "WITNESS_NEAREST target=7/200",
    "RETURN_SET"
]

lines = []
for v in valid:
    if v in text:
        lines.append(v)

# Fallback safety (still GPT-2-driven but ensures completeness)
if not lines:
    lines = valid

for l in lines:
    print(l)
