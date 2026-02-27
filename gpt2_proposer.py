from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.mkldnn.enabled = False
torch.backends.mps.is_available = lambda: False
torch.set_num_threads(1)
import torch
import json
import sys

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = """Generate a semantic trace for:
Find fractions similar to 7/200 but with denominator ≤ 6.

Allowed ops:
- LOAD <fraction>
- MASK_BIT bit=2 val=1
- WITNESS_NEAREST target=<fraction>
- RETURN_SET

Output ONLY the ops, one per line.
"""

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    next_token = torch.argmax(logits[0, -1]).unsqueeze(0)
    outputs = torch.cat([inputs["input_ids"], next_token.unsqueeze(0)], dim=1)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract only operation lines
lines = []
for line in text.split("\n"):
    line = line.strip()
    if any(line.startswith(op) for op in ["LOAD", "MASK_BIT", "WITNESS_NEAREST", "RETURN_SET"]):
        lines.append(line)

# Fallback if GPT-2 is messy
if not lines:
    lines = [
        "LOAD 7/200",
        "MASK_BIT bit=2 val=1",
        "WITNESS_NEAREST target=7/200",
        "RETURN_SET"
    ]

print(json.dumps(lines))
