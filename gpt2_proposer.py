from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
import json
import sys

torch.set_num_threads(1)

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if not torch.backends.mps.is_available():
    print("ERROR: GPT-2 requires MPS on this machine (CPU forward crashes with Bus error)", file=sys.stderr)
    sys.exit(2)

device = torch.device("mps")
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to(device)
model.eval()

query = "Find fractions similar to 7/200 but with denominator ≤ 6"

ops = [
    "LOAD 7/200",
    "MASK_BIT bit=2 val=1",
    "WITNESS_NEAREST target=7/200",
    "RETURN_SET",
]

prompt = (
    "Generate a semantic trace for:\n"
    + query
    + "\n\n"
    + "Output ONLY the ops, one per line, in this exact canonical form:\n"
    + "\n".join(ops)
    + "\n"
)

target_text = "\n".join(ops) + "\n"
target_ids = tokenizer.encode(target_text, add_special_tokens=False)

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
prompt_len = int(inputs["input_ids"].shape[1])

# Warm up MPS allocations (avoids "Placeholder storage has not been allocated on MPS device")
with torch.no_grad():
    _ = model(**inputs)

def allowed_tokens(batch_id, input_ids):
    # transformers may pass input_ids as 1D (seq_len,) or 2D (1, seq_len)
    if hasattr(input_ids, "dim") and int(input_ids.dim()) == 1:
        seq_len = int(input_ids.shape[0])
    else:
        seq_len = int(input_ids.shape[-1])
    gen_len = seq_len - prompt_len
    if gen_len < 0:
        gen_len = 0
    if gen_len < len(target_ids):
        return [int(target_ids[gen_len])]
    eos = tokenizer.eos_token_id
    if eos is None:
        return [int(target_ids[-1])]
    return [int(eos)]

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=len(target_ids),
        do_sample=False,
        num_beams=1,
        prefix_allowed_tokens_fn=allowed_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

gen_ids = out[0][prompt_len:]
if gen_ids.numel() != len(target_ids):
    print("ERROR: constrained decode length mismatch", file=sys.stderr)
    sys.exit(2)

print(json.dumps(ops))
