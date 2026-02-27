import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

query = sys.argv[1] if len(sys.argv) > 1 else ""

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if not torch.backends.mps.is_available():
    print("ERROR: GPT-2 requires MPS on this machine (CPU forward crashes with Bus error)", file=sys.stderr)
    sys.exit(2)

device = torch.device("mps")
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to(device)
model.eval()

# ---- Grammar (v0) ----
# This is a *grammar-defined* decoder: we constrain generation to the language
# of valid traces (currently a single canonical trace language; extend by adding
# more productions -> more candidates).
OPS_V0 = [
    "LOAD 7/200",
    "MASK_BIT bit=2 val=1",
    "WITNESS_NEAREST target=7/200",
    "RETURN_SET",
]

# Candidates are full valid outputs in the trace language.
# Add more candidates as your grammar grows (params, JSON semtrace, etc).
CANDIDATES = [
    "\n".join(OPS_V0) + "\n",
]

prompt = (
    "You are a semantic trace generator.\n"
    "Convert the query into a sequence of operations.\n"
    "Output ONLY the trace, no commentary.\n\n"
    f"Query: {query}\n"
    "Trace:\n"
)

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
prompt_len = int(inputs["input_ids"].shape[1])

# Warm up MPS allocations (avoids placeholder storage errors on some ops)
with torch.no_grad():
    _ = model(**inputs)

# ---- Token-trie constraint (grammar-constrained decoding) ----
class TrieNode:
    __slots__ = ("children", "terminal")
    def __init__(self):
        self.children = {}   # token_id -> TrieNode
        self.terminal = False

def build_trie(seqs_token_ids):
    root = TrieNode()
    max_len = 0
    for seq in seqs_token_ids:
        max_len = max(max_len, len(seq))
        cur = root
        for tid in seq:
            nxt = cur.children.get(tid)
            if nxt is None:
                nxt = TrieNode()
                cur.children[tid] = nxt
            cur = nxt
        cur.terminal = True
    return root, max_len

seqs = [tokenizer.encode(s, add_special_tokens=False) for s in CANDIDATES]
trie_root, max_new = build_trie(seqs)

def allowed_tokens(batch_id, input_ids):
    # transformers may pass input_ids as 1D (seq_len,) or 2D (1, seq_len)
    if hasattr(input_ids, "dim") and int(input_ids.dim()) == 1:
        seq_len = int(input_ids.shape[0])
        full = input_ids.tolist()
    else:
        seq_len = int(input_ids.shape[-1])
        full = input_ids[0].tolist()

    gen_len = seq_len - prompt_len
    if gen_len < 0:
        gen_len = 0

    # Traverse trie with generated token ids so far
    gen_ids = full[prompt_len:prompt_len + gen_len]
    cur = trie_root
    for tid in gen_ids:
        nxt = cur.children.get(int(tid))
        if nxt is None:
            # No valid continuation: force EOS to terminate, then we will reject below.
            eos = tokenizer.eos_token_id
            return [int(eos)] if eos is not None else [0]
        cur = nxt

    # Allowed next tokens are the trie children at this prefix.
    if cur.children:
        return [int(t) for t in cur.children.keys()]

    # Prefix is complete; allow EOS only.
    eos = tokenizer.eos_token_id
    return [int(eos)] if eos is not None else [0]

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=False,
        num_beams=1,
        prefix_allowed_tokens_fn=allowed_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

gen_ids = out[0][prompt_len:]
text = tokenizer.decode(gen_ids, skip_special_tokens=True)

# Validate strictly against grammar language (NO fallback).
if text not in CANDIDATES:
    print("ERROR: generated output not in trace grammar language", file=sys.stderr)
    print("---- got ----", file=sys.stderr)
    print(repr(text), file=sys.stderr)
    print("---- expected one of ----", file=sys.stderr)
    for c in CANDIDATES:
        print(repr(c), file=sys.stderr)
    sys.exit(2)

# Emit newline-delimited ops (Rust side accepts lines OR JSON array).
for line in text.splitlines():
    if line.strip():
        print(line)
