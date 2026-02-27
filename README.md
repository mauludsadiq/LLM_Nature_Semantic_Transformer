# LLM Nature — Semantic Transformer (Grounded Proposer/Executor/Verifier)

This repo implements the **Semantic Transformer pivot**: the model is not a probabilistic truth-teller. It is a probabilistic **Proposer** of *execution traces* through a **certified finite semantic universe**. Truth is decided only by deterministic execution and replay verification.

In standard transformers, “reasoning” is next-token mimicry: the model emits a plausible continuation of text. That architecture has a structural failure mode: **hallucination** (plausible nonsense) because there is no grounded model that must be satisfied.

This project replaces that with a **PEV loop**:

- **Proposer (Transformer / GPT‑2)** proposes a *trace* (a sequence of discrete semantic operations).
- **Executor (Finite Model Runtime)** executes the trace against a certified universe and emits **step digests**.
- **Verifier (Replay Auditor)** re-executes every step and checks that all digests match.  
  If any step is inconsistent, the trace is **invalid** and is rejected.

In this system, hallucination is not “wrong output.” It is an **invalid trace**, detected and rejected by deterministic verification.

---

## What this project is derived from

This code is derived from the “Grounded Transformers / Semantic Engine” design built around:

1. **A grounded universe** of rational numbers (QE) with a certified enumeration.
2. **A fixed predicate family** that maps each element to a short semantic signature (7 bits).
3. **A quotient partition** into realized semantic classes (55 realized signatures out of 128 possible).
4. **Trace execution** with step digests (state hashes) and a chain hash anchor.

This repo implements the minimal, executable v0 of that architecture:
- deterministic QE construction (size 48,927),
- 7-bit semantic signatures,
- a minimal SemTrace instruction set,
- executor output as NDJSON with step digests,
- verifier replay.

---

## The problem it solves

**Problem:** LLMs generate text that *sounds right* but is not guaranteed to be *true* or even *coherent* under any grounded model. Confidence scores are not proof.

**Solution:** Move semantics out of the transformer and into a certified state space. The model proposes *paths*; the runtime and verifier determine validity. Outputs are accompanied by **cryptographic evidence** (digests), not confidence.

Concretely, we guarantee:

- **Determinism:** same universe + same trace ⇒ same outputs, byte-for-byte.
- **No hallucination:** any claim must correspond to a valid trace; invalid traces are rejected.
- **Replayability:** verification is re-execution; there is no trust in the executor output.
- **Human-legible artifacts:** each run writes a timestamped folder containing a paragraph artifact, trace NDJSON, digests, and a proof certificate.

---

## The certified universe (QE)

QE is built exactly as:

- denominators `b ∈ [1, 200]`
- numerators `a ∈ [-200, 200]`
- reduce each fraction to lowest terms
- take the set of unique reduced fractions

This produces **48,927** unique rationals (matching the certified run you posted), with the maximum value `200/1`.

Predicates (bit order, 0..6):

0. `positive`      : numerator > 0
1. `integer`       : denominator == 1
2. `den<=6`        : denominator <= 6
3. `num_even`      : numerator % 2 == 0
4. `den_mod3`      : denominator % 3 == 0
5. `proper`        : |numerator| < denominator
6. `num_abs<=5`    : |numerator| <= 5

The semantic signature of a rational is the 7-bit vector produced by these predicates.

---

## What “Semantic Transformer” means here

The transformer is allowed to output **only traces**. A trace is a small DSL over semantic constraints, for example:

- START from an element (e.g. `7/200`)
- force a predicate bit (e.g. `den<=6 = 1`)
- query the universe for the matching set
- pick a witness using a canonical nearest rule
- return a deterministic paragraph artifact

In v0, the included proposer is a deterministic stub that always produces a valid trace for a single fixed paragraph (so you can see the system “never waver”).
The integration point for GPT‑2 is explicit: GPT‑2 proposes the trace text; the executor/verifier accept or reject.

---

## Local setup (VS Code)

### 1) Clone and build

Open a terminal:

```bash
git clone <YOUR_REPO_URL_HERE> "LLM Nature Semantic Transformer"
cd "LLM Nature Semantic Transformer"
cargo build
```

### 2) Run a deterministic demo (writes a timestamped run folder)

```bash
cargo run -- demo
```

This creates:

- `runs/YYYYMMDD_HHMMSSZ/paragraph.txt` (the human-legible artifact)
- `runs/YYYYMMDD_HHMMSSZ/trace.ndjson`   (step-by-step execution trace)
- `runs/YYYYMMDD_HHMMSSZ/result.json`    (final result summary)
- `runs/YYYYMMDD_HHMMSSZ/proof.json`     (verification certificate)
- `runs/YYYYMMDD_HHMMSSZ/digests.json`   (domain/tests/trace chain hashes)

### 3) Verify an existing run

Copy the `trace.ndjson` path from the run folder and execute:

```bash
cargo run -- verify --trace runs/YYYYMMDD_HHMMSSZ/trace.ndjson
```

Exit code:
- `0` = VALID
- `1` = INVALID

---

## Running in a separate terminal

Terminal A (run demo):

```bash
cargo run -- demo
```

Terminal B (verify latest run):

```bash
cargo run -- verify --trace runs/<PASTE_LATEST_FOLDER>/trace.ndjson
```

---

## GPT‑2 integration (proposer)

This repo is ready for GPT‑2 as the **trace proposer**. The rule is simple:

> GPT‑2 may propose any trace string, but only traces that verify are accepted.

Practical options:
- Export GPT‑2 to ONNX and call it from Rust (onnxruntime / ort).
- Run GPT‑2 in a separate process (Python) and pipe the proposed trace into `cargo run -- exec`.

This v0 repo ships a deterministic proposer (`demo`) so you can validate the PEV pipeline without external model files.

---

## Notes on determinism

- No floating point is used for ordering or distance.
- Nearest witness selection uses exact rational distance:
  |a/b - c/d| = |ad - bc| / (bd)
  compared by cross-multiplication.

This ensures Rust and any other reference executor will agree exactly.

---

## Repo layout

- `src/` — executor, verifier, QE universe, predicates, digests
- `artifact/paragraph.txt` — the “never wavering” paragraph
- `runs/` — timestamped run outputs (created at runtime)

---

## License

MIT OR Apache-2.0
