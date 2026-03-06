# Certified Execution Traces for Language Grounding: A Neural Proposer over a Cryptographic Linguistic Tower

**Draft — March 2026**

---

## Abstract

We present the **LLM Nature Semantic Transformer**, a system that replaces probabilistic next-token generation with certified execution traces through a deterministic, cryptographically-verified linguistic universe. The architecture separates three concerns that are conflated in standard transformers: a neural **Proposer** that generates candidate operation sequences, a deterministic **Executor** that runs them, and an independent **Verifier** that certifies correctness via SHA-256 digest chains. The proposer -- a 21,904-parameter two-layer MLP exported to ONNX -- achieves 100% accuracy (528/528 ops) on the full 44-phoneme inventory, 98.3% accuracy under 5% positional noise, and 83.3% at 30% noise. Every forward pass is cryptographically verified; no floating-point comparison is used for identity or ordering. The system ships as a single Rust binary with a unified CLI: `tower query P` produces a verified NDJSON trace that any holder of the binary can independently certify. We document an architectural finding: the proposer and verifier have non-overlapping responsibilities, and attempts to conflate them degrade both. The full system comprises 322 passing tests, zero warnings, and a 17,230-record multi-phoneme training corpus with chain-hash continuity across phoneme boundaries.

---

## 1. Introduction

Large language models generate fluent, plausible text by predicting the next token given a context. This works remarkably well as a compression of human-generated text, but it has a structural failure mode: the model has no grounded semantic model that its outputs must satisfy. Hallucination is not a bug to be patched -- it is a necessary consequence of an architecture that conflates the generation of plausible continuations with the verification of truth.

The approach we take here is different. We build a **finite, certified semantic universe** -- a 7-layer linguistic tower from Phoneme to Discourse -- and train a small neural model to **propose execution traces** through it. Truth is not decided by the neural model. It is decided by deterministic execution followed by independent cryptographic replay. The neural model's job is to find the right path; the verifier's job is to certify it.

This separation is the central architectural commitment of the system. It has several consequences:

- The neural model can be small (21,904 parameters) because it navigates a structured space rather than modeling an open-ended distribution over tokens.
- Correctness is binary and auditable: a trace is either verified or not, with no degrees of plausibility.
- Any trace produced by the system can be independently verified by anyone with the binary, without access to the model or the original execution environment.
- Hallucination is structurally impossible: if the trace verifies, the result is correct; if it does not, the system reports failure.

The system is implemented in Rust (tower, verifier, executor) and Python (training), with the proposer exported to ONNX for cross-language deployment via the `ort` crate.

---

## 2. The Certified Linguistic Tower

### 2.1 Layer Structure

The tower comprises seven linguistic layers arranged in a strict hierarchy:

    Phoneme -> Syllable -> Morpheme -> Word -> Phrase -> Semantic -> Discourse

Each layer is a finite indexed set with a structural identity function -- a 32-bit signature derived from the element's position and properties in the layer. No floating-point comparison is used for identity. Two elements are the same if and only if their signatures are equal.

**Certified Attention.** Given a query signature and parameters (tau, k), the attention mechanism returns the top-k nearest elements by structural distance, together with a `result_digest` -- the SHA-256 hash of the attention output. The digest is deterministic: the same query always produces the same digest.

**Certified FFN Step.** A cross-layer feedforward operation that maps an element in layer L to its most structurally similar counterpart in layer L+1. The output includes a `step_digest` -- the SHA-256 hash of the FFN output.

### 2.2 The Forward Pass

A complete tower forward pass for phoneme index i with parameters (tau, k):

1. SELECT_UNIVERSE -- select the phoneme layer as the active universe.
2. WITNESS_NEAREST -- find the nearest phoneme to i in the signature index.
3. ATTEND -- certified attention within the phoneme layer.
4. FFN_STEP x6 -- six cross-layer transitions: Phoneme->Syllable->Morpheme->Word->Phrase->Semantic->Discourse.
5. PROJECT_LAYER -- project the discourse signature back to the semantic layer.
6. RETURN_SET -- collect the result set.
7. ACCEPT -- terminal acceptance with pass digest.

Each step advances a ProposerContext -- a running state that includes the chain_hash (a rolling SHA-256 of all step digests so far), the step_count, the active_layer, and optional pass_digest and witness_digest.

### 2.3 Tower Root

The tower root is the Merkle root of all layer manifests, stable across sessions:

    tower_root: 06d344512ff8de62c1b6b0cbefa1d31e5778a69e76f69ac55375a8632142bae4

---

## 3. The Proposer

### 3.1 Architecture

    Input (25) -> Linear(25->128) -> ReLU -> Linear(128->128) -> ReLU -> [op_head, tgt_head]
    op_head:  Linear(128->8)   # 8 op classes
    tgt_head: Linear(128->8)   # 8 target layer classes

Total parameters: 21,904. ONNX export size: 86 KB.

### 3.2 Feature Encoding

| Dims  | Feature        | Encoding                           |
|-------|----------------|------------------------------------|
| 0-11  | block_idx % 12 | one-hot (canonical pass position)  |
| 12-18 | active_layer   | one-hot (7 layers)                 |
| 19    | step_count     | normalized: step_count / 60.0      |
| 20-23 | tau bin        | one-hot over {0.5, 1.0, 2.0, 4.0} |
| 24    | top_k          | normalized: top_k / 10.0           |

The block_idx % 12 encoding maps extended multi-phoneme sequences (block_idx 0..59) onto the canonical 12-position one-hot, allowing the model to generalize to multi-phoneme sequences without feature space expansion.

### 3.3 Training

Phase 1 -- Imitation Learning on corpus_v3 (17,230 records, early stopping patience=8). Best validation accuracy: 99.48%.

Phase 2 -- Process Supervision was found to degrade the IL checkpoint from 99.48% to ~70% due to conflict between the noise curriculum and KL regularization. The IL checkpoint was retained as the production model.

### 3.4 Corpus

| Version | Records | Notes                                      |
|---------|---------|--------------------------------------------|
| v1      | ~4,259  | Single-phoneme, rule-based labels          |
| v2      | ~4,259  | Full 44-phoneme, certified digests         |
| v3      | 17,230  | Multi-phoneme seqs, 10,122 chain states    |
| v4      | 18,730  | + 5 rejection scenario types, 8.8% REJECT |

---

## 4. The Verifier

### 4.1 Digest Chain

At each step: chain_hash_{t+1} = SHA256(chain_hash_t || step_digest_t)

The pass_digest is the SHA-256 of the final chain state. Any modification to any step produces a different pass_digest with overwhelming probability.

### 4.2 Verification Protocol

1. Rebuild the tower and check tower_root matches.
2. Re-run tower.forward(ph_idx, tau, top_k) to obtain the certified pass.
3. Verify the forward pass internally.
4. Compare pass_digest from result.json against the live pass.
5. Compare each record's step_digest against the recomputed expected digest.
6. Verify terminal op is ACCEPT and exactly 12 blocks are present.

### 4.3 Adversarial Soundness

Nine attack vectors tested -- all 9/9 detected:

| Attack                          | Detected |
|---------------------------------|----------|
| Bit-flip step_digest[0]         | YES      |
| Bit-flip step_digest[5]         | YES      |
| Bit-flip step_digest[11]        | YES      |
| Wrong op at block 0             | YES      |
| Wrong op at block 3             | YES      |
| Swapped steps 0 and 1           | YES      |
| Truncated trace (9 of 12 steps) | YES      |
| Duplicated step 0               | YES      |
| All-zero digests                | YES      |

---

## 5. Results

### 5.1 Canonical Accuracy

| Metric                  | Value        |
|-------------------------|--------------|
| Correct ops             | 528 / 528    |
| Accuracy                | 100.0%       |
| Forward passes verified | 44 / 44      |
| Evaluation time         | 86.2 seconds |

### 5.2 Noise Robustness

| Noise | Accuracy | vs Stage 20 Baseline |
|-------|----------|----------------------|
| 0.00  | 100.0%   | --                   |
| 0.05  | 98.3%    | +1.7pp               |
| 0.10  | 95.0%    | +0.1pp               |
| 0.15  | 91.7%    | --                   |
| 0.20  | 91.7%    | +1.4pp               |
| 0.30  | 83.3%    | +6.6pp               |

The multi-phoneme curriculum (corpus_v3) improved noise robustness at all levels, with the largest gain at noise=0.30 (+6.6pp). Chain-hash diversity in training improves positional generalization.

### 5.3 Model Efficiency

100% canonical accuracy with 21,904 parameters and an 86 KB ONNX export. Efficiency is a direct consequence of architectural separation: the proposer navigates a structured finite space, not an open-ended token distribution.

---

## 6. Architectural Findings

### 6.1 Proposer-Verifier Separation

Attempts to train the proposer to predict REJECT failed at any class weight (4x, 7x, 8x, 12x). The ceiling was ~65-87% reject accuracy with corresponding non-reject degradation. The reason is structural: rejection scenarios share identical (block_idx, active_layer) features with valid ops. The proposer proposes the most likely correct op; the verifier rejects incorrect ones. These are non-overlapping functions.

### 6.2 Process Supervision Degradation

PS with noise curriculum (0->15% over 20 epochs) conflicted with KL regularization: high noise required large policy deviations from the reference model, but KL resisted them. Result: trapped policy, 99.48% -> 70% degradation. Fix: anneal KL weight as noise increases, or replace KL with L2 logit constraint.

### 6.3 Chain Hash as Verification Signal

The chain_hash contributes to the cryptographic spine rather than the neural feature vector. Cryptographic chain is for verification; position + layer is for proposing.

---

## 7. Limitations and Future Work

- PS failure: anneal KL weight with noise curriculum.
- Noise ceiling at 30%: wider noise curriculum or explicit position-uncertainty features.
- Rejection prediction: dedicated predictor using tower-derived semantic validity signals.
- Scaling: tower build time (17.4s) remains bottleneck for large inventories.
- Multi-layer queries: semantic/discourse-level queries not currently supported.

---

## 8. Conclusion

Key contributions:

1. Certified tower: 7-layer linguistic universe with SHA-256 digest chains; every pass independently verifiable.
2. Proposer: 21,904-parameter MLP, 100% canonical accuracy, 98.3% at noise=0.05.
3. Separation principle: proposer and verifier have non-overlapping responsibilities.
4. Audit trail: every query produces a timestamped NDJSON trace independently certifiable without access to original execution.
5. CLI: unified tower binary with query/verify/eval/corpus/list subcommands.

---

## Appendix A: Stage Progression

| Stage | Description                   | Gate               |
|-------|-------------------------------|--------------------|
| 0-17  | Tower, corpus v1/v2, IL+RL+PS | 1.000 accuracy     |
| 18    | ONNX + Rust ORT               | 528/528            |
| 19    | Full inventory validation     | 528/528            |
| 20    | Adversarial stress            | 9/9 verifier       |
| 21    | Warning cleanup               | 0 warnings, 322 tests |
| 22    | Corpus v3 multi-phoneme       | 17,230 records     |
| 23    | Curriculum retrain            | 99.48% val         |
| 24    | Rejection corpus v4           | Architectural finding |
| 25    | Live execution                | tower query P      |
| 26    | Trace verification            | tower verify       |
| 27    | Unified CLI                   | All subcommands    |
| 28    | Benchmark                     | benchmark.json     |
| 29    | Paper                         | PAPER.md           |

---

## Appendix B: Reproducibility

    cargo run --bin benchmark
    cargo run --bin tower -- query P
    cargo run --bin tower -- verify runs/$(ls -t runs/ | head -1)/
    source train/.venv/bin/activate && python3 train/train_v3.py

Tower root 06d344512ff8de62... is stable across builds and platforms.

---

Tower root: 06d344512ff8de62c1b6b0cbefa1d31e5778a69e76f69ac55375a8632142bae4
Model: train/model_v3.onnx -- 86 KB, 21,904 parameters
Benchmark: 528/528 (100.0%), 44/44 verified, 98.3%@noise=0.05
