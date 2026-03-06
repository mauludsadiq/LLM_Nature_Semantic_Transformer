# LLM Nature — Semantic Transformer

## Grounded Proposer / Executor / Verifier over a Certified 7-Layer Linguistic Universe

[![tests](https://img.shields.io/badge/tests-322%20passing-brightgreen)]()
[![accuracy](https://img.shields.io/badge/accuracy-528%2F528%20100%25-brightgreen)]()
[![verified](https://img.shields.io/badge/verified-44%2F44%20passes-brightgreen)]()
[![model](https://img.shields.io/badge/model-86KB%2021904%20params-blue)]()

---

## What this is

This repository implements the **Semantic Transformer pivot**: a system in which a language model is not a probabilistic truth-teller but a probabilistic **Proposer** of *execution traces* through a **certified finite semantic universe**. Truth is decided only by deterministic execution and cryptographic replay verification.

In standard transformers, reasoning is next-token mimicry. That architecture has a structural failure mode -- **hallucination** -- because there is no grounded model that must be satisfied. This repo builds that grounded model.

| Role | What it does |
|------|--------------|
| **Proposer** | Neural MLP (21,904 params) generates candidate op sequences through the tower |
| **Executor** | Runs the trace deterministically; produces a witness and a SHA-256 digest chain |
| **Verifier** | Replays the trace independently; certifies or rejects every step |

A trace is accepted only when Executor and Verifier agree on every step digest. No floating point is used for ordering or distance. All identity is structural.

---

## Quick start

    cargo build --release
    cargo run --bin tower -- query P
    cargo run --bin tower -- verify runs/<timestamp>/
    cargo run --bin tower -- eval --n 44
    cargo run --bin tower -- list
    cargo run --bin tower -- corpus --version v3
    cargo run --bin benchmark

---

## Results

| Metric | Value |
|--------|-------|
| Canonical accuracy | **528/528 (100.0%)** |
| Forward passes verified | **44/44** |
| Noise robustness @0.05 | **98.3%** |
| Noise robustness @0.30 | **83.3%** |
| Adversarial soundness | **9/9 attacks rejected** |
| Model size | **86 KB ONNX** |
| Parameters | **21,904** |
| Tests | **322 passing, 0 failures, 0 warnings** |

See [BENCHMARK.md](BENCHMARK.md) for full metrics and [PAPER.md](PAPER.md) for the technical writeup.

---

## Architecture

### The 7-Layer Certified Tower

    PHONEME (44) -> SYLLABLE (~423K) -> MORPHEME (16) -> WORD (34,487)
        -> PHRASE (5 trees) -> SEMANTIC (6 graphs) -> DISCOURSE (5 graphs)

Each layer has a canonical byte encoding, a structural signature (8-16 bits), and a SHA-256 digest. The tower root is a Merkle root over all layer manifests:

    tower_root: 06d344512ff8de62c1b6b0cbefa1d31e5778a69e76f69ac55375a8632142bae4

Stable across builds and platforms. Any change to any element propagates immediately.

### The Canonical 12-Block Sequence

| block | Op | Layer |
|-------|----|-------|
| 0 | SELECT_UNIVERSE | PHONEME |
| 1 | WITNESS_NEAREST | PHONEME |
| 2 | ATTEND | PHONEME |
| 3 | FFN_STEP | PHONEME -> SYLLABLE |
| 4 | FFN_STEP | SYLLABLE -> MORPHEME |
| 5 | FFN_STEP | MORPHEME -> WORD |
| 6 | FFN_STEP | WORD -> PHRASE |
| 7 | FFN_STEP | PHRASE -> SEMANTIC |
| 8 | FFN_STEP | SEMANTIC -> DISCOURSE |
| 9 | PROJECT_LAYER | DISCOURSE |
| 10 | RETURN_SET | DISCOURSE |
| 11 | ACCEPT | DISCOURSE |

### The Neural Proposer

    Input(25) -> Linear(25->128) -> ReLU -> Linear(128->128) -> ReLU -> op_head(8)

| Dims | Feature | Encoding |
|------|---------|----------|
| 0-11 | block_idx % 12 | one-hot |
| 12-18 | active_layer | one-hot (7 layers) |
| 19 | step_count | normalized / 60.0 |
| 20-23 | tau bin | one-hot over {0.5,1.0,2.0,4.0} |
| 24 | top_k | normalized / 10.0 |

### The Cryptographic Spine

    chain_hash_{t+1} = SHA256(chain_hash_t || step_digest_t)

Every trace produced by `tower query` can be independently certified by `tower verify` without access to the model or original execution environment.

---

## CLI reference

    tower query <phoneme>           run query, write verified trace
    tower verify <path>             certify a trace from disk
    tower eval [--n N]              evaluate proposer over N phonemes
    tower corpus [--version V]      show corpus statistics
    tower list                      list all 44 phonemes

    Flags (all subcommands):
      --model <path>       ONNX model (default: train/model_v3.onnx)
      --proposer onnx|rule proposer backend (default: onnx)
      --tau <f64>          attention temperature (default: 1.0)
      --top-k <usize>      top-k candidates (default: 3)
      --verbose            verbose output

---

## Architectural findings

**Proposer-verifier separation.** Attempts to train the proposer to predict REJECT failed at any class weight. Rejection scenarios share identical (block_idx, active_layer) features with valid ops. The proposer proposes the most likely correct op; the verifier rejects incorrect ones. Non-overlapping functions.

**Process supervision degradation.** PS with a noise curriculum conflicted with KL regularization, degrading IL from 99.48% to ~70%. The IL checkpoint is the production model. Fix: anneal KL weight as noise increases.

**Chain hash scope.** The chain_hash contributes to the cryptographic spine, not the neural features. Cryptographic chain = for verification; position + layer = for proposing.

---

## Stage progression

| Stages | Description | Key result |
|--------|-------------|------------|
| 0-9 | Certified tower: 7 layers, attention, FFN, transformer | tower_root stable |
| 10-17 | Corpus v1/v2, IL, RL, process supervision | 1.000 accuracy |
| 18-19 | ONNX + Rust ORT, full inventory | 528/528 |
| 20 | Adversarial stress | 9/9 verifier |
| 21 | Warning cleanup | 0 warnings, 322 tests |
| 22 | Multi-phoneme corpus v3 | 17,230 records |
| 23 | Curriculum retraining | 99.48% val, 98.3%@noise=0.05 |
| 24 | Rejection corpus v4 | Architectural finding |
| 25-27 | Live execution, trace verification, unified CLI | tower query P |
| 28-30 | Benchmark, paper, README | BENCHMARK.md, PAPER.md |

---

## Repo layout

    src/bin/
      tower.rs              Unified CLI
      benchmark.rs          Benchmark report
      verify_trace.rs       Standalone trace verifier
      generate_corpus_v3.rs Multi-phoneme corpus generator
      adversarial_stress.rs Verifier + proposer stress test
    src/
      phoneme..discourse    7 certified linguistic layers
      attention.rs          CertifiedAttention
      transformer.rs        TowerTransformer + ForwardPass
      onnx_proposer.rs      OnnxProposer via ort
      features.rs           FeatureEncoder (25-dim)
      tower.rs              Unified Tower artifact
      verify.rs             Digest chain verifier
    train/
      train_v3.py           IL training on corpus_v3
      model_v3.onnx         Production model (86 KB)
    training_data/
      corpus_v3.ndjson      17,230 records
      corpus_v4.ndjson      18,730 records
    PAPER.md
    BENCHMARK.md
    benchmark.json

---

## Dependencies

**Rust:** serde_json, sha2, anyhow, chrono, hex, ort (ONNX Runtime), clap

**Python:** torch, onnx, onnxruntime, numpy

---

## License

MIT

---

## Platform notes

**Apple Silicon (M1/M2/M3):** builds out of the box.

**Intel Mac (x86_64):** ONNX Runtime prebuilt binaries are not available for this target.
Install ORT via Homebrew first, then set ORT_DYLIB_PATH before building:

    brew install onnxruntime
    export ORT_DYLIB_PATH=$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib
    cargo build --release

To make it permanent:

    echo 'export ORT_DYLIB_PATH=$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib' >> ~/.zshrc

**Linux x86_64:** download a prebuilt release from
https://github.com/microsoft/onnxruntime/releases, extract it, then:

    export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
    cargo build --release
