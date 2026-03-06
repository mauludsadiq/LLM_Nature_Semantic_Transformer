# LLM Nature Semantic Transformer — Benchmark Report

**Generated:** 2026-03-06 05:40:48Z

## System Overview

A certified linguistic tower with a neural ONNX proposer and cryptographic
step-digest verification. The proposer navigates a 12-block canonical sequence
across 7 linguistic layers (Phoneme → Discourse). Every step is verified via
SHA-256 digest chain.

## Results

### Canonical Accuracy

| Metric | Value |
|--------|-------|
| Phonemes evaluated | 44 |
| Blocks per phoneme | 12 |
| Total ops | 528 |
| Correct | 528 |
| **Accuracy** | **100.0%** |
| Eval time | 86182ms |

### Forward Pass Verification

| Metric | Value |
|--------|-------|
| Passes verified | 44/44 |
| Tower root | `06d344512ff8de62...` |

### Noise Robustness

| Noise | Accuracy | Gate |
|-------|----------|------|
| 0.00 | 1.000 | ✓ |
| 0.05 | 0.983 | ✓ |
| 0.10 | 0.950 | ~ |
| 0.15 | 0.917 | ~ |
| 0.20 | 0.917 | ~ |
| 0.30 | 0.833 | ✗ |

### Adversarial Stress

| Test | Result |
|------|--------|
| Verifier soundness (9 attack vectors) | 9/9 rejected |
| Proposer accuracy (clean) | 100.0% |

## Model

| Property | Value |
|----------|-------|
| Parameters | 21904 |
| ONNX size | 86 KB |
| Feature dim | 25 |
| Hidden dim | 128 |
| Architecture | 2-layer MLP |

## Corpus

| Version | Records | Notes |
|---------|---------|-------|
| v3 | 17230 | Multi-phoneme sequences, chain hash across phonemes |
| v4 | 18730 | + 5 rejection scenario types |

## Test Suite

322 tests, 0 failures, 0 warnings.

## Stage Progression

| Stage | Description | Gate |
|-------|-------------|------|
| 0–17 | Tower build, corpus v1/v2, IL+RL+PS training | ✓ |
| 18 | ONNX export + Rust ORT integration | 528/528 |
| 19 | Full 44-phoneme inventory validation | 528/528 |
| 20 | Adversarial stress test | 9/9 verifier, 96.6%@noise=0.05 |
| 21 | Warning cleanup | 0 warnings, 322/322 tests |
| 22 | Multi-phoneme corpus v3 | 17,230 records, 720 seqs |
| 23 | Curriculum retraining | 99.48% val, 97%@noise=0.05 |
| 24 | Rejection corpus v4 | 8.8% REJECT, entropy 2.520 bits |
| 25 | Live query execution | `tower query P` → verified trace |
| 26 | Trace replay verification | `tower verify runs/...` |
| 27 | Unified CLI | query/verify/eval/corpus/list |
| 28 | Benchmark report | This document |
