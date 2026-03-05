//! OnnxProposer — Stage 18.
//! Drop-in replacement for RuleBasedProposer using train/model.onnx via `ort`.

use anyhow::{bail, Context, Result};
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};

use crate::layer::LayerId;
use crate::proposer::{OpDistribution, OpKind, ProposedOp, ProposerContext};

// ── Index maps ────────────────────────────────────────────────────────────────

fn op_class_to_kind(cls: usize) -> Option<OpKind> {
    match cls {
        0 => Some(OpKind::SelectUniverse),
        1 => Some(OpKind::WitnessNearest),
        2 => Some(OpKind::Attend),
        3 => Some(OpKind::FFNStep),
        4 => Some(OpKind::ProjectLayer),
        5 => Some(OpKind::ReturnSet),
        6 => Some(OpKind::Accept),
        7 => Some(OpKind::Reject),
        _ => None,
    }
}

fn tgt_class_to_layer(cls: usize) -> Option<LayerId> {
    match cls {
        0 => Some(LayerId::Phoneme),
        1 => Some(LayerId::Syllable),
        2 => Some(LayerId::Morpheme),
        3 => Some(LayerId::Word),
        4 => Some(LayerId::Phrase),
        5 => Some(LayerId::Semantic),
        6 => Some(LayerId::Discourse),
        _ => None,
    }
}

fn tau_bin(tau: f64) -> usize {
    if tau <= 0.5 { 0 } else if tau <= 1.0 { 1 } else if tau <= 2.0 { 2 } else { 3 }
}

fn layer_idx(layer: LayerId) -> usize {
    match layer {
        LayerId::Phoneme   => 0,
        LayerId::Syllable  => 1,
        LayerId::Morpheme  => 2,
        LayerId::Word      => 3,
        LayerId::Phrase    => 4,
        LayerId::Semantic  => 5,
        LayerId::Discourse => 6,
        _                  => 0,
    }
}

// ── Feature encoding ──────────────────────────────────────────────────────────

fn encode_features(ctx: &ProposerContext, block_idx: usize, tau: f64, top_k: usize) -> Vec<f32> {
    let mut x = vec![0.0f32; 25];
    if block_idx < 12 { x[block_idx] = 1.0; }
    x[12 + layer_idx(ctx.active_layer)] = 1.0;
    x[19] = (ctx.step_count as f32 / 20.0).min(1.0);
    x[20 + tau_bin(tau)] = 1.0;
    x[24] = (top_k as f32 / 10.0).min(1.0);
    x
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = logits.iter().map(|&l| ((l - max) as f64).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ── OnnxProposer ─────────────────────────────────────────────────────────────

pub struct OnnxProposer {
    session: Session,
}

impl OnnxProposer {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("ort builder failed: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("optimization level failed: {e}"))?
            .commit_from_file(model_path)
            .with_context(|| format!("failed to load ONNX model: {model_path}"))?;
        Ok(OnnxProposer { session })
    }

    pub fn propose(
        &mut self,
        ctx:       &ProposerContext,
        block_idx: usize,
        tau:       f64,
        top_k:     usize,
    ) -> Result<OpDistribution> {
        let feats = encode_features(ctx, block_idx, tau, top_k);
        let shape = vec![1i64, 25i64];

        // Build DynTensor input
        let tensor = Tensor::<f32>::from_array((shape, feats))
            .context("failed to build feature tensor")?
            .upcast();

        let outputs = self.session
            .run(inputs!["features" => tensor])
            .context("ort inference failed")?;

        // Extract op_logits — returns (&Shape, &[f32])
        let (_, op_data) = outputs["op_logits"]
            .try_extract_tensor::<f32>()
            .context("extract op_logits failed")?;
        if op_data.len() != 8 {
            bail!("expected 8 op_logits, got {}", op_data.len());
        }
        let op_logits: Vec<f32> = op_data.to_vec();

        // Extract tgt_logits
        let (_, tgt_data) = outputs["tgt_logits"]
            .try_extract_tensor::<f32>()
            .context("extract tgt_logits failed")?;
        if tgt_data.len() != 8 {
            bail!("expected 8 tgt_logits, got {}", tgt_data.len());
        }
        let tgt_logits: Vec<f32> = tgt_data.to_vec();

        let op_probs  = softmax(&op_logits);
        let tgt_probs = softmax(&tgt_logits);

        let best_tgt = tgt_probs.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .and_then(|(i, _)| tgt_class_to_layer(i));

        let ops: Vec<ProposedOp> = (0..8usize)
            .filter_map(|cls| {
                Some(ProposedOp {
                    kind:      op_class_to_kind(cls)?,
                    tgt_layer: best_tgt,
                    query_sig: None,
                    tau,
                    log_score: op_probs[cls].ln(),
                })
            })
            .collect();

        Ok(OpDistribution::from_ops(ops))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: &str = "train/model.onnx";

    fn ctx(layer: LayerId, step: usize) -> ProposerContext {
        let mut c = ProposerContext::new(layer);
        c.step_count = step;
        c
    }

    #[test]
    fn test_canonical_sequence() {
        let mut p = OnnxProposer::new(MODEL).expect("load model");
        let seq: &[(usize, OpKind, LayerId)] = &[
            (0,  OpKind::SelectUniverse, LayerId::Phoneme),
            (1,  OpKind::WitnessNearest, LayerId::Phoneme),
            (2,  OpKind::Attend,         LayerId::Phoneme),
            (3,  OpKind::FFNStep,        LayerId::Phoneme),
            (4,  OpKind::FFNStep,        LayerId::Syllable),
            (5,  OpKind::FFNStep,        LayerId::Morpheme),
            (6,  OpKind::FFNStep,        LayerId::Word),
            (7,  OpKind::FFNStep,        LayerId::Phrase),
            (8,  OpKind::FFNStep,        LayerId::Semantic),
            (9,  OpKind::ProjectLayer,   LayerId::Discourse),
            (10, OpKind::ReturnSet,      LayerId::Discourse),
            (11, OpKind::Accept,         LayerId::Discourse),
        ];
        for &(bi, expected, layer) in seq {
            let dist = p.propose(&ctx(layer, bi), bi, 1.0, 3)
                .unwrap_or_else(|e| panic!("block {bi}: {e}"));
            let top = dist.top().expect("empty dist");
            assert_eq!(top.kind, expected,
                "block {bi}: expected {expected:?} got {:?}", top.kind);
        }
    }

    #[test]
    fn test_feature_bounds() {
        let feats = encode_features(&ctx(LayerId::Phoneme, 0), 0, 1.0, 3);
        for (i, &v) in feats.iter().enumerate() {
            assert!((0.0..=1.0).contains(&v), "feat[{i}]={v} out of [0,1]");
        }
    }

    #[test]
    fn test_block_idx_one_hot() {
        for bi in 0..12 {
            let feats = encode_features(&ctx(LayerId::Phoneme, bi), bi, 1.0, 3);
            let ones = feats[0..12].iter().filter(|&&v| v == 1.0).count();
            assert_eq!(ones, 1, "block {bi}: {ones} one-hot bits");
            assert_eq!(feats[bi], 1.0);
        }
    }

    #[test]
    fn test_weights_sum() {
        let mut p    = OnnxProposer::new(MODEL).unwrap();
        let dist = p.propose(&ctx(LayerId::Phoneme, 0), 0, 1.0, 3).unwrap();
        dist.verify_sum().unwrap();
    }

    #[test]
    fn test_determinism() {
        let mut p  = OnnxProposer::new(MODEL).unwrap();
        let c  = ctx(LayerId::Phoneme, 0);
        let d1 = p.propose(&c, 0, 1.0, 3).unwrap();
        let d2 = p.propose(&c, 0, 1.0, 3).unwrap();
        assert_eq!(d1.top().unwrap().kind, d2.top().unwrap().kind);
    }
}