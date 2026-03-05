//! Feature encoder — Stage 11 of the Phase 2 blueprint.
//!
//! Converts a ProposerContext into a fixed-width f32 vector for neural net input.
//!
//! Feature vector layout (32 dims):
//!   [0..6]    one-hot: active_layer (7 dims)
//!   [7]       step_count / 20.0  (normalized, clipped at 1.0)
//!   [8]       rejection_count / 10.0 (normalized, clipped at 1.0)
//!   [9]       pass_digest present (0.0 or 1.0)
//!   [10]      witness_digest present (0.0 or 1.0)
//!   [11..26]  chain_hash[0..15] as f32 / 255.0 (16 dims, structural fingerprint)
//!   [27..30]  tau one-hot: [0.5, 1.0, 2.0, 4.0+] (4 dims)
//!   [31]      top_k / 10.0 (normalized)
//!
//! Properties:
//!   - All values in [0.0, 1.0]
//!   - Deterministic: same context → same vector on any platform
//!   - Differentiable: no discrete operations in the vector itself
//!   - Stable: adding new features appends, never shifts existing indices
//!
//! The encoder also provides label encoding for training:
//!   op_class:  u8 in [0..7]  (OpKind index)
//!   tgt_class: u8 in [0..7]  (LayerId index, 7=none)

use crate::layer::LayerId;
use crate::proposer::{ProposerContext, OpKind};

pub const FEATURE_DIM: usize = 34;

// ── LayerId one-hot indices ───────────────────────────────────────────────────

pub fn layer_index(id: LayerId) -> usize {
    match id {
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

pub fn layer_from_index(i: usize) -> Option<LayerId> {
    match i {
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

// ── OpKind label indices ──────────────────────────────────────────────────────

pub fn op_index(kind: OpKind) -> u8 {
    match kind {
        OpKind::SelectUniverse => 0,
        OpKind::WitnessNearest => 1,
        OpKind::Attend         => 2,
        OpKind::FFNStep        => 3,
        OpKind::ProjectLayer   => 4,
        OpKind::ReturnSet      => 5,
        OpKind::Accept         => 6,
        OpKind::Reject         => 7,
    }
}

pub fn op_from_index(i: u8) -> Option<OpKind> {
    match i {
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

// ── TauBin ────────────────────────────────────────────────────────────────────

/// Discretize tau into 4 bins for one-hot encoding.
pub fn tau_bin(tau: f64) -> usize {
    if tau <= 0.5       { 0 }
    else if tau <= 1.0  { 1 }
    else if tau <= 2.0  { 2 }
    else                { 3 }
}

// ── FeatureEncoder ────────────────────────────────────────────────────────────

pub struct FeatureEncoder;

impl FeatureEncoder {
    /// Encode a ProposerContext into a 32-dim f32 feature vector.
    /// All values in [0.0, 1.0].
    pub fn encode(ctx: &ProposerContext, tau: f64, top_k: usize) -> [f32; FEATURE_DIM] {
        let mut v = [0.0f32; FEATURE_DIM];

        // [0..6] one-hot: active_layer
        let li = layer_index(ctx.active_layer);
        v[li] = 1.0;

        // [7] step_count normalized
        v[7] = (ctx.step_count as f32 / 20.0).min(1.0);

        // [8] rejection_count normalized
        v[8] = (ctx.rejection_count as f32 / 10.0).min(1.0);

        // [9] pass_digest present
        v[9] = if ctx.pass_digest.is_some() { 1.0 } else { 0.0 };

        // [10] witness_digest present
        v[10] = if ctx.witness_digest.is_some() { 1.0 } else { 0.0 };

        // [11..26] chain_hash structural fingerprint
        for i in 0..16 {
            v[11 + i] = ctx.chain_hash[i] as f32 / 255.0;
        }

        // [27..30] tau one-hot
        let tb = tau_bin(tau);
        v[27 + tb] = 1.0;

        // [31] top_k normalized
        v[31] = (top_k as f32 / 10.0).min(1.0);

        v
    }

    /// Encode from raw fields (for use when loading from NDJSON corpus).
    pub fn encode_raw(
        active_layer:     LayerId,
        step_count:       usize,
        rejection_count:  usize,
        pass_present:     bool,
        witness_present:  bool,
        chain_hash:       &[u8; 32],
        tau:              f64,
        top_k:            usize,
    ) -> [f32; FEATURE_DIM] {
        Self::encode_raw_full(
            active_layer, step_count, rejection_count,
            pass_present, witness_present, chain_hash,
            tau, top_k, step_count, false,
        )
    }

    pub fn encode_raw_full(
        active_layer:     LayerId,
        step_count:       usize,
        rejection_count:  usize,
        pass_present:     bool,
        witness_present:  bool,
        chain_hash:       &[u8; 32],
        tau:              f64,
        top_k:            usize,
        block_idx:        usize,
        is_terminal:      bool,
    ) -> [f32; FEATURE_DIM] {
        let mut v = [0.0f32; FEATURE_DIM];

        v[layer_index(active_layer)] = 1.0;
        v[7]  = (step_count as f32 / 20.0).min(1.0);
        v[8]  = (rejection_count as f32 / 10.0).min(1.0);
        v[9]  = if pass_present    { 1.0 } else { 0.0 };
        v[10] = if witness_present { 1.0 } else { 0.0 };
        for i in 0..16 { v[11 + i] = chain_hash[i] as f32 / 255.0; }
        v[27 + tau_bin(tau)] = 1.0;
        v[31] = (top_k as f32 / 10.0).min(1.0);
        v[32] = (block_idx as f32 / 12.0).min(1.0);
        v[33] = if is_terminal { 1.0 } else { 0.0 };

        v
    }

    /// Verify all values are in [0.0, 1.0].
    pub fn verify(v: &[f32; FEATURE_DIM]) -> Result<(), String> {
        for (i, &x) in v.iter().enumerate() {
            if x < 0.0 || x > 1.0 || x.is_nan() || x.is_infinite() {
                return Err(format!("feature[{}] = {} out of [0,1]", i, x));
            }
        }
        Ok(())
    }

    /// Verify the one-hot layer section sums to exactly 1.0.
    pub fn verify_layer_onehot(v: &[f32; FEATURE_DIM]) -> Result<(), String> {
        let sum: f32 = v[0..7].iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            Err(format!("layer one-hot sum = {} (expected 1.0)", sum))
        } else {
            Ok(())
        }
    }

    /// Verify the tau one-hot section sums to exactly 1.0.
    pub fn verify_tau_onehot(v: &[f32; FEATURE_DIM]) -> Result<(), String> {
        let sum: f32 = v[27..31].iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            Err(format!("tau one-hot sum = {} (expected 1.0)", sum))
        } else {
            Ok(())
        }
    }

    /// Full verification.
    pub fn verify_all(v: &[f32; FEATURE_DIM]) -> Result<(), String> {
        Self::verify(v)?;
        Self::verify_layer_onehot(v)?;
        Self::verify_tau_onehot(v)?;
        Ok(())
    }

    /// Decode the active_layer from a feature vector.
    pub fn decode_layer(v: &[f32; FEATURE_DIM]) -> Option<LayerId> {
        let idx = v[0..7].iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)?;
        layer_from_index(idx)
    }

    /// Decode tau bin from a feature vector.
    pub fn decode_tau_bin(v: &[f32; FEATURE_DIM]) -> usize {
        v[27..31].iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Human-readable feature vector description.
    pub fn describe(v: &[f32; FEATURE_DIM]) -> String {
        let layer = Self::decode_layer(v)
            .map(|l| l.as_str().to_string())
            .unwrap_or("?".to_string());
        let tau_labels = ["0.5", "1.0", "2.0", "4.0+"];
        let tau = tau_labels[Self::decode_tau_bin(v)];
        let step = (v[7] * 20.0).round() as usize;
        let rej  = (v[8] * 10.0).round() as usize;
        format!(
            "layer={} step={} rej={} pass={} witness={} tau={} top_k={}",
            layer, step, rej,
            if v[9] > 0.5 { "yes" } else { "no" },
            if v[10] > 0.5 { "yes" } else { "no" },
            tau,
            (v[31] * 10.0).round() as usize,
        )
    }
}

// ── FeatureMatrix ─────────────────────────────────────────────────────────────

/// A matrix of feature vectors for batch encoding.
pub struct FeatureMatrix {
    pub data:   Vec<[f32; FEATURE_DIM]>,
    pub labels: Vec<(u8, u8)>,   // (op_class, tgt_class)
}

impl FeatureMatrix {
    pub fn new() -> Self {
        FeatureMatrix { data: Vec::new(), labels: Vec::new() }
    }

    pub fn push(&mut self, v: [f32; FEATURE_DIM], op_class: u8, tgt_class: u8) {
        self.data.push(v);
        self.labels.push((op_class, tgt_class));
    }

    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }

    /// Flatten to row-major f32 bytes for binary serialization.
    pub fn to_feature_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data.len() * FEATURE_DIM * 4);
        for row in &self.data {
            for &x in row {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    /// Flatten labels to (op_class, tgt_class) byte pairs.
    pub fn to_label_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data.len() * 2);
        for &(op, tgt) in &self.labels {
            out.push(op);
            out.push(tgt);
        }
        out
    }

    /// Op class distribution (count per class).
    pub fn op_distribution(&self) -> [usize; 8] {
        let mut counts = [0usize; 8];
        for &(op, _) in &self.labels {
            if (op as usize) < 8 { counts[op as usize] += 1; }
        }
        counts
    }

    /// Max class imbalance ratio. Gate: must be < 10.0 for Stage 12.
    pub fn max_imbalance(&self) -> f64 {
        let dist = self.op_distribution();
        let max = *dist.iter().max().unwrap_or(&1) as f64;
        let min = dist.iter().filter(|&&x| x > 0).min()
            .copied().unwrap_or(1) as f64;
        max / min
    }

    /// Verify all rows.
    pub fn verify_all(&self) -> Result<(), String> {
        for (i, row) in self.data.iter().enumerate() {
            FeatureEncoder::verify_all(row)
                .map_err(|e| format!("row {}: {}", i, e))?;
        }
        Ok(())
    }
}

impl Default for FeatureMatrix {
    fn default() -> Self { Self::new() }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proposer::ProposerContext;
    use crate::layer::LayerId;

    fn make_ctx(layer: LayerId, step: usize) -> ProposerContext {
        let mut ctx = ProposerContext::new(layer);
        ctx.step_count = step;
        ctx
    }

    #[test]
    fn feature_dim_correct() {
        assert_eq!(FEATURE_DIM, 32);
    }

    #[test]
    fn encode_all_values_in_range() {
        let ctx = make_ctx(LayerId::Word, 3);
        let v = FeatureEncoder::encode(&ctx, 1.0, 5);
        FeatureEncoder::verify(&v).unwrap();
    }

    #[test]
    fn encode_layer_onehot_valid() {
        for id in [LayerId::Phoneme, LayerId::Syllable, LayerId::Morpheme,
                   LayerId::Word, LayerId::Phrase, LayerId::Semantic, LayerId::Discourse] {
            let ctx = make_ctx(id, 0);
            let v = FeatureEncoder::encode(&ctx, 1.0, 3);
            FeatureEncoder::verify_layer_onehot(&v).unwrap();
        }
    }

    #[test]
    fn encode_tau_onehot_valid() {
        for tau in [0.5, 1.0, 2.0, 4.0] {
            let ctx = make_ctx(LayerId::Phoneme, 0);
            let v = FeatureEncoder::encode(&ctx, tau, 3);
            FeatureEncoder::verify_tau_onehot(&v).unwrap();
        }
    }

    #[test]
    fn encode_verify_all() {
        let ctx = make_ctx(LayerId::Semantic, 5);
        let v = FeatureEncoder::encode(&ctx, 2.0, 5);
        FeatureEncoder::verify_all(&v).unwrap();
    }

    #[test]
    fn encode_deterministic() {
        let ctx = make_ctx(LayerId::Word, 3);
        let v1 = FeatureEncoder::encode(&ctx, 1.0, 5);
        let v2 = FeatureEncoder::encode(&ctx, 1.0, 5);
        assert_eq!(v1, v2);
    }

    #[test]
    fn encode_different_layers_differ() {
        let ctx_ph  = make_ctx(LayerId::Phoneme,   0);
        let ctx_wrd = make_ctx(LayerId::Word,       0);
        let v1 = FeatureEncoder::encode(&ctx_ph,  1.0, 3);
        let v2 = FeatureEncoder::encode(&ctx_wrd, 1.0, 3);
        assert_ne!(v1, v2);
    }

    #[test]
    fn encode_different_taus_differ() {
        let ctx = make_ctx(LayerId::Phoneme, 0);
        let v1 = FeatureEncoder::encode(&ctx, 0.5, 3);
        let v2 = FeatureEncoder::encode(&ctx, 4.0, 3);
        assert_ne!(v1, v2);
    }

    #[test]
    fn encode_different_steps_differ() {
        let ctx0 = make_ctx(LayerId::Phoneme, 0);
        let ctx5 = make_ctx(LayerId::Phoneme, 5);
        let v1 = FeatureEncoder::encode(&ctx0, 1.0, 3);
        let v2 = FeatureEncoder::encode(&ctx5, 1.0, 3);
        assert_ne!(v1, v2);
    }

    #[test]
    fn decode_layer_roundtrip() {
        for id in [LayerId::Phoneme, LayerId::Syllable, LayerId::Morpheme,
                   LayerId::Word, LayerId::Phrase, LayerId::Semantic, LayerId::Discourse] {
            let ctx = make_ctx(id, 0);
            let v = FeatureEncoder::encode(&ctx, 1.0, 3);
            let decoded = FeatureEncoder::decode_layer(&v).unwrap();
            assert_eq!(decoded, id, "layer roundtrip failed for {:?}", id);
        }
    }

    #[test]
    fn decode_tau_bin_roundtrip() {
        let cases = [(0.5, 0), (1.0, 1), (2.0, 2), (4.0, 3), (8.0, 3)];
        for (tau, expected_bin) in cases {
            let ctx = make_ctx(LayerId::Phoneme, 0);
            let v = FeatureEncoder::encode(&ctx, tau, 3);
            assert_eq!(FeatureEncoder::decode_tau_bin(&v), expected_bin,
                "tau_bin failed for tau={}", tau);
        }
    }

    #[test]
    fn layer_index_roundtrip() {
        for id in [LayerId::Phoneme, LayerId::Syllable, LayerId::Morpheme,
                   LayerId::Word, LayerId::Phrase, LayerId::Semantic, LayerId::Discourse] {
            let i = layer_index(id);
            let back = layer_from_index(i).unwrap();
            assert_eq!(back, id);
        }
    }

    #[test]
    fn op_index_roundtrip() {
        for kind in [OpKind::SelectUniverse, OpKind::WitnessNearest, OpKind::Attend,
                     OpKind::FFNStep, OpKind::ProjectLayer, OpKind::ReturnSet,
                     OpKind::Accept, OpKind::Reject] {
            let i = op_index(kind);
            let back = op_from_index(i).unwrap();
            assert_eq!(back, kind);
        }
    }

    #[test]
    fn encode_raw_matches_encode() {
        let mut ctx = ProposerContext::new(LayerId::Word);
        ctx.step_count = 3;
        let v1 = FeatureEncoder::encode(&ctx, 1.0, 5);
        let v2 = FeatureEncoder::encode_raw(
            LayerId::Word, 3, 0, false, false,
            &ctx.chain_hash, 1.0, 5,
        );
        assert_eq!(v1, v2);
    }

    #[test]
    fn describe_nonempty() {
        let ctx = make_ctx(LayerId::Semantic, 4);
        let v = FeatureEncoder::encode(&ctx, 2.0, 5);
        let desc = FeatureEncoder::describe(&v);
        assert!(desc.contains("SEMANTIC"));
        assert!(desc.contains("tau=2.0"));
    }

    #[test]
    fn feature_matrix_push_and_verify() {
        let mut mat = FeatureMatrix::new();
        for step in 0..10 {
            let ctx = make_ctx(LayerId::Word, step);
            let v = FeatureEncoder::encode(&ctx, 1.0, 3);
            mat.push(v, 3, 3);
        }
        assert_eq!(mat.len(), 10);
        mat.verify_all().unwrap();
    }

    #[test]
    fn feature_matrix_to_bytes_length() {
        let mut mat = FeatureMatrix::new();
        let ctx = make_ctx(LayerId::Phoneme, 0);
        let v = FeatureEncoder::encode(&ctx, 1.0, 3);
        mat.push(v, 0, 7);
        let fb = mat.to_feature_bytes();
        let lb = mat.to_label_bytes();
        assert_eq!(fb.len(), FEATURE_DIM * 4);
        assert_eq!(lb.len(), 2);
        assert_eq!(lb[0], 0); // op_class
        assert_eq!(lb[1], 7); // tgt_class
    }

    #[test]
    fn feature_matrix_op_distribution() {
        let mut mat = FeatureMatrix::new();
        let ctx = make_ctx(LayerId::Phoneme, 0);
        let v = FeatureEncoder::encode(&ctx, 1.0, 3);
        for op in 0u8..8 {
            mat.push(v, op, 7);
        }
        let dist = mat.op_distribution();
        assert!(dist.iter().all(|&c| c == 1));
    }

    #[test]
    fn feature_matrix_imbalance_balanced() {
        let mut mat = FeatureMatrix::new();
        let ctx = make_ctx(LayerId::Phoneme, 0);
        let v = FeatureEncoder::encode(&ctx, 1.0, 3);
        for op in 0u8..8 {
            mat.push(v, op, 7);
        }
        let imb = mat.max_imbalance();
        assert!((imb - 1.0).abs() < 1e-6, "imbalance = {}", imb);
    }

    #[test]
    fn step_count_clip_at_one() {
        let ctx = make_ctx(LayerId::Phoneme, 100); // way over 20
        let v = FeatureEncoder::encode(&ctx, 1.0, 3);
        assert!((v[7] - 1.0).abs() < 1e-6, "step clipped to 1.0: {}", v[7]);
    }

    #[test]
    fn top_k_clip_at_one() {
        let ctx = make_ctx(LayerId::Phoneme, 0);
        let v = FeatureEncoder::encode(&ctx, 1.0, 100); // way over 10
        assert!((v[31] - 1.0).abs() < 1e-6, "top_k clipped to 1.0: {}", v[31]);
    }
}
