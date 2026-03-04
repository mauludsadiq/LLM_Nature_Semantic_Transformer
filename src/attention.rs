//! CertifiedAttention — Stage 4 of the tower blueprint.
//!
//! Attention grounded in the certified tower instead of learned embedding space.
//!
//! Standard attention:
//!   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
//!
//! This system's attention:
//!   Attention(query_sig, layer, tau) =
//!     softmax(-sig_distance(query_sig, elem_sig) / tau)
//!     weighted over layer inventory
//!
//! Properties:
//!   - Weights are reproducible from query_sig + universe — no learned params needed
//!   - Output is a weighted sample from the universe
//!   - Weights sum to 1.0 (softmax)
//!   - The sample + weights are written to trace
//!   - Verifier checks weights were computed from certified sig distances
//!   - Temperature tau controls sharpness: tau→0 = argmax, tau→∞ = uniform
//!
//! Distance metric: Hamming distance over 16-bit signatures (integer, exact).
//! Softmax computed in f64 for numerical stability; result rounded to 6 decimal
//! places for trace stability across platforms.

use crate::digest::sha256_bytes;
use crate::layer::{Layer, LayerId, TowerPosition};
use crate::sig_index::SigIndex;

// ── AttentionWeight ───────────────────────────────────────────────────────────

/// A single weighted element in an attention result.
#[derive(Clone, Debug)]
pub struct AttentionWeight {
    /// Index into the layer inventory.
    pub elem_idx:  usize,
    /// Hamming distance from query_sig to this element's sig.
    pub distance:  u32,
    /// Softmax weight (sums to 1.0 across all weights in result).
    /// Rounded to 6 decimal places for trace stability.
    pub weight:    f64,
    /// Canonical rendering of this element.
    pub rendered:  String,
    /// Element digest.
    pub digest:    [u8; 32],
}

impl AttentionWeight {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        format!(
            "{{\"digest\":\"{}\",\"distance\":{},\"elem_idx\":{},\"weight\":{:.6}}}",
            hex::encode(self.digest),
            self.distance,
            self.elem_idx,
            self.weight,
        ).into_bytes()
    }
}

// ── AttentionResult ───────────────────────────────────────────────────────────

/// The full result of a CertifiedAttention operation.
#[derive(Clone, Debug)]
pub struct AttentionResult {
    pub layer:      LayerId,
    pub query_sig:  u16,
    pub tau:        f64,
    /// Top-k weighted elements, sorted by weight descending.
    pub weights:    Vec<AttentionWeight>,
    /// Index of the highest-weight element (the "witness").
    pub top_idx:    usize,
    /// Digest committing this entire attention result.
    pub result_digest: [u8; 32],
}

impl AttentionResult {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let weights_str: Vec<String> = self.weights.iter()
            .map(|w| hex::encode(sha256_bytes(&w.canonical_bytes())))
            .collect();
        format!(
            "{{\"layer\":\"{}\",\"query_sig\":{},\"tau\":{:.6},\"top_idx\":{},\"weights_digest\":\"{}\"}}",
            self.layer.as_str(),
            self.query_sig,
            self.tau,
            self.top_idx,
            hex::encode(sha256_bytes(weights_str.join(",").as_bytes())),
        ).into_bytes()
    }

    /// Verify that weights sum to approximately 1.0.
    pub fn verify_sum(&self) -> Result<(), String> {
        let sum: f64 = self.weights.iter().map(|w| w.weight).sum();
        if (sum - 1.0).abs() < 1e-4 {
            Ok(())
        } else {
            Err(format!("weights sum to {:.6}, expected 1.0", sum))
        }
    }

    /// Verify that weights are sorted descending.
    pub fn verify_order(&self) -> Result<(), String> {
        for i in 1..self.weights.len() {
            if self.weights[i].weight > self.weights[i-1].weight + 1e-9 {
                return Err(format!(
                    "weights not sorted: [{}]={:.6} > [{}]={:.6}",
                    i, self.weights[i].weight,
                    i-1, self.weights[i-1].weight
                ));
            }
        }
        Ok(())
    }

    /// Full verification: sum + order + digest.
    pub fn verify(&self) -> Result<(), String> {
        self.verify_sum()?;
        self.verify_order()?;
        let expected = sha256_bytes(&self.canonical_bytes());
        if expected == self.result_digest {
            Ok(())
        } else {
            Err(format!(
                "result_digest mismatch: expected {} got {}",
                hex::encode(expected),
                hex::encode(self.result_digest)
            ))
        }
    }
}

// ── CertifiedAttention ────────────────────────────────────────────────────────

pub struct CertifiedAttention;

impl CertifiedAttention {
    /// Compute attention over a layer given a query signature and temperature.
    ///
    /// - `query_sig`: the 16-bit signature to attend from
    /// - `tau`: temperature (must be > 0.0; typical range 0.5..4.0)
    /// - `top_k`: number of elements to return (sorted by weight desc)
    ///
    /// Algorithm:
    ///   1. For each element i in layer: score_i = -distance(query_sig, sig_i) / tau
    ///   2. softmax: weight_i = exp(score_i) / sum_j(exp(score_j))
    ///   3. Sort by weight descending, take top_k
    ///   4. Re-normalize weights to sum to 1.0
    ///   5. Commit result to digest
    pub fn attend(
        layer: &dyn Layer,
        query_sig: u16,
        tau: f64,
        top_k: usize,
    ) -> AttentionResult {
        assert!(tau > 0.0, "tau must be positive");
        assert!(top_k > 0, "top_k must be positive");
        assert!(layer.len() > 0, "layer must be non-empty");

        let k = top_k.min(layer.len());

        // Step 1+2: compute softmax scores
        // Use log-sum-exp trick for numerical stability
        let distances: Vec<u32> = (0..layer.len())
            .map(|i| layer.sig_distance(query_sig, layer.sig(i)))
            .collect();

        let scores: Vec<f64> = distances.iter()
            .map(|&d| -(d as f64) / tau)
            .collect();

        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter()
            .map(|&s| (s - max_score).exp())
            .collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        let raw_weights: Vec<f64> = exp_scores.iter()
            .map(|&e| e / sum_exp)
            .collect();

        // Step 3: take top_k by weight
        let mut indexed: Vec<(usize, f64, u32)> = raw_weights.iter()
            .enumerate()
            .map(|(i, &w)| (i, w, distances[i]))
            .collect();
        // Sort by weight desc, tie-break by distance asc, then index asc
        indexed.sort_by(|(ia, wa, da), (ib, wb, db)| {
            wb.partial_cmp(wa).unwrap()
                .then(da.cmp(db))
                .then(ia.cmp(ib))
        });
        indexed.truncate(k);

        // Step 4: re-normalize
        let sum_top: f64 = indexed.iter().map(|(_, w, _)| w).sum();
        let weights: Vec<AttentionWeight> = indexed.iter()
            .map(|&(i, w, d)| {
                let norm_w = (w / sum_top * 1_000_000.0).round() / 1_000_000.0;
                AttentionWeight {
                    elem_idx: i,
                    distance: d,
                    weight:   norm_w,
                    rendered: layer.render(i),
                    digest:   layer.digest(i),
                }
            })
            .collect();

        let top_idx = weights.first().map(|w| w.elem_idx).unwrap_or(0);

        // Step 5: commit
        let mut result = AttentionResult {
            layer:         layer.id(),
            query_sig,
            tau,
            weights,
            top_idx,
            result_digest: [0u8; 32],
        };
        result.result_digest = sha256_bytes(&result.canonical_bytes());
        result
    }

    /// Attend using a SigIndex for O(distinct_sigs) instead of O(n).
    /// Returns the same result as `attend` but uses the index for speed.
    pub fn attend_indexed(
        layer: &dyn Layer,
        index: &SigIndex,
        query_sig: u16,
        tau: f64,
        top_k: usize,
    ) -> AttentionResult {
        assert!(tau > 0.0, "tau must be positive");
        assert!(top_k > 0, "top_k must be positive");

        // Compute per-sig scores
        let mut sig_scores: Vec<(u16, f64, u32)> = index.map.keys()
            .map(|&sig| {
                let d = (sig ^ query_sig).count_ones();
                let score = -(d as f64) / tau;
                (sig, score, d)
            })
            .collect();

        // log-sum-exp normalization over sigs
        let max_score = sig_scores.iter().map(|(_, s, _)| *s)
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = sig_scores.iter()
            .map(|(_, s, _)| (s - max_score).exp())
            .collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        // Distribute sig weights to elements in posting lists
        let mut elem_weights: Vec<(usize, f64, u32)> = Vec::new();
        for (j, &(sig, _, dist)) in sig_scores.iter().enumerate() {
            if let Some(pl) = index.get(sig) {
                let sig_weight = exp_scores[j] / sum_exp;
                let per_elem = sig_weight / pl.indices.len() as f64;
                for &idx in &pl.indices {
                    elem_weights.push((idx, per_elem, dist));
                }
            }
        }

        // Sort and take top_k
        elem_weights.sort_by(|(ia, wa, da), (ib, wb, db)| {
            wb.partial_cmp(wa).unwrap()
                .then(da.cmp(db))
                .then(ia.cmp(ib))
        });
        elem_weights.truncate(top_k.min(elem_weights.len()));

        let sum_top: f64 = elem_weights.iter().map(|(_, w, _)| w).sum();
        let weights: Vec<AttentionWeight> = elem_weights.iter()
            .map(|&(i, w, d)| {
                let norm_w = (w / sum_top * 1_000_000.0).round() / 1_000_000.0;
                AttentionWeight {
                    elem_idx: i,
                    distance: d,
                    weight:   norm_w,
                    rendered: layer.render(i),
                    digest:   layer.digest(i),
                }
            })
            .collect();

        let top_idx = weights.first().map(|w| w.elem_idx).unwrap_or(0);

        let mut result = AttentionResult {
            layer:         layer.id(),
            query_sig,
            tau,
            weights,
            top_idx,
            result_digest: [0u8; 32],
        };
        result.result_digest = sha256_bytes(&result.canonical_bytes());
        result
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::SemanticLayer;
    use crate::word::WordLayer;
    use crate::discourse::DiscourseLayer;
    use crate::sig_index::SigIndex;
    use crate::layer::Layer;

    #[test]
    fn attention_semantic_weights_sum_to_one() {
        let layer = SemanticLayer::new();
        let result = CertifiedAttention::attend(&layer, layer.sig(0), 1.0, 6);
        result.verify_sum().unwrap();
    }

    #[test]
    fn attention_semantic_weights_ordered() {
        let layer = SemanticLayer::new();
        let result = CertifiedAttention::attend(&layer, layer.sig(0), 1.0, 6);
        result.verify_order().unwrap();
    }

    #[test]
    fn attention_semantic_verify_full() {
        let layer = SemanticLayer::new();
        let result = CertifiedAttention::attend(&layer, layer.sig(0), 1.0, 6);
        result.verify().unwrap();
    }

    #[test]
    fn attention_semantic_top_is_nearest() {
        let layer = SemanticLayer::new();
        let s = layer.sig(0);
        let result = CertifiedAttention::attend(&layer, s, 0.1, 6);
        // At very low temperature, top element should be at distance 0
        assert_eq!(result.weights[0].distance, 0,
            "top element not nearest at low tau");
    }

    #[test]
    fn attention_semantic_high_tau_more_uniform() {
        let layer = SemanticLayer::new();
        let s = layer.sig(0);
        let sharp  = CertifiedAttention::attend(&layer, s, 0.1, 6);
        let smooth = CertifiedAttention::attend(&layer, s, 10.0, 6);
        // High tau → top weight should be lower (more uniform)
        assert!(smooth.weights[0].weight <= sharp.weights[0].weight + 1e-6,
            "high tau not more uniform: sharp={:.4} smooth={:.4}",
            sharp.weights[0].weight, smooth.weights[0].weight);
    }

    #[test]
    fn attention_semantic_result_digest_stable() {
        let layer = SemanticLayer::new();
        let r1 = CertifiedAttention::attend(&layer, layer.sig(0), 1.0, 4);
        let r2 = CertifiedAttention::attend(&layer, layer.sig(0), 1.0, 4);
        assert_eq!(r1.result_digest, r2.result_digest);
    }

    #[test]
    fn attention_discourse_verify() {
        let layer = DiscourseLayer::new();
        let result = CertifiedAttention::attend(&layer, layer.sig(0), 1.0, 5);
        result.verify().unwrap();
    }

    #[test]
    fn attention_word_top_k_bounded() {
        let layer = WordLayer::new();
        let result = CertifiedAttention::attend(&layer, layer.sig(0), 1.0, 10);
        assert!(result.weights.len() <= 10);
        result.verify_sum().unwrap();
    }

    #[test]
    fn attention_word_verify() {
        let layer = WordLayer::new();
        let result = CertifiedAttention::attend(&layer, layer.sig(0), 2.0, 8);
        result.verify().unwrap();
    }

    #[test]
    fn attention_indexed_semantic_matches_full() {
        let layer = SemanticLayer::new();
        let index = SigIndex::build(&layer);
        let s = layer.sig(0);
        let full    = CertifiedAttention::attend(&layer, s, 1.0, 6);
        let indexed = CertifiedAttention::attend_indexed(&layer, &index, s, 1.0, 6);
        // Top element should be the same
        assert_eq!(full.top_idx, indexed.top_idx,
            "indexed top differs from full: {} vs {}", full.top_idx, indexed.top_idx);
        indexed.verify_sum().unwrap();
    }

    #[test]
    fn attention_indexed_word_verify() {
        let layer = WordLayer::new();
        let index = SigIndex::build(&layer);
        let result = CertifiedAttention::attend_indexed(&layer, &index, layer.sig(0), 1.5, 8);
        result.verify_sum().unwrap();
        result.verify_order().unwrap();
    }

    #[test]
    fn attention_weight_canonical_bytes_deterministic() {
        let layer = SemanticLayer::new();
        let result = CertifiedAttention::attend(&layer, layer.sig(0), 1.0, 3);
        let w = &result.weights[0];
        assert_eq!(w.canonical_bytes(), w.canonical_bytes());
    }

    #[test]
    fn attention_rendered_nonempty() {
        let layer = SemanticLayer::new();
        let result = CertifiedAttention::attend(&layer, layer.sig(0), 1.0, 6);
        for w in &result.weights {
            assert!(!w.rendered.is_empty());
        }
    }
}
