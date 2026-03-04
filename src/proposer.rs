//! Probabilistic proposer — Stage 8 of the tower blueprint.
//!
//! The proposer is the learned component of the system. It emits a
//! distribution over certified tower operations given context.
//!
//! Architecture:
//!   Input:  TowerContext (active_layer, chain_hash, witness_digest, pass_digest)
//!   Output: P(next_op | context) where next_op ∈ {SELECT_UNIVERSE, WITNESS_NEAREST,
//!           RETURN_SET, PROJECT_LAYER, ATTEND, FFN_STEP}
//!
//! Key invariants:
//!   - Proposer can propose ANYTHING — it is unconstrained at proposal time
//!   - Executor/verifier filters: invalid proposals are rejected
//!   - Rejection is a training signal (binary reward: verifies or not)
//!   - No reward hacking: verifier is deterministic, rules live in universe digests
//!
//! This module provides:
//!   1. ProposedOp — a typed operation proposal with confidence score
//!   2. OpDistribution — a distribution over ProposedOps (softmax over scores)
//!   3. RuleBasedProposer — a deterministic proposer for testing/bootstrapping
//!   4. TraceRecord — a verified trace record for imitation learning
//!   5. ProposerTrainer — collects verified traces, computes training stats

use crate::digest::sha256_bytes;
use crate::layer::LayerId;

// ── OpKind ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum OpKind {
    SelectUniverse,
    WitnessNearest,
    ReturnSet,
    ProjectLayer,
    Attend,
    FFNStep,
    Accept,   // terminal: accept current output
    Reject,   // terminal: reject and restart
}

impl OpKind {
    pub fn as_str(self) -> &'static str {
        match self {
            OpKind::SelectUniverse => "SELECT_UNIVERSE",
            OpKind::WitnessNearest => "WITNESS_NEAREST",
            OpKind::ReturnSet      => "RETURN_SET",
            OpKind::ProjectLayer   => "PROJECT_LAYER",
            OpKind::Attend         => "ATTEND",
            OpKind::FFNStep        => "FFN_STEP",
            OpKind::Accept         => "ACCEPT",
            OpKind::Reject         => "REJECT",
        }
    }

    pub fn all() -> &'static [OpKind] {
        &[
            OpKind::SelectUniverse,
            OpKind::WitnessNearest,
            OpKind::ReturnSet,
            OpKind::ProjectLayer,
            OpKind::Attend,
            OpKind::FFNStep,
            OpKind::Accept,
            OpKind::Reject,
        ]
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, OpKind::Accept | OpKind::Reject)
    }
}

// ── ProposerContext ───────────────────────────────────────────────────────────

/// The context fed to the proposer at each step.
#[derive(Clone, Debug)]
pub struct ProposerContext {
    /// Currently active layer.
    pub active_layer:    LayerId,
    /// SHA-256 chain hash of all previous step digests.
    pub chain_hash:      [u8; 32],
    /// Digest of the current witness element (if any).
    pub witness_digest:  Option<[u8; 32]>,
    /// Digest of the last forward pass (if any).
    pub pass_digest:     Option<[u8; 32]>,
    /// Number of steps taken so far in this trace.
    pub step_count:      usize,
    /// Number of rejections so far (proposer's error rate).
    pub rejection_count: usize,
}

impl ProposerContext {
    pub fn new(active_layer: LayerId) -> Self {
        ProposerContext {
            active_layer,
            chain_hash:      sha256_bytes(b"proposer_context_genesis"),
            witness_digest:  None,
            pass_digest:     None,
            step_count:      0,
            rejection_count: 0,
        }
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        format!(
            "{{\"active_layer\":\"{}\",\"chain_hash\":\"{}\",\"rejection_count\":{},\"step_count\":{}}}",
            self.active_layer.as_str(),
            hex::encode(self.chain_hash),
            self.rejection_count,
            self.step_count,
        ).into_bytes()
    }

    pub fn digest(&self) -> [u8; 32] {
        sha256_bytes(&self.canonical_bytes())
    }

    /// Advance chain hash after a verified step.
    pub fn advance(&mut self, step_digest: &[u8; 32], new_layer: LayerId) {
        let mut combined = Vec::with_capacity(64);
        combined.extend_from_slice(&self.chain_hash);
        combined.extend_from_slice(step_digest);
        self.chain_hash   = sha256_bytes(&combined);
        self.active_layer = new_layer;
        self.step_count  += 1;
    }

    /// Record a rejection.
    pub fn record_rejection(&mut self) {
        self.rejection_count += 1;
    }
}

// ── ProposedOp ────────────────────────────────────────────────────────────────

/// A single proposed operation with a confidence score.
#[derive(Clone, Debug)]
pub struct ProposedOp {
    pub kind:       OpKind,
    /// Target layer for PROJECT_LAYER / ATTEND / FFN_STEP; None for others.
    pub tgt_layer:  Option<LayerId>,
    /// Query signature for ATTEND / WITNESS_NEAREST; None for others.
    pub query_sig:  Option<u16>,
    /// Temperature for ATTEND / FFN_STEP.
    pub tau:        f64,
    /// Raw log-probability score (unnormalized).
    pub log_score:  f64,
}

impl ProposedOp {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        format!(
            "{{\"kind\":\"{}\",\"log_score\":{:.6},\"tau\":{:.6},\"tgt_layer\":\"{}\"}}",
            self.kind.as_str(),
            self.log_score,
            self.tau,
            self.tgt_layer.map(|l| l.as_str()).unwrap_or("none"),
        ).into_bytes()
    }

    pub fn digest(&self) -> [u8; 32] {
        sha256_bytes(&self.canonical_bytes())
    }
}

// ── OpDistribution ────────────────────────────────────────────────────────────

/// A distribution over ProposedOps (softmax over log scores).
#[derive(Clone, Debug)]
pub struct OpDistribution {
    pub ops:     Vec<ProposedOp>,
    pub weights: Vec<f64>,   // softmax of log_scores, sums to 1.0
}

impl OpDistribution {
    pub fn from_ops(mut ops: Vec<ProposedOp>) -> Self {
        if ops.is_empty() {
            return OpDistribution { ops: vec![], weights: vec![] };
        }
        // Softmax over log_scores
        let max_score = ops.iter().map(|o| o.log_score).fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = ops.iter().map(|o| (o.log_score - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();
        let weights: Vec<f64> = exp_scores.iter()
            .map(|&e| (e / sum_exp * 1_000_000.0).round() / 1_000_000.0)
            .collect();

        // Sort ops and weights by weight descending
        let mut indexed: Vec<(usize, f64)> = weights.iter().enumerate()
            .map(|(i, &w)| (i, w)).collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        let sorted_ops:     Vec<ProposedOp> = indexed.iter().map(|(i, _)| ops[*i].clone()).collect();
        let sorted_weights: Vec<f64>        = indexed.iter().map(|(_, w)| *w).collect();

        OpDistribution { ops: sorted_ops, weights: sorted_weights }
    }

    /// Top proposed operation.
    pub fn top(&self) -> Option<&ProposedOp> {
        self.ops.first()
    }

    /// Verify weights sum to ~1.0.
    pub fn verify_sum(&self) -> Result<(), String> {
        let sum: f64 = self.weights.iter().sum();
        if (sum - 1.0).abs() < 1e-3 {
            Ok(())
        } else {
            Err(format!("op weights sum to {:.6}, expected 1.0", sum))
        }
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let parts: Vec<String> = self.ops.iter().zip(self.weights.iter())
            .map(|(op, &w)| format!("{}:{:.6}", op.kind.as_str(), w))
            .collect();
        format!("{{\"dist\":[{}]}}", parts.join(",")).into_bytes()
    }

    pub fn digest(&self) -> [u8; 32] {
        sha256_bytes(&self.canonical_bytes())
    }
}

// ── RuleBasedProposer ─────────────────────────────────────────────────────────

/// A deterministic rule-based proposer for bootstrapping and testing.
/// Does NOT use learned weights — uses hand-coded heuristics.
/// Used to generate the initial verified trace corpus for imitation learning.
pub struct RuleBasedProposer;

impl RuleBasedProposer {
    /// Propose the next operation given context.
    /// Rules:
    ///   - Step 0: always SELECT_UNIVERSE
    ///   - Step 1: always WITNESS_NEAREST
    ///   - Steps 2..7: PROJECT_LAYER upward through tower
    ///   - Step 7: ATTEND on current layer
    ///   - Step 8+: ACCEPT if pass_digest present, else FFN_STEP
    pub fn propose(ctx: &ProposerContext) -> OpDistribution {
        let ops = match ctx.step_count {
            0 => vec![
                ProposedOp {
                    kind:      OpKind::SelectUniverse,
                    tgt_layer: Some(ctx.active_layer),
                    query_sig: None,
                    tau:       1.0,
                    log_score: 2.0,
                },
                ProposedOp {
                    kind:      OpKind::WitnessNearest,
                    tgt_layer: None,
                    query_sig: None,
                    tau:       1.0,
                    log_score: 0.5,
                },
            ],
            1 => vec![
                ProposedOp {
                    kind:      OpKind::WitnessNearest,
                    tgt_layer: None,
                    query_sig: None,
                    tau:       1.0,
                    log_score: 2.0,
                },
                ProposedOp {
                    kind:      OpKind::Attend,
                    tgt_layer: Some(ctx.active_layer),
                    query_sig: None,
                    tau:       1.0,
                    log_score: 0.5,
                },
            ],
            2..=7 => {
                let tgt = Self::next_layer(ctx.active_layer);
                let mut ops = vec![
                    ProposedOp {
                        kind:      OpKind::ProjectLayer,
                        tgt_layer: tgt,
                        query_sig: None,
                        tau:       1.0,
                        log_score: 2.0,
                    },
                    ProposedOp {
                        kind:      OpKind::FFNStep,
                        tgt_layer: tgt,
                        query_sig: None,
                        tau:       1.0,
                        log_score: 1.0,
                    },
                ];
                if ctx.pass_digest.is_some() {
                    ops.push(ProposedOp {
                        kind:      OpKind::Accept,
                        tgt_layer: None,
                        query_sig: None,
                        tau:       1.0,
                        log_score: 0.1,
                    });
                }
                ops
            }
            _ => vec![
                ProposedOp {
                    kind:      OpKind::Accept,
                    tgt_layer: None,
                    query_sig: None,
                    tau:       1.0,
                    log_score: if ctx.pass_digest.is_some() { 2.0 } else { 0.5 },
                },
                ProposedOp {
                    kind:      OpKind::Attend,
                    tgt_layer: Some(ctx.active_layer),
                    query_sig: None,
                    tau:       1.0,
                    log_score: 0.5,
                },
                ProposedOp {
                    kind:      OpKind::Reject,
                    tgt_layer: None,
                    query_sig: None,
                    tau:       1.0,
                    log_score: -2.0,
                },
            ],
        };
        OpDistribution::from_ops(ops)
    }

    fn next_layer(current: LayerId) -> Option<LayerId> {
        match current {
            LayerId::Phoneme   => Some(LayerId::Syllable),
            LayerId::Syllable  => Some(LayerId::Morpheme),
            LayerId::Morpheme  => Some(LayerId::Word),
            LayerId::Word      => Some(LayerId::Phrase),
            LayerId::Phrase    => Some(LayerId::Semantic),
            LayerId::Semantic  => Some(LayerId::Discourse),
            LayerId::Discourse => None,
            _                  => None,
        }
    }
}

// ── TraceRecord ───────────────────────────────────────────────────────────────

/// A single verified trace record for imitation learning.
/// Collected during verified execution; used to train the learned proposer.
#[derive(Clone, Debug)]
pub struct TraceRecord {
    /// Context at the time of proposal.
    pub context:       ProposerContext,
    /// The op that was proposed AND verified successfully.
    pub accepted_op:   ProposedOp,
    /// The full distribution that was proposed (for policy gradient).
    pub distribution:  OpDistribution,
    /// Step digest from the executor after verification.
    pub step_digest:   [u8; 32],
    /// Whether this op was accepted on first try (no rejection).
    pub first_try:     bool,
}

impl TraceRecord {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        format!(
            "{{\"context_digest\":\"{}\",\"first_try\":{},\"op_kind\":\"{}\",\"step_digest\":\"{}\"}}",
            hex::encode(self.context.digest()),
            self.first_try,
            self.accepted_op.kind.as_str(),
            hex::encode(self.step_digest),
        ).into_bytes()
    }

    pub fn digest(&self) -> [u8; 32] {
        sha256_bytes(&self.canonical_bytes())
    }
}

// ── ProposerTrainer ───────────────────────────────────────────────────────────

/// Collects verified trace records and computes training statistics.
/// The corpus here is used for imitation learning (Stage 1 of training).
pub struct ProposerTrainer {
    pub records:          Vec<TraceRecord>,
    pub total_proposals:  usize,
    pub total_rejections: usize,
}

impl ProposerTrainer {
    pub fn new() -> Self {
        ProposerTrainer {
            records:          Vec::new(),
            total_proposals:  0,
            total_rejections: 0,
        }
    }

    pub fn record(&mut self, record: TraceRecord) {
        if !record.first_try {
            self.total_rejections += 1;
        }
        self.total_proposals += 1;
        self.records.push(record);
    }

    /// Acceptance rate (fraction of proposals accepted on first try).
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_proposals == 0 { return 0.0; }
        let accepted = self.total_proposals - self.total_rejections;
        accepted as f64 / self.total_proposals as f64
    }

    /// Op frequency table: how often each OpKind was accepted.
    pub fn op_frequencies(&self) -> Vec<(OpKind, usize)> {
        let mut counts = std::collections::HashMap::new();
        for r in &self.records {
            *counts.entry(r.accepted_op.kind).or_insert(0usize) += 1;
        }
        let mut freq: Vec<(OpKind, usize)> = counts.into_iter().collect();
        freq.sort_by(|(_, a), (_, b)| b.cmp(a));
        freq
    }

    /// Corpus digest: MerkleRoot over all record digests.
    pub fn corpus_digest(&self) -> [u8; 32] {
        if self.records.is_empty() {
            return sha256_bytes(b"empty_corpus");
        }
        let mut leaves: Vec<[u8; 32]> = self.records.iter()
            .map(|r| r.digest())
            .collect();
        leaves.sort_unstable();
        crate::digest::merkle_root(&leaves)
    }

    /// Summary string for diagnostics.
    pub fn summary(&self) -> String {
        format!(
            "ProposerTrainer: {} records, {:.1}% acceptance, corpus_digest={}",
            self.records.len(),
            self.acceptance_rate() * 100.0,
            hex::encode(&self.corpus_digest()[..8]),
        )
    }
}

impl Default for ProposerTrainer {
    fn default() -> Self { Self::new() }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx(layer: LayerId, step: usize) -> ProposerContext {
        let mut ctx = ProposerContext::new(layer);
        ctx.step_count = step;
        ctx
    }

    #[test]
    fn op_kind_all_roundtrip() {
        for &kind in OpKind::all() {
            assert!(!kind.as_str().is_empty());
        }
    }

    #[test]
    fn op_kind_terminal() {
        assert!(OpKind::Accept.is_terminal());
        assert!(OpKind::Reject.is_terminal());
        assert!(!OpKind::Attend.is_terminal());
        assert!(!OpKind::FFNStep.is_terminal());
    }

    #[test]
    fn proposer_context_digest_stable() {
        let ctx = ProposerContext::new(LayerId::Phoneme);
        assert_eq!(ctx.digest(), ctx.digest());
    }

    #[test]
    fn proposer_context_advance_changes_hash() {
        let mut ctx = ProposerContext::new(LayerId::Phoneme);
        let h0 = ctx.chain_hash;
        let step_digest = sha256_bytes(b"step1");
        ctx.advance(&step_digest, LayerId::Syllable);
        assert_ne!(ctx.chain_hash, h0);
        assert_eq!(ctx.active_layer, LayerId::Syllable);
        assert_eq!(ctx.step_count, 1);
    }

    #[test]
    fn op_distribution_weights_sum_to_one() {
        let ctx = make_ctx(LayerId::Phoneme, 0);
        let dist = RuleBasedProposer::propose(&ctx);
        dist.verify_sum().unwrap();
    }

    #[test]
    fn op_distribution_sorted_descending() {
        let ctx = make_ctx(LayerId::Phoneme, 0);
        let dist = RuleBasedProposer::propose(&ctx);
        for i in 1..dist.weights.len() {
            assert!(dist.weights[i] <= dist.weights[i-1] + 1e-9,
                "weights not sorted: [{}]={:.6} > [{}]={:.6}",
                i, dist.weights[i], i-1, dist.weights[i-1]);
        }
    }

    #[test]
    fn rule_proposer_step0_top_is_select_universe() {
        let ctx = make_ctx(LayerId::Phoneme, 0);
        let dist = RuleBasedProposer::propose(&ctx);
        assert_eq!(dist.top().unwrap().kind, OpKind::SelectUniverse);
    }

    #[test]
    fn rule_proposer_step1_top_is_witness_nearest() {
        let ctx = make_ctx(LayerId::Phoneme, 1);
        let dist = RuleBasedProposer::propose(&ctx);
        assert_eq!(dist.top().unwrap().kind, OpKind::WitnessNearest);
    }

    #[test]
    fn rule_proposer_mid_steps_propose_project() {
        for step in 2..=7 {
            let ctx = make_ctx(LayerId::Word, step);
            let dist = RuleBasedProposer::propose(&ctx);
            assert_eq!(dist.top().unwrap().kind, OpKind::ProjectLayer,
                "step {} should propose ProjectLayer", step);
        }
    }

    #[test]
    fn rule_proposer_late_step_proposes_accept() {
        let mut ctx = make_ctx(LayerId::Discourse, 10);
        ctx.pass_digest = Some(sha256_bytes(b"some_pass"));
        let dist = RuleBasedProposer::propose(&ctx);
        assert_eq!(dist.top().unwrap().kind, OpKind::Accept);
    }

    #[test]
    fn rule_proposer_all_layers_nonempty() {
        let layers = [
            LayerId::Phoneme, LayerId::Syllable, LayerId::Morpheme,
            LayerId::Word, LayerId::Phrase, LayerId::Semantic, LayerId::Discourse,
        ];
        for layer in layers {
            for step in 0..10 {
                let ctx = make_ctx(layer, step);
                let dist = RuleBasedProposer::propose(&ctx);
                assert!(!dist.ops.is_empty(), "empty dist for layer={:?} step={}", layer, step);
                dist.verify_sum().unwrap();
            }
        }
    }

    #[test]
    fn op_distribution_digest_stable() {
        let ctx = make_ctx(LayerId::Word, 3);
        let d1 = RuleBasedProposer::propose(&ctx);
        let d2 = RuleBasedProposer::propose(&ctx);
        assert_eq!(d1.digest(), d2.digest());
    }

    #[test]
    fn trace_record_digest_stable() {
        let ctx = ProposerContext::new(LayerId::Phoneme);
        let dist = RuleBasedProposer::propose(&ctx);
        let op = dist.top().unwrap().clone();
        let record = TraceRecord {
            context:      ctx,
            accepted_op:  op,
            distribution: dist,
            step_digest:  sha256_bytes(b"step"),
            first_try:    true,
        };
        assert_eq!(record.digest(), record.digest());
    }

    #[test]
    fn proposer_trainer_acceptance_rate() {
        let mut trainer = ProposerTrainer::new();
        let ctx = ProposerContext::new(LayerId::Phoneme);
        let dist = RuleBasedProposer::propose(&ctx);
        let op = dist.top().unwrap().clone();

        // Add 3 records: 2 first-try, 1 not
        for first_try in [true, true, false] {
            trainer.record(TraceRecord {
                context:      ctx.clone(),
                accepted_op:  op.clone(),
                distribution: dist.clone(),
                step_digest:  sha256_bytes(b"s"),
                first_try,
            });
        }

        let rate = trainer.acceptance_rate();
        assert!((rate - 2.0/3.0).abs() < 1e-6,
            "expected 0.667, got {:.6}", rate);
    }

    #[test]
    fn proposer_trainer_corpus_digest_stable() {
        let mut trainer = ProposerTrainer::new();
        let ctx = ProposerContext::new(LayerId::Phoneme);
        let dist = RuleBasedProposer::propose(&ctx);
        let op = dist.top().unwrap().clone();
        trainer.record(TraceRecord {
            context:      ctx,
            accepted_op:  op,
            distribution: dist,
            step_digest:  sha256_bytes(b"s"),
            first_try:    true,
        });
        assert_eq!(trainer.corpus_digest(), trainer.corpus_digest());
    }

    #[test]
    fn proposer_trainer_summary_nonempty() {
        let trainer = ProposerTrainer::new();
        let s = trainer.summary();
        assert!(s.contains("ProposerTrainer"));
        assert!(s.contains("records"));
    }

    #[test]
    fn proposer_trainer_op_frequencies() {
        let mut trainer = ProposerTrainer::new();
        let ctx = ProposerContext::new(LayerId::Phoneme);
        let dist = RuleBasedProposer::propose(&ctx);
        let op = dist.top().unwrap().clone();
        for _ in 0..3 {
            trainer.record(TraceRecord {
                context:      ctx.clone(),
                accepted_op:  op.clone(),
                distribution: dist.clone(),
                step_digest:  sha256_bytes(b"s"),
                first_try:    true,
            });
        }
        let freq = trainer.op_frequencies();
        assert!(!freq.is_empty());
        assert_eq!(freq[0].1, 3);
    }
}
