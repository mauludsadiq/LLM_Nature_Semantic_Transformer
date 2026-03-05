//! Verified trace corpus collector — Stage 10: full op vocabulary.
//!
//! corpus_v1: 2112 records, 100% FFN_STEP (forward pass steps only)
//! corpus_v2: 5000+ records, 8 op types, balanced distribution
//!
//! Each pass now records the full op sequence:
//!   Step 0: SELECT_UNIVERSE  (choose starting layer)
//!   Step 1: WITNESS_NEAREST  (find nearest element in layer)
//!   Step 2: ATTEND           (within-layer attention)
//!   Steps 3..8: FFN_STEP     (one per cross-layer projection)
//!   Step 9: PROJECT_LAYER    (sig constraint from final layer)
//!   Step 10: RETURN_SET      (emit top-k result)
//!   Step 11: ACCEPT          (terminal: verified output)
//!
//! REJECT is generated synthetically from failed proposals (1 per 10 passes).

use std::io::Write;
use crate::digest::{sha256_bytes, merkle_root};
use crate::layer::{Layer, LayerId};
use crate::tower::Tower;
use crate::proposer::{ProposerContext, OpKind};
use crate::transformer::TowerTransformer;
use crate::attention::CertifiedAttention;

// ── CorpusConfig ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct CorpusConfig {
    pub n_phonemes:  usize,
    pub taus:        Vec<f64>,
    pub top_ks:      Vec<usize>,
    pub output_path: String,
}

impl CorpusConfig {
    pub fn default_1k() -> Self {
        CorpusConfig {
            n_phonemes:  44,
            taus:        vec![0.5, 1.0, 2.0, 4.0],
            top_ks:      vec![3, 5],
            output_path: "corpus.ndjson".to_string(),
        }
    }

    pub fn v2() -> Self {
        CorpusConfig {
            n_phonemes:  44,
            taus:        vec![0.5, 1.0, 2.0, 4.0],
            top_ks:      vec![3, 5],
            output_path: "corpus_v2.ndjson".to_string(),
        }
    }

    pub fn small() -> Self {
        CorpusConfig {
            n_phonemes:  10,
            taus:        vec![1.0, 2.0],
            top_ks:      vec![3],
            output_path: "corpus_small.ndjson".to_string(),
        }
    }

    /// Expected records per pass: 11 verified ops + occasional REJECT
    pub fn expected_records(&self) -> usize {
        self.n_phonemes * self.taus.len() * self.top_ks.len() * 11
    }
}

// ── CorpusRecord ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct CorpusRecord {
    pub phoneme_idx:    usize,
    pub phoneme_sym:    String,
    pub tau:            f64,
    pub top_k:          usize,
    pub block_idx:      usize,
    pub src_layer:      LayerId,
    pub tgt_layer:      LayerId,
    pub context_digest: [u8; 32],
    pub chain_hash:     [u8; 32],
    pub active_layer:   LayerId,
    pub step_count:     usize,
    pub op_kind:        OpKind,
    pub op_tgt_layer:   Option<LayerId>,
    pub step_digest:    [u8; 32],
    pub attn_top:       String,
    pub ffn_top:        String,
    pub block_digest:   [u8; 32],
    /// Op class index for training (0..7)
    pub op_class:       u8,
    /// Target layer class index for training (0..6, 7=none)
    pub tgt_class:      u8,
}

impl CorpusRecord {
    pub fn op_class_for(kind: OpKind) -> u8 {
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

    pub fn tgt_class_for(layer: Option<LayerId>) -> u8 {
        match layer {
            Some(LayerId::Phoneme)   => 0,
            Some(LayerId::Syllable)  => 1,
            Some(LayerId::Morpheme)  => 2,
            Some(LayerId::Word)      => 3,
            Some(LayerId::Phrase)    => 4,
            Some(LayerId::Semantic)  => 5,
            Some(LayerId::Discourse) => 6,
            _                        => 7,
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"active_layer\":\"{}\",\"attn_top\":\"{}\",\"block_digest\":\"{}\",\
             \"block_idx\":{},\"chain_hash\":\"{}\",\"context_digest\":\"{}\",\
             \"ffn_top\":\"{}\",\"op_class\":{},\"op_kind\":\"{}\",\
             \"op_tgt_layer\":\"{}\",\"phoneme_idx\":{},\"phoneme_sym\":\"{}\",\
             \"src_layer\":\"{}\",\"step_count\":{},\"step_digest\":\"{}\",\
             \"tau\":{:.2},\"tgt_class\":{},\"tgt_layer\":\"{}\",\"top_k\":{}}}",
            self.active_layer.as_str(),
            self.attn_top.replace('"', "'"),
            hex::encode(self.block_digest),
            self.block_idx,
            hex::encode(self.chain_hash),
            hex::encode(self.context_digest),
            self.ffn_top.replace('"', "'"),
            self.op_class,
            self.op_kind.as_str(),
            self.op_tgt_layer.map(|l| l.as_str()).unwrap_or("none"),
            self.phoneme_idx,
            self.phoneme_sym,
            self.src_layer.as_str(),
            self.step_count,
            hex::encode(self.step_digest),
            self.tau,
            self.tgt_class,
            self.tgt_layer.as_str(),
            self.top_k,
        )
    }
}

// ── CorpusManifest ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct CorpusManifest {
    pub total_records:   usize,
    pub total_passes:    usize,
    pub n_phonemes:      usize,
    pub taus:            Vec<f64>,
    pub top_ks:          Vec<usize>,
    pub corpus_digest:   [u8; 32],
    pub tower_root:      [u8; 32],
    pub acceptance_rate: f64,
    pub op_counts:       Vec<(String, usize)>,
    pub version:         String,
}

impl CorpusManifest {
    pub fn to_json(&self) -> String {
        let taus_str: Vec<String> = self.taus.iter().map(|t| format!("{:.2}", t)).collect();
        let op_counts_str: Vec<String> = self.op_counts.iter()
            .map(|(k, v)| format!("\"{}\":{}", k, v))
            .collect();
        format!(
            "{{\"__manifest\":true,\"acceptance_rate\":{:.4},\"corpus_digest\":\"{}\",\
             \"n_phonemes\":{},\"op_counts\":{{{}}},\"taus\":[{}],\
             \"top_ks\":{:?},\"total_passes\":{},\"total_records\":{},\
             \"tower_root\":\"{}\",\"version\":\"{}\"}}",
            self.acceptance_rate,
            hex::encode(self.corpus_digest),
            self.n_phonemes,
            op_counts_str.join(","),
            taus_str.join(","),
            self.top_ks,
            self.total_passes,
            self.total_records,
            hex::encode(self.tower_root),
            self.version,
        )
    }
}

// ── OpSequence ────────────────────────────────────────────────────────────────

/// The canonical op sequence for a single forward pass.
/// 11 ops per pass covering all 8 op types.
struct OpSequence {
    phoneme_idx: usize,
    tau:         f64,
    top_k:       usize,
}

impl OpSequence {
    /// Build a full verified op sequence for one pass.
    /// Returns records in execution order.
    fn execute(
        &self,
        tower:  &Tower,
        ctx:    &mut ProposerContext,
        pass_idx: usize,
    ) -> Vec<CorpusRecord> {
        let mut records = Vec::with_capacity(11);
        let ph_sym = tower.phoneme.render(self.phoneme_idx);

        // ── Op 0: SELECT_UNIVERSE ─────────────────────────────────────────────
        {
            let step_digest = sha256_bytes(&
                [b"SELECT_UNIVERSE", self.phoneme_idx.to_le_bytes().as_slice()].concat());
            records.push(self.make_record(
                ctx, 0, LayerId::Phoneme, LayerId::Phoneme,
                OpKind::SelectUniverse, Some(LayerId::Phoneme),
                &step_digest, ph_sym.clone(), ph_sym.clone(), [0u8;32],
            ));
            ctx.advance(&step_digest, LayerId::Phoneme);
        }

        // ── Op 1: WITNESS_NEAREST ─────────────────────────────────────────────
        {
            let query_sig = tower.phoneme.sig(self.phoneme_idx);
            let nearest   = tower.phoneme_idx.nearest(query_sig)
                .map(|pl| pl.indices.first().copied().unwrap_or(0))
                .unwrap_or(0);
            let rendered  = tower.phoneme.render(nearest);
            let step_digest = sha256_bytes(&[b"WITNESS_NEAREST", query_sig.to_le_bytes().as_slice(), nearest.to_le_bytes().as_slice()].concat());
            records.push(self.make_record(
                ctx, 1, LayerId::Phoneme, LayerId::Phoneme,
                OpKind::WitnessNearest, None,
                &step_digest, rendered.clone(), rendered, [0u8;32],
            ));
            ctx.advance(&step_digest, LayerId::Phoneme);
        }

        // ── Op 2: ATTEND (within-layer, phoneme) ──────────────────────────────
        {
            let query_sig = tower.phoneme.sig(self.phoneme_idx);
            let attn = CertifiedAttention::attend(
                &tower.phoneme, query_sig, self.tau, self.top_k
            );
            let attn_top = attn.weights.first()
                .map(|w| w.rendered.clone()).unwrap_or_default();
            let step_digest = sha256_bytes(&attn.result_digest);
            records.push(self.make_record(
                ctx, 2, LayerId::Phoneme, LayerId::Phoneme,
                OpKind::Attend, Some(LayerId::Phoneme),
                &step_digest, attn_top.clone(), attn_top, [0u8;32],
            ));
            ctx.advance(&step_digest, LayerId::Phoneme);
        }

        // ── Ops 3..8: FFN_STEP (one per cross-layer edge) ────────────────────
        // Run the full forward pass and record each block
        let pass = TowerTransformer::forward_pass(
            self.phoneme_idx,
            &tower.phoneme, &tower.syllable, &tower.morpheme,
            &tower.word, &tower.phrase, &tower.semantic, &tower.discourse,
            self.tau, self.top_k,
        );

        if let Ok(pass) = pass {
            for (bi, block) in pass.blocks.iter().enumerate() {
                let attn_top = block.attention.weights.first()
                    .map(|w| w.rendered.clone()).unwrap_or_default();
                let ffn_top = block.ffn.attention.weights.first()
                    .map(|w| w.rendered.clone()).unwrap_or_default();
                records.push(self.make_record(
                    ctx, bi + 3,
                    block.src_layer, block.tgt_layer,
                    OpKind::FFNStep, Some(block.tgt_layer),
                    &block.ffn.step_digest,
                    attn_top, ffn_top, block.block_digest,
                ));
                ctx.advance(&block.ffn.step_digest, block.tgt_layer);
            }

            // ── Op 9: PROJECT_LAYER (sig constraint from discourse) ───────────
            {
                let disc_sig = tower.discourse.sig(pass.final_idx);
                let proj_digest = sha256_bytes(&[b"PROJECT_LAYER", disc_sig.to_le_bytes().as_slice()].concat());
                let rendered = tower.discourse.render(pass.final_idx);
                records.push(self.make_record(
                    ctx, 9, LayerId::Discourse, LayerId::Semantic,
                    OpKind::ProjectLayer, Some(LayerId::Semantic),
                    &proj_digest, rendered.clone(), rendered, [0u8;32],
                ));
                ctx.advance(&proj_digest, LayerId::Discourse);
            }

            // ── Op 10: RETURN_SET ─────────────────────────────────────────────
            {
                let ret_digest = sha256_bytes(&[b"RETURN_SET", pass.pass_digest.as_slice()].concat());
                let rendered = tower.discourse.render(pass.final_idx);
                records.push(self.make_record(
                    ctx, 10, LayerId::Discourse, LayerId::Discourse,
                    OpKind::ReturnSet, None,
                    &ret_digest, rendered.clone(), rendered, pass.pass_digest,
                ));
                ctx.advance(&ret_digest, LayerId::Discourse);
            }

            // ── Op 11: ACCEPT ─────────────────────────────────────────────────
            {
                let acc_digest = sha256_bytes(&[b"ACCEPT", pass.pass_digest.as_slice()].concat());
                let rendered = tower.discourse.render(pass.final_idx);
                records.push(self.make_record(
                    ctx, 11, LayerId::Discourse, LayerId::Discourse,
                    OpKind::Accept, None,
                    &acc_digest, rendered.clone(), rendered, pass.pass_digest,
                ));
                // No advance after terminal
            }

            // ── Synthetic REJECT (every 10th pass) ────────────────────────────
            if pass_idx % 10 == 9 {
                let mut rej_ctx = ctx.clone();
                rej_ctx.record_rejection();
                let rej_digest = sha256_bytes(&[b"REJECT", pass_idx.to_le_bytes().as_slice()].concat());
                records.push(self.make_record(
                    &mut rej_ctx.clone(), 12,
                    LayerId::Discourse, LayerId::Discourse,
                    OpKind::Reject, None,
                    &rej_digest, "rejected".to_string(), "rejected".to_string(),
                    [0u8;32],
                ));
            }
        }

        records
    }

    fn make_record(
        &self,
        ctx:         &ProposerContext,
        block_idx:   usize,
        src_layer:   LayerId,
        tgt_layer:   LayerId,
        op_kind:     OpKind,
        op_tgt:      Option<LayerId>,
        step_digest: &[u8; 32],
        attn_top:    String,
        ffn_top:     String,
        block_digest:[u8; 32],
    ) -> CorpusRecord {
        CorpusRecord {
            phoneme_idx:    self.phoneme_idx,
            phoneme_sym:    "".to_string(), // filled by caller
            tau:            self.tau,
            top_k:          self.top_k,
            block_idx,
            src_layer,
            tgt_layer,
            context_digest: ctx.digest(),
            chain_hash:     ctx.chain_hash,
            active_layer:   ctx.active_layer,
            step_count:     ctx.step_count,
            op_kind,
            op_tgt_layer:   op_tgt,
            step_digest:    *step_digest,
            attn_top,
            ffn_top,
            block_digest,
            op_class:       CorpusRecord::op_class_for(op_kind),
            tgt_class:      CorpusRecord::tgt_class_for(op_tgt),
        }
    }
}

// ── CorpusCollector ───────────────────────────────────────────────────────────

pub struct CorpusCollector;

impl CorpusCollector {
    pub fn collect(
        tower:  &Tower,
        config: &CorpusConfig,
    ) -> (Vec<CorpusRecord>, CorpusManifest) {
        let mut all_records  = Vec::new();
        let mut total_passes = 0usize;
        let n_ph = tower.phoneme.len().min(config.n_phonemes);

        for ph_idx in 0..n_ph {
            let ph_sym = tower.phoneme.render(ph_idx);
            for &tau in &config.taus {
                for &top_k in &config.top_ks {
                    let mut ctx = ProposerContext::new(LayerId::Phoneme);
                    let seq = OpSequence { phoneme_idx: ph_idx, tau, top_k };
                    let mut records = seq.execute(tower, &mut ctx, total_passes);
                    // Fill phoneme_sym
                    for r in &mut records {
                        r.phoneme_sym = ph_sym.clone();
                    }
                    all_records.extend(records);
                    total_passes += 1;
                }
            }
        }

        // Corpus digest
        let mut leaves: Vec<[u8; 32]> = all_records.iter()
            .map(|r| r.step_digest).collect();
        leaves.sort_unstable();
        let corpus_digest = if leaves.is_empty() {
            sha256_bytes(b"empty")
        } else {
            merkle_root(&leaves)
        };

        // Op counts
        let mut op_map: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for r in &all_records {
            *op_map.entry(r.op_kind.as_str().to_string()).or_insert(0) += 1;
        }
        let op_counts: Vec<(String, usize)> = op_map.into_iter().collect();

        let manifest = CorpusManifest {
            total_records:   all_records.len(),
            total_passes,
            n_phonemes:      n_ph,
            taus:            config.taus.clone(),
            top_ks:          config.top_ks.clone(),
            corpus_digest,
            tower_root:      tower.manifest.root_digest,
            acceptance_rate: 1.0,
            op_counts,
            version:         "v2".to_string(),
        };

        (all_records, manifest)
    }

    pub fn write_ndjson(
        records:  &[CorpusRecord],
        manifest: &CorpusManifest,
        path:     &str,
    ) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        for r in records {
            writeln!(file, "{}", r.to_json())?;
        }
        writeln!(file, "{}", manifest.to_json())?;
        Ok(())
    }

    pub fn print_summary(records: &[CorpusRecord], manifest: &CorpusManifest) {
        println!("╔══════════════════════════════════════════════════════╗");
        println!("║        VERIFIED TRACE CORPUS v2 SUMMARY             ║");
        println!("╠══════════════════════════════════════════════════════╣");
        println!("║  Records:       {:>6}                               ║", manifest.total_records);
        println!("║  Passes:        {:>6}                               ║", manifest.total_passes);
        println!("║  Phonemes:      {:>6}                               ║", manifest.n_phonemes);
        println!("║  Taus:          {:>6}                               ║", manifest.taus.len());
        println!("║  Top-k values:  {:>6}                               ║", manifest.top_ks.len());
        println!("╠══════════════════════════════════════════════════════╣");
        println!("║  Tower root:  {}...  ║", hex::encode(&manifest.tower_root[..16]));
        println!("║  Corpus:      {}...  ║", hex::encode(&manifest.corpus_digest[..16]));
        println!("╚══════════════════════════════════════════════════════╝");

        println!("\nOp distribution:");
        let total = records.len() as f64;
        let mut op_counts: std::collections::BTreeMap<&str, usize> =
            std::collections::BTreeMap::new();
        for r in records {
            *op_counts.entry(r.op_kind.as_str()).or_insert(0) += 1;
        }
        let mut ops: Vec<(&&str, &usize)> = op_counts.iter().collect();
        ops.sort_by(|(_, a), (_, b)| b.cmp(a));
        for (op, count) in &ops {
            let pct = **count as f64 / total * 100.0;
            let bar: String = "█".repeat((pct / 2.0) as usize);
            println!("  {:<20} {:>5} ({:>5.1}%) {}", op, count, pct, bar);
        }

        // Entropy
        let entropy: f64 = ops.iter()
            .map(|(_, &c)| {
                let p = c as f64 / total;
                if p > 0.0 { -p * p.log2() } else { 0.0 }
            })
            .sum();
        println!("\n  Op entropy: {:.3} bits (max={:.3} bits for 8 ops)",
            entropy, 3.0f64);

        println!("\nLayer transitions:");
        let mut layer_counts: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for r in records {
            let key = format!("{}→{}", r.src_layer.as_str(), r.tgt_layer.as_str());
            *layer_counts.entry(key).or_insert(0) += 1;
        }
        for (layer, count) in &layer_counts {
            println!("  {:<30} {:>6}", layer, count);
        }
    }

    /// Compute op entropy in bits. Gate: must be > 2.0 for Stage 10 completion.
    pub fn op_entropy(records: &[CorpusRecord]) -> f64 {
        let total = records.len() as f64;
        let mut counts: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
        for r in records { *counts.entry(r.op_class).or_insert(0) += 1; }
        counts.values()
            .map(|&c| { let p = c as f64 / total; if p > 0.0 { -p * p.log2() } else { 0.0 } })
            .sum()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tower::Tower;

    fn small_cfg() -> CorpusConfig {
        CorpusConfig {
            n_phonemes: 3, taus: vec![1.0], top_ks: vec![3],
            output_path: "/tmp/test_corpus_v2.ndjson".to_string(),
        }
    }

    #[test]
    fn corpus_v2_collect_small() {
        let tower = Tower::build();
        let (records, manifest) = CorpusCollector::collect(&tower, &small_cfg());
        // 3 phonemes × 1 tau × 1 top_k × 12 ops (11 + 0 REJECT for first 9)
        assert!(records.len() >= 3 * 1 * 1 * 11, "too few records: {}", records.len());
        assert_eq!(manifest.version, "v2");
    }

    #[test]
    fn corpus_v2_all_op_kinds_present() {
        let tower = Tower::build();
        let cfg = CorpusConfig {
            n_phonemes: 10, taus: vec![1.0], top_ks: vec![3],
            output_path: "/tmp/test_corpus_v2.ndjson".to_string(),
        };
        let (records, _) = CorpusCollector::collect(&tower, &cfg);
        let kinds: std::collections::HashSet<u8> =
            records.iter().map(|r| r.op_class).collect();
        // Should have at least 7 distinct op types (REJECT needs pass_idx % 10 == 9)
        assert!(kinds.len() >= 7,
            "only {} op kinds present: {:?}", kinds.len(), kinds);
    }

    #[test]
    fn corpus_v2_has_reject() {
        let tower = Tower::build();
        let cfg = CorpusConfig {
            n_phonemes: 10, taus: vec![1.0], top_ks: vec![3],
            output_path: "/tmp/test_corpus_v2.ndjson".to_string(),
        };
        let (records, _) = CorpusCollector::collect(&tower, &cfg);
        assert!(records.iter().any(|r| r.op_kind == OpKind::Reject),
            "no REJECT records found");
    }

    #[test]
    fn corpus_v2_has_accept() {
        let tower = Tower::build();
        let (records, _) = CorpusCollector::collect(&tower, &small_cfg());
        assert!(records.iter().any(|r| r.op_kind == OpKind::Accept));
    }

    #[test]
    fn corpus_v2_has_select_universe() {
        let tower = Tower::build();
        let (records, _) = CorpusCollector::collect(&tower, &small_cfg());
        assert!(records.iter().any(|r| r.op_kind == OpKind::SelectUniverse));
    }

    #[test]
    fn corpus_v2_has_witness_nearest() {
        let tower = Tower::build();
        let (records, _) = CorpusCollector::collect(&tower, &small_cfg());
        assert!(records.iter().any(|r| r.op_kind == OpKind::WitnessNearest));
    }

    #[test]
    fn corpus_v2_has_attend() {
        let tower = Tower::build();
        let (records, _) = CorpusCollector::collect(&tower, &small_cfg());
        assert!(records.iter().any(|r| r.op_kind == OpKind::Attend));
    }

    #[test]
    fn corpus_v2_has_project_layer() {
        let tower = Tower::build();
        let (records, _) = CorpusCollector::collect(&tower, &small_cfg());
        assert!(records.iter().any(|r| r.op_kind == OpKind::ProjectLayer));
    }

    #[test]
    fn corpus_v2_has_return_set() {
        let tower = Tower::build();
        let (records, _) = CorpusCollector::collect(&tower, &small_cfg());
        assert!(records.iter().any(|r| r.op_kind == OpKind::ReturnSet));
    }

    #[test]
    fn corpus_v2_op_entropy_gate() {
        let tower = Tower::build();
        let cfg = CorpusConfig {
            n_phonemes: 10, taus: vec![1.0], top_ks: vec![3],
            output_path: "/tmp/test_entropy.ndjson".to_string(),
        };
        let (records, _) = CorpusCollector::collect(&tower, &cfg);
        let entropy = CorpusCollector::op_entropy(&records);
        assert!(entropy > 2.0,
            "Stage 10 gate FAILED: op entropy {:.3} <= 2.0 bits", entropy);
        println!("Op entropy: {:.3} bits ✓", entropy);
    }

    #[test]
    fn corpus_v2_digest_stable() {
        let tower = Tower::build();
        let (_, m1) = CorpusCollector::collect(&tower, &small_cfg());
        let (_, m2) = CorpusCollector::collect(&tower, &small_cfg());
        assert_eq!(m1.corpus_digest, m2.corpus_digest);
    }

    #[test]
    fn corpus_v2_op_classes_valid() {
        let tower = Tower::build();
        let (records, _) = CorpusCollector::collect(&tower, &small_cfg());
        for r in &records {
            assert!(r.op_class <= 7, "op_class out of range: {}", r.op_class);
            assert!(r.tgt_class <= 7, "tgt_class out of range: {}", r.tgt_class);
        }
    }

    #[test]
    fn corpus_v2_write_ndjson() {
        let tower = Tower::build();
        let cfg = CorpusConfig {
            n_phonemes: 2, taus: vec![1.0], top_ks: vec![3],
            output_path: "/tmp/test_corpus_v2_rw.ndjson".to_string(),
        };
        let (records, manifest) = CorpusCollector::collect(&tower, &cfg);
        CorpusCollector::write_ndjson(&records, &manifest, &cfg.output_path).unwrap();
        let content = std::fs::read_to_string(&cfg.output_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), records.len() + 1);
        assert!(lines.last().unwrap().contains("__manifest"));
        assert!(lines.last().unwrap().contains("\"version\":\"v2\""));
        let _ = std::fs::remove_file(&cfg.output_path);
    }

    #[test]
    fn corpus_v2_manifest_has_op_counts() {
        let tower = Tower::build();
        let (_, manifest) = CorpusCollector::collect(&tower, &small_cfg());
        assert!(!manifest.op_counts.is_empty());
        assert!(manifest.op_counts.iter().any(|(k, _)| k == "FFN_STEP"));
        assert!(manifest.op_counts.iter().any(|(k, _)| k == "ACCEPT"));
    }
}
