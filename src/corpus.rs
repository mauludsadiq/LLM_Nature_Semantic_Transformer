//! Verified trace corpus collector.
//!
//! Runs real Tower forward passes and records every verified step as a
//! TraceRecord for imitation learning. Saves corpus to NDJSON on disk.
//!
//! Corpus structure:
//!   Each line = one TraceRecord encoded as JSON
//!   Final line = corpus manifest (digest, count, stats)
//!
//! Training pipeline:
//!   1. Run collect() → corpus.ndjson
//!   2. Load corpus.ndjson → (context, accepted_op) pairs
//!   3. Train small transformer: P(op | context)
//!   4. Replace RuleBasedProposer with learned model
//!
//! Corpus size target: 1000+ records across all phonemes, taus, top_k values.

use std::io::Write;
use crate::digest::{sha256_bytes, merkle_root};
use crate::layer::{Layer, LayerId};
use crate::tower::Tower;
use crate::proposer::{
    ProposerContext, ProposerTrainer, TraceRecord,
    RuleBasedProposer, OpKind, ProposedOp, OpDistribution,
};
use crate::transformer::TowerTransformer;
use crate::feedforward::TowerFFN;
use crate::attention::CertifiedAttention;
use crate::edges::EdgeId;

// ── CorpusConfig ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct CorpusConfig {
    /// Number of phonemes to use as starting points (max 44).
    pub n_phonemes:  usize,
    /// Temperature values to sweep.
    pub taus:        Vec<f64>,
    /// Top-k values to sweep.
    pub top_ks:      Vec<usize>,
    /// Output path for corpus NDJSON.
    pub output_path: String,
}

impl CorpusConfig {
    pub fn default_1k() -> Self {
        CorpusConfig {
            n_phonemes:  44,          // all phonemes
            taus:        vec![0.5, 1.0, 2.0, 4.0],
            top_ks:      vec![3, 5],
            output_path: "corpus.ndjson".to_string(),
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

    /// Expected record count: phonemes × taus × top_ks × 6 steps per pass.
    pub fn expected_records(&self) -> usize {
        self.n_phonemes * self.taus.len() * self.top_ks.len() * 6
    }
}

// ── CorpusRecord ──────────────────────────────────────────────────────────────

/// A single NDJSON line in the corpus.
#[derive(Clone, Debug)]
pub struct CorpusRecord {
    /// Which phoneme started this pass.
    pub phoneme_idx:   usize,
    pub phoneme_sym:   String,
    /// Configuration for this pass.
    pub tau:           f64,
    pub top_k:         usize,
    /// Which block in the forward pass (0=PH→SYL, ..., 5=SEM→DISC).
    pub block_idx:     usize,
    pub src_layer:     LayerId,
    pub tgt_layer:     LayerId,
    /// The proposer context at this step.
    pub context_digest: [u8; 32],
    pub chain_hash:    [u8; 32],
    pub active_layer:  LayerId,
    pub step_count:    usize,
    /// The op that was executed and verified.
    pub op_kind:       OpKind,
    pub op_tgt_layer:  Option<LayerId>,
    /// The verified step digest from the executor.
    pub step_digest:   [u8; 32],
    /// Attention top result rendered.
    pub attn_top:      String,
    /// FFN top result rendered.
    pub ffn_top:       String,
    /// Block digest (chained).
    pub block_digest:  [u8; 32],
}

impl CorpusRecord {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"active_layer\":\"{}\",\"attn_top\":\"{}\",\"block_digest\":\"{}\",\
             \"block_idx\":{},\"chain_hash\":\"{}\",\"context_digest\":\"{}\",\
             \"ffn_top\":\"{}\",\"op_kind\":\"{}\",\"op_tgt_layer\":\"{}\",\
             \"phoneme_idx\":{},\"phoneme_sym\":\"{}\",\"src_layer\":\"{}\",\
             \"step_count\":{},\"step_digest\":\"{}\",\"tau\":{:.2},\
             \"tgt_layer\":\"{}\",\"top_k\":{}}}",
            self.active_layer.as_str(),
            self.attn_top.replace('"', "'"),
            hex::encode(self.block_digest),
            self.block_idx,
            hex::encode(self.chain_hash),
            hex::encode(self.context_digest),
            self.ffn_top.replace('"', "'"),
            self.op_kind.as_str(),
            self.op_tgt_layer.map(|l| l.as_str()).unwrap_or("none"),
            self.phoneme_idx,
            self.phoneme_sym,
            self.src_layer.as_str(),
            self.step_count,
            hex::encode(self.step_digest),
            self.tau,
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
}

impl CorpusManifest {
    pub fn to_json(&self) -> String {
        let taus_str: Vec<String> = self.taus.iter().map(|t| format!("{:.2}", t)).collect();
        format!(
            "{{\"__manifest\":true,\"acceptance_rate\":{:.4},\"corpus_digest\":\"{}\",\
             \"n_phonemes\":{},\"taus\":[{}],\"top_ks\":{:?},\
             \"total_passes\":{},\"total_records\":{},\"tower_root\":\"{}\"}}",
            self.acceptance_rate,
            hex::encode(self.corpus_digest),
            self.n_phonemes,
            taus_str.join(","),
            self.top_ks,
            self.total_passes,
            self.total_records,
            hex::encode(self.tower_root),
        )
    }
}

// ── CorpusCollector ───────────────────────────────────────────────────────────

pub struct CorpusCollector;

impl CorpusCollector {
    /// Collect a verified trace corpus from the tower.
    /// Returns (records, manifest).
    pub fn collect(
        tower:  &Tower,
        config: &CorpusConfig,
    ) -> (Vec<CorpusRecord>, CorpusManifest) {
        let mut records  = Vec::new();
        let mut trainer  = ProposerTrainer::new();
        let n_ph         = tower.phoneme.len().min(config.n_phonemes);
        let mut total_passes = 0usize;

        for ph_idx in 0..n_ph {
            let ph_sym = tower.phoneme.render(ph_idx);

            for &tau in &config.taus {
                for &top_k in &config.top_ks {
                    // Run a real verified forward pass
                    let pass = match TowerTransformer::forward_pass(
                        ph_idx,
                        &tower.phoneme, &tower.syllable, &tower.morpheme,
                        &tower.word, &tower.phrase, &tower.semantic, &tower.discourse,
                        tau, top_k,
                    ) {
                        Ok(p)  => p,
                        Err(_) => continue,
                    };

                    total_passes += 1;

                    // Build proposer context for this pass
                    let mut ctx = ProposerContext::new(LayerId::Phoneme);

                    // Record one CorpusRecord per block
                    for (block_idx, block) in pass.blocks.iter().enumerate() {
                        let op_kind   = OpKind::FFNStep;
                        let op_tgt    = Some(block.tgt_layer);

                        let attn_top = block.attention.weights.first()
                            .map(|w| w.rendered.clone())
                            .unwrap_or_default();
                        let ffn_top = block.ffn.attention.weights.first()
                            .map(|w| w.rendered.clone())
                            .unwrap_or_default();

                        let record = CorpusRecord {
                            phoneme_idx:    ph_idx,
                            phoneme_sym:    ph_sym.clone(),
                            tau,
                            top_k,
                            block_idx,
                            src_layer:      block.src_layer,
                            tgt_layer:      block.tgt_layer,
                            context_digest: ctx.digest(),
                            chain_hash:     ctx.chain_hash,
                            active_layer:   ctx.active_layer,
                            step_count:     ctx.step_count,
                            op_kind,
                            op_tgt_layer:   op_tgt,
                            step_digest:    block.ffn.step_digest,
                            attn_top,
                            ffn_top,
                            block_digest:   block.block_digest,
                        };

                        // Record in trainer for stats
                        let dist = RuleBasedProposer::propose(&ctx);
                        let proposed_op = ProposedOp {
                            kind:      op_kind,
                            tgt_layer: op_tgt,
                            query_sig: None,
                            tau,
                            log_score: 1.0,
                        };
                        trainer.record(crate::proposer::TraceRecord {
                            context:      ctx.clone(),
                            accepted_op:  proposed_op,
                            distribution: dist,
                            step_digest:  block.ffn.step_digest,
                            first_try:    true,
                        });

                        ctx.advance(&block.ffn.step_digest, block.tgt_layer);
                        records.push(record);
                    }
                }
            }
        }

        // Build corpus digest over all step digests
        let mut leaves: Vec<[u8; 32]> = records.iter()
            .map(|r| r.step_digest)
            .collect();
        leaves.sort_unstable();
        let corpus_digest = if leaves.is_empty() {
            sha256_bytes(b"empty")
        } else {
            merkle_root(&leaves)
        };

        let manifest = CorpusManifest {
            total_records:   records.len(),
            total_passes,
            n_phonemes:      n_ph,
            taus:            config.taus.clone(),
            top_ks:          config.top_ks.clone(),
            corpus_digest,
            tower_root:      tower.manifest.root_digest,
            acceptance_rate: trainer.acceptance_rate(),
        };

        (records, manifest)
    }

    /// Write corpus to NDJSON file.
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

    /// Print a summary of the corpus to stdout.
    pub fn print_summary(records: &[CorpusRecord], manifest: &CorpusManifest) {
        println!("╔══════════════════════════════════════════════════════╗");
        println!("║           VERIFIED TRACE CORPUS SUMMARY             ║");
        println!("╠══════════════════════════════════════════════════════╣");
        println!("║  Records:       {:>6}                               ║", manifest.total_records);
        println!("║  Passes:        {:>6}                               ║", manifest.total_passes);
        println!("║  Phonemes:      {:>6}                               ║", manifest.n_phonemes);
        println!("║  Taus:          {:>6}                               ║", manifest.taus.len());
        println!("║  Top-k values:  {:>6}                               ║", manifest.top_ks.len());
        println!("║  Acceptance:  {:>6.1}%                               ║",
            manifest.acceptance_rate * 100.0);
        println!("╠══════════════════════════════════════════════════════╣");
        println!("║  Tower root:  {}...  ║", hex::encode(&manifest.tower_root[..16]));
        println!("║  Corpus:      {}...  ║", hex::encode(&manifest.corpus_digest[..16]));
        println!("╚══════════════════════════════════════════════════════╝");

        // Op distribution
        let mut op_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for r in records {
            *op_counts.entry(r.op_kind.as_str()).or_insert(0) += 1;
        }
        println!("\nOp distribution:");
        let mut ops: Vec<(&&str, &usize)> = op_counts.iter().collect();
        ops.sort_by(|(_, a), (_, b)| b.cmp(a));
        for (op, count) in ops {
            println!("  {:<20} {:>6} ({:.1}%)",
                op, count, *count as f64 / records.len() as f64 * 100.0);
        }

        // Layer transition distribution
        println!("\nLayer transitions:");
        let mut layer_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for r in records {
            let key = format!("{}→{}", r.src_layer.as_str(), r.tgt_layer.as_str());
            *layer_counts.entry(key).or_insert(0) += 1;
        }
        let mut layers: Vec<(String, usize)> = layer_counts.into_iter().collect();
        layers.sort_by(|(ka, _), (kb, _)| ka.cmp(kb));
        for (layer, count) in layers {
            println!("  {:<25} {:>6}", layer, count);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tower::Tower;

    #[test]
    fn corpus_config_expected_records() {
        let cfg = CorpusConfig {
            n_phonemes: 5, taus: vec![1.0, 2.0], top_ks: vec![3],
            output_path: "test.ndjson".to_string(),
        };
        assert_eq!(cfg.expected_records(), 5 * 2 * 1 * 6);
    }

    #[test]
    fn corpus_collect_small() {
        let tower = Tower::build();
        let cfg   = CorpusConfig {
            n_phonemes: 3, taus: vec![1.0], top_ks: vec![3],
            output_path: "/tmp/test_corpus.ndjson".to_string(),
        };
        let (records, manifest) = CorpusCollector::collect(&tower, &cfg);
        assert_eq!(records.len(), 3 * 1 * 1 * 6, "expected 18 records");
        assert_eq!(manifest.total_records, records.len());
        assert_eq!(manifest.n_phonemes, 3);
    }

    #[test]
    fn corpus_collect_digest_stable() {
        let tower  = Tower::build();
        let cfg    = CorpusConfig {
            n_phonemes: 2, taus: vec![1.0], top_ks: vec![3],
            output_path: "/tmp/test_corpus.ndjson".to_string(),
        };
        let (_, m1) = CorpusCollector::collect(&tower, &cfg);
        let (_, m2) = CorpusCollector::collect(&tower, &cfg);
        assert_eq!(m1.corpus_digest, m2.corpus_digest);
    }

    #[test]
    fn corpus_collect_tower_root_matches() {
        let tower  = Tower::build();
        let cfg    = CorpusConfig {
            n_phonemes: 2, taus: vec![1.0], top_ks: vec![3],
            output_path: "/tmp/test_corpus.ndjson".to_string(),
        };
        let (_, manifest) = CorpusCollector::collect(&tower, &cfg);
        assert_eq!(manifest.tower_root, tower.manifest.root_digest);
    }

    #[test]
    fn corpus_records_all_verified_layers() {
        let tower  = Tower::build();
        let cfg    = CorpusConfig {
            n_phonemes: 2, taus: vec![1.0], top_ks: vec![3],
            output_path: "/tmp/test_corpus.ndjson".to_string(),
        };
        let (records, _) = CorpusCollector::collect(&tower, &cfg);
        // All 6 block indices present
        for bi in 0..6 {
            assert!(records.iter().any(|r| r.block_idx == bi),
                "block_idx {} not found", bi);
        }
    }

    #[test]
    fn corpus_records_json_parseable() {
        let tower  = Tower::build();
        let cfg    = CorpusConfig {
            n_phonemes: 1, taus: vec![1.0], top_ks: vec![3],
            output_path: "/tmp/test_corpus.ndjson".to_string(),
        };
        let (records, _) = CorpusCollector::collect(&tower, &cfg);
        for r in &records {
            let json = r.to_json();
            // Basic checks: starts with {, ends with }, has key fields
            assert!(json.starts_with('{'), "not JSON: {}", &json[..20]);
            assert!(json.ends_with('}'));
            assert!(json.contains("block_idx"));
            assert!(json.contains("step_digest"));
        }
    }

    #[test]
    fn corpus_write_and_read_ndjson() {
        let tower  = Tower::build();
        let cfg    = CorpusConfig {
            n_phonemes: 2, taus: vec![1.0], top_ks: vec![3],
            output_path: "/tmp/test_corpus_rw.ndjson".to_string(),
        };
        let (records, manifest) = CorpusCollector::collect(&tower, &cfg);
        CorpusCollector::write_ndjson(&records, &manifest, &cfg.output_path).unwrap();

        // Read back and count lines
        let content = std::fs::read_to_string(&cfg.output_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        // records + 1 manifest line
        assert_eq!(lines.len(), records.len() + 1);
        // Last line is manifest
        assert!(lines.last().unwrap().contains("__manifest"));
        // Clean up
        let _ = std::fs::remove_file(&cfg.output_path);
    }

    #[test]
    fn corpus_manifest_json_nonempty() {
        let tower  = Tower::build();
        let cfg    = CorpusConfig {
            n_phonemes: 1, taus: vec![1.0], top_ks: vec![3],
            output_path: "/tmp/test_corpus.ndjson".to_string(),
        };
        let (_, manifest) = CorpusCollector::collect(&tower, &cfg);
        let json = manifest.to_json();
        assert!(json.contains("__manifest"));
        assert!(json.contains("total_records"));
        assert!(json.contains("corpus_digest"));
    }

    #[test]
    fn corpus_full_44_phonemes_count() {
        let tower  = Tower::build();
        let cfg    = CorpusConfig {
            n_phonemes: 44, taus: vec![1.0], top_ks: vec![3],
            output_path: "/tmp/test_corpus_full.ndjson".to_string(),
        };
        let (records, manifest) = CorpusCollector::collect(&tower, &cfg);
        assert_eq!(records.len(), 44 * 1 * 1 * 6, "expected 264 records");
        assert_eq!(manifest.total_passes, 44);
        println!("Full corpus: {} records, digest={}",
            records.len(), hex::encode(&manifest.corpus_digest[..8]));
    }

    #[test]
    fn corpus_multi_tau_increases_records() {
        let tower  = Tower::build();
        let cfg    = CorpusConfig {
            n_phonemes: 5, taus: vec![0.5, 1.0, 2.0, 4.0], top_ks: vec![3, 5],
            output_path: "/tmp/test_corpus_multi.ndjson".to_string(),
        };
        let (records, _) = CorpusCollector::collect(&tower, &cfg);
        assert_eq!(records.len(), 5 * 4 * 2 * 6, "expected 240 records");
    }
}
