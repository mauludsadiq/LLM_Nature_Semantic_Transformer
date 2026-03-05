//! Training data writer — Stage 12 of the Phase 2 blueprint.
//!
//! Reads corpus_v2.ndjson, encodes every record through FeatureEncoder,
//! writes binary training files for the PyTorch model.
//!
//! Output files:
//!   features.bin   f32 matrix [N × 32] little-endian
//!   labels.bin     u8 pairs   [N × 2]  (op_class, tgt_class)
//!   splits.json    {train, val, test indices, seed, class_weights}
//!
//! Stage 12 gate: max class imbalance < 100× (FFN_STEP structurally dominates: 6 per pass vs 1 REJECT per 10)

use std::io::{Write, BufRead};
use std::collections::HashMap;
use crate::features::{FeatureEncoder, FeatureMatrix, FEATURE_DIM, layer_index, op_index};
use crate::layer::LayerId;
use crate::proposer::OpKind;
use crate::digest::sha256_bytes;

// ── NdjsonRecord ──────────────────────────────────────────────────────────────

/// A parsed line from corpus_v2.ndjson.
#[derive(Clone, Debug)]
pub struct NdjsonRecord {
    pub active_layer:    LayerId,
    pub step_count:      usize,
    pub rejection_count: usize,
    pub pass_present:    bool,
    pub witness_present: bool,
    pub chain_hash:      [u8; 32],
    pub tau:             f64,
    pub top_k:           usize,
    pub op_class:        u8,
    pub tgt_class:       u8,
    pub op_kind:         String,
    pub phoneme_idx:     usize,
    pub block_idx:       usize,
}

impl NdjsonRecord {
    /// Parse a JSON line from corpus_v2.ndjson.
    /// Uses manual field extraction (no serde dependency).
    pub fn parse(line: &str) -> Option<Self> {
        if line.contains("__manifest") { return None; }

        let get_str = |key: &str| -> Option<String> {
            let pat = format!("\"{}\":\"", key);
            let start = line.find(&pat)? + pat.len();
            let end = line[start..].find('"')? + start;
            Some(line[start..end].to_string())
        };
        let get_num = |key: &str| -> Option<f64> {
            let pat = format!("\"{}\":", key);
            let start = line.find(&pat)? + pat.len();
            let end = line[start..].find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')?
                + start;
            line[start..end].parse().ok()
        };

        let active_layer = match get_str("active_layer")?.as_str() {
            "PHONEME"   => LayerId::Phoneme,
            "SYLLABLE"  => LayerId::Syllable,
            "MORPHEME"  => LayerId::Morpheme,
            "WORD"      => LayerId::Word,
            "PHRASE"    => LayerId::Phrase,
            "SEMANTIC"  => LayerId::Semantic,
            "DISCOURSE" => LayerId::Discourse,
            _           => LayerId::Phoneme,
        };

        let chain_hash_hex = get_str("chain_hash")?;
        let mut chain_hash = [0u8; 32];
        for i in 0..32 {
            let byte = u8::from_str_radix(&chain_hash_hex[i*2..i*2+2], 16).ok()?;
            chain_hash[i] = byte;
        }

        Some(NdjsonRecord {
            active_layer,
            step_count:      get_num("step_count")? as usize,
            rejection_count: 0,
            pass_present:    false,
            witness_present: false,
            chain_hash,
            tau:             get_num("tau")?,
            top_k:           get_num("top_k")? as usize,
            op_class:        get_num("op_class")? as u8,
            tgt_class:       get_num("tgt_class")? as u8,
            op_kind:         get_str("op_kind").unwrap_or_default(),
            phoneme_idx:     get_num("phoneme_idx")? as usize,
            block_idx:       get_num("block_idx")? as usize,
        })
    }

    pub fn to_features(&self) -> [f32; FEATURE_DIM] {
        let is_terminal = self.op_kind == "ACCEPT" || self.op_kind == "REJECT";
        FeatureEncoder::encode_raw_full(
            self.active_layer,
            self.step_count,
            self.rejection_count,
            self.pass_present,
            self.witness_present,
            &self.chain_hash,
            self.tau,
            self.top_k,
            self.block_idx,
            is_terminal,
        )
    }
}

// ── DataSplit ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct DataSplit {
    pub train_indices: Vec<usize>,
    pub val_indices:   Vec<usize>,
    pub test_indices:  Vec<usize>,
    pub seed:          u64,
    pub n_total:       usize,
}

impl DataSplit {
    /// Split N indices into train/val/test using deterministic shuffle.
    pub fn split(n: usize, train_frac: f64, val_frac: f64, seed: u64) -> Self {
        // Deterministic shuffle via LCG
        let mut order: Vec<usize> = (0..n).collect();
        let mut state = seed;
        for i in (1..n).rev() {
            state = state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (state >> 33) as usize % (i + 1);
            order.swap(i, j);
        }

        let n_train = (n as f64 * train_frac) as usize;
        let n_val   = (n as f64 * val_frac) as usize;

        DataSplit {
            train_indices: order[..n_train].to_vec(),
            val_indices:   order[n_train..n_train+n_val].to_vec(),
            test_indices:  order[n_train+n_val..].to_vec(),
            seed,
            n_total: n,
        }
    }

    pub fn to_json(&self, class_weights: &[f64; 8]) -> String {
        let weights_str: Vec<String> = class_weights.iter()
            .map(|w| format!("{:.6}", w)).collect();
        format!(
            "{{\"class_weights\":[{}],\"n_test\":{},\"n_total\":{},\
             \"n_train\":{},\"n_val\":{},\"seed\":{},\
             \"test_frac\":{:.3},\"train_frac\":{:.3},\"val_frac\":{:.3}}}",
            weights_str.join(","),
            self.test_indices.len(),
            self.n_total,
            self.train_indices.len(),
            self.val_indices.len(),
            self.seed,
            self.test_indices.len() as f64 / self.n_total as f64,
            self.train_indices.len() as f64 / self.n_total as f64,
            self.val_indices.len() as f64 / self.n_total as f64,
        )
    }
}

// ── ClassStats ────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ClassStats {
    pub op_counts:    [usize; 8],
    pub tgt_counts:   [usize; 8],
    pub total:        usize,
    pub max_imbalance: f64,
    /// Inverse-frequency class weights for loss weighting.
    pub class_weights: [f64; 8],
}

impl ClassStats {
    pub fn compute(matrix: &FeatureMatrix) -> Self {
        let mut op_counts  = [0usize; 8];
        let mut tgt_counts = [0usize; 8];
        for &(op, tgt) in &matrix.labels {
            if (op as usize) < 8 { op_counts[op as usize] += 1; }
            if (tgt as usize) < 8 { tgt_counts[tgt as usize] += 1; }
        }
        let total = matrix.len();
        let max_count = *op_counts.iter().max().unwrap_or(&1) as f64;
        let min_count = op_counts.iter().filter(|&&x| x > 0)
            .min().copied().unwrap_or(1) as f64;
        let max_imbalance = if min_count > 0.0 { max_count / min_count } else { f64::INFINITY };

        // Inverse frequency weights (normalized so mean weight = 1.0)
        let mut class_weights = [1.0f64; 8];
        let total_f = total as f64;
        for i in 0..8 {
            if op_counts[i] > 0 {
                class_weights[i] = total_f / (8.0 * op_counts[i] as f64);
            }
        }

        ClassStats { op_counts, tgt_counts, total, max_imbalance, class_weights }
    }

    pub fn print_report(&self) {
        let op_names = ["SELECT_UNIVERSE", "WITNESS_NEAREST", "ATTEND",
                        "FFN_STEP", "PROJECT_LAYER", "RETURN_SET", "ACCEPT", "REJECT"];
        println!("\nClass distribution (op_kind):");
        for (i, name) in op_names.iter().enumerate() {
            let count = self.op_counts[i];
            let pct   = count as f64 / self.total as f64 * 100.0;
            let w     = self.class_weights[i];
            println!("  [{i}] {name:<20} {count:>5} ({pct:>5.1}%)  weight={w:.3}");
        }
        println!("\n  Total: {}", self.total);
        println!("  Max imbalance: {:.1}×", self.max_imbalance);
        if self.max_imbalance < 100.0 {
            println!("  Stage 12 gate: ✓ PASSED (< 100×, structural imbalance expected)");
        } else {
            println!("  Stage 12 gate: ✗ FAILED (>= 100×) — needs rebalancing");
        }
    }
}

// ── TrainingDataWriter ────────────────────────────────────────────────────────

pub struct TrainingDataWriter;

impl TrainingDataWriter {
    /// Load corpus NDJSON, encode features, build FeatureMatrix.
    pub fn load_corpus(ndjson_path: &str) -> Result<FeatureMatrix, String> {
        let file = std::fs::File::open(ndjson_path)
            .map_err(|e| format!("open {}: {}", ndjson_path, e))?;
        let reader = std::io::BufReader::new(file);
        let mut matrix = FeatureMatrix::new();
        let mut skipped = 0usize;

        for line in reader.lines() {
            let line = line.map_err(|e| format!("read: {}", e))?;
            if line.trim().is_empty() { continue; }
            match NdjsonRecord::parse(&line) {
                Some(rec) => {
                    let features = rec.to_features();
                    matrix.push(features, rec.op_class, rec.tgt_class);
                }
                None => { skipped += 1; }
            }
        }

        if matrix.is_empty() {
            return Err("no records loaded".to_string());
        }
        eprintln!("Loaded {} records ({} skipped)", matrix.len(), skipped);
        Ok(matrix)
    }

    /// Write features.bin, labels.bin, splits.json to output_dir.
    pub fn write(
        matrix:     &FeatureMatrix,
        output_dir: &str,
        seed:       u64,
    ) -> Result<DataSplit, String> {
        std::fs::create_dir_all(output_dir)
            .map_err(|e| format!("mkdir {}: {}", output_dir, e))?;

        // features.bin
        let features_path = format!("{}/features.bin", output_dir);
        let mut f = std::fs::File::create(&features_path)
            .map_err(|e| format!("create features.bin: {}", e))?;
        f.write_all(&matrix.to_feature_bytes())
            .map_err(|e| format!("write features.bin: {}", e))?;

        // labels.bin
        let labels_path = format!("{}/labels.bin", output_dir);
        let mut f = std::fs::File::create(&labels_path)
            .map_err(|e| format!("create labels.bin: {}", e))?;
        f.write_all(&matrix.to_label_bytes())
            .map_err(|e| format!("write labels.bin: {}", e))?;

        // splits.json
        let stats  = ClassStats::compute(matrix);
        let split  = DataSplit::split(matrix.len(), 0.8, 0.1, seed);
        let splits_path = format!("{}/splits.json", output_dir);
        let mut f = std::fs::File::create(&splits_path)
            .map_err(|e| format!("create splits.json: {}", e))?;
        writeln!(f, "{}", split.to_json(&stats.class_weights))
            .map_err(|e| format!("write splits.json: {}", e))?;

        // manifest.json
        let features_bytes = matrix.to_feature_bytes();
        let labels_bytes   = matrix.to_label_bytes();
        let manifest = format!(
            "{{\"features_digest\":\"{}\",\"features_shape\":[{},{}],\
             \"labels_digest\":\"{}\",\"labels_shape\":[{},2],\
             \"max_imbalance\":{:.3},\"n_records\":{},\"seed\":{}}}",
            hex::encode(sha256_bytes(&features_bytes)),
            matrix.len(), FEATURE_DIM,
            hex::encode(sha256_bytes(&labels_bytes)),
            matrix.len(),
            stats.max_imbalance,
            matrix.len(),
            seed,
        );
        let manifest_path = format!("{}/manifest.json", output_dir);
        std::fs::write(&manifest_path, &manifest)
            .map_err(|e| format!("write manifest.json: {}", e))?;

        eprintln!("Wrote {}/features.bin  ({} bytes)", output_dir, features_bytes.len());
        eprintln!("Wrote {}/labels.bin    ({} bytes)", output_dir, labels_bytes.len());
        eprintln!("Wrote {}/splits.json", output_dir);
        eprintln!("Wrote {}/manifest.json", output_dir);

        Ok(split)
    }

    /// Full pipeline: load → verify → write → report.
    pub fn run(
        ndjson_path: &str,
        output_dir:  &str,
        seed:        u64,
    ) -> Result<ClassStats, String> {
        let matrix = Self::load_corpus(ndjson_path)?;
        matrix.verify_all()
            .map_err(|e| format!("feature verify: {}", e))?;
        let stats = ClassStats::compute(&matrix);
        let _split = Self::write(&matrix, output_dir, seed)?;
        Ok(stats)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{FeatureEncoder, FeatureMatrix};
    use crate::layer::LayerId;
    use crate::proposer::ProposerContext;

    fn make_matrix(n: usize) -> FeatureMatrix {
        let mut mat = FeatureMatrix::new();
        for i in 0..n {
            let mut ctx = ProposerContext::new(LayerId::Phoneme);
            ctx.step_count = i % 12;
            let v = FeatureEncoder::encode(&ctx, 1.0, 3);
            mat.push(v, (i % 8) as u8, (i % 7) as u8);
        }
        mat
    }

    #[test]
    fn data_split_sizes_correct() {
        let split = DataSplit::split(100, 0.8, 0.1, 42);
        assert_eq!(split.train_indices.len(), 80);
        assert_eq!(split.val_indices.len(),   10);
        assert_eq!(split.test_indices.len(),  10);
        assert_eq!(split.n_total, 100);
    }

    #[test]
    fn data_split_deterministic() {
        let s1 = DataSplit::split(100, 0.8, 0.1, 42);
        let s2 = DataSplit::split(100, 0.8, 0.1, 42);
        assert_eq!(s1.train_indices, s2.train_indices);
        assert_eq!(s1.val_indices,   s2.val_indices);
        assert_eq!(s1.test_indices,  s2.test_indices);
    }

    #[test]
    fn data_split_different_seeds_differ() {
        let s1 = DataSplit::split(100, 0.8, 0.1, 42);
        let s2 = DataSplit::split(100, 0.8, 0.1, 99);
        assert_ne!(s1.train_indices, s2.train_indices);
    }

    #[test]
    fn data_split_no_overlap() {
        let split = DataSplit::split(100, 0.8, 0.1, 42);
        let mut all: Vec<usize> = split.train_indices.iter()
            .chain(&split.val_indices)
            .chain(&split.test_indices)
            .copied().collect();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 100, "overlapping indices detected");
    }

    #[test]
    fn class_stats_compute_balanced() {
        let mat = make_matrix(80); // 10 per class
        let stats = ClassStats::compute(&mat);
        assert_eq!(stats.total, 80);
        assert!(stats.max_imbalance < 2.0,
            "imbalance too high: {}", stats.max_imbalance);
    }

    #[test]
    fn class_stats_weights_nonzero() {
        let mat = make_matrix(80);
        let stats = ClassStats::compute(&mat);
        for w in &stats.class_weights {
            assert!(*w > 0.0, "zero weight found");
        }
    }

    #[test]
    fn class_stats_gate_balanced() {
        let mat = make_matrix(80);
        let stats = ClassStats::compute(&mat);
        assert!(stats.max_imbalance < 100.0,
            "Stage 12 gate FAILED: imbalance={:.1}", stats.max_imbalance);
    }

    #[test]
    fn writer_write_creates_files() {
        let mat   = make_matrix(80);
        let dir   = "/tmp/test_training_data";
        let split = TrainingDataWriter::write(&mat, dir, 42).unwrap();
        assert!(std::path::Path::new(&format!("{}/features.bin", dir)).exists());
        assert!(std::path::Path::new(&format!("{}/labels.bin", dir)).exists());
        assert!(std::path::Path::new(&format!("{}/splits.json", dir)).exists());
        assert!(std::path::Path::new(&format!("{}/manifest.json", dir)).exists());
        // Clean up
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn writer_features_bin_size() {
        let mat = make_matrix(10);
        let dir = "/tmp/test_feat_size";
        TrainingDataWriter::write(&mat, dir, 42).unwrap();
        let meta = std::fs::metadata(format!("{}/features.bin", dir)).unwrap();
        assert_eq!(meta.len(), (10 * FEATURE_DIM * 4) as u64);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn writer_labels_bin_size() {
        let mat = make_matrix(10);
        let dir = "/tmp/test_lbl_size";
        TrainingDataWriter::write(&mat, dir, 42).unwrap();
        let meta = std::fs::metadata(format!("{}/labels.bin", dir)).unwrap();
        assert_eq!(meta.len(), (10 * 2) as u64);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn writer_features_roundtrip() {
        let mat = make_matrix(4);
        let dir = "/tmp/test_feat_rt";
        TrainingDataWriter::write(&mat, dir, 42).unwrap();
        let bytes = std::fs::read(format!("{}/features.bin", dir)).unwrap();
        // Re-parse first row
        let mut row = [0f32; FEATURE_DIM];
        for i in 0..FEATURE_DIM {
            let b = &bytes[i*4..(i+1)*4];
            row[i] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
        assert_eq!(row, mat.data[0]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn writer_splits_json_valid() {
        let mat = make_matrix(100);
        let dir = "/tmp/test_splits";
        TrainingDataWriter::write(&mat, dir, 42).unwrap();
        let content = std::fs::read_to_string(format!("{}/splits.json", dir)).unwrap();
        assert!(content.contains("n_train"));
        assert!(content.contains("n_val"));
        assert!(content.contains("n_test"));
        assert!(content.contains("class_weights"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn ndjson_record_parse_manifest_returns_none() {
        let line = r#"{"__manifest":true,"total_records":100}"#;
        assert!(NdjsonRecord::parse(line).is_none());
    }

    #[test]
    fn ndjson_record_parse_valid_line() {
        // Minimal valid record
        let line = r#"{"active_layer":"WORD","attn_top":"word:cat","block_digest":"0000000000000000000000000000000000000000000000000000000000000000","block_idx":3,"chain_hash":"7edf727d6542a8950abdaa8b6e94ff53e072b98e5d53ddf879bd0c3b11657552","context_digest":"0000000000000000000000000000000000000000000000000000000000000000","ffn_top":"phrase:1","op_class":3,"op_kind":"FFN_STEP","op_tgt_layer":"PHRASE","phoneme_idx":0,"phoneme_sym":"ph:P","src_layer":"WORD","step_count":6,"step_digest":"0000000000000000000000000000000000000000000000000000000000000000","tau":1.00,"tgt_class":4,"tgt_layer":"PHRASE","top_k":3}"#;
        let rec = NdjsonRecord::parse(line).unwrap();
        assert_eq!(rec.active_layer, LayerId::Word);
        assert_eq!(rec.step_count, 6);
        assert_eq!(rec.op_class, 3);
        assert_eq!(rec.tgt_class, 4);
        assert!((rec.tau - 1.0).abs() < 1e-6);
        assert_eq!(rec.top_k, 3);
    }

    #[test]
    fn load_corpus_from_real_file() {
        // Only run if corpus_v2.ndjson exists
        let path = "corpus_v2.ndjson";
        if !std::path::Path::new(path).exists() { return; }
        let matrix = TrainingDataWriter::load_corpus(path).unwrap();
        assert!(matrix.len() > 100, "too few records: {}", matrix.len());
        matrix.verify_all().unwrap();
        let stats = ClassStats::compute(&matrix);
        assert!(stats.max_imbalance < 100.0,
            "Stage 12 gate FAILED: imbalance={:.1}", stats.max_imbalance);
        println!("Loaded {} records, imbalance={:.1}×", matrix.len(), stats.max_imbalance);
    }

    #[test]
    fn full_pipeline_from_real_corpus() {
        let path = "corpus_v2.ndjson";
        if !std::path::Path::new(path).exists() { return; }
        let dir = "/tmp/test_full_pipeline";
        let stats = TrainingDataWriter::run(path, dir, 42).unwrap();
        assert!(stats.max_imbalance < 100.0);
        assert!(std::path::Path::new(&format!("{}/features.bin", dir)).exists());
        assert!(std::path::Path::new(&format!("{}/labels.bin", dir)).exists());
        stats.print_report();
        let _ = std::fs::remove_dir_all(dir);
    }
}
