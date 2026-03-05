//! Generate training_data/ from corpus_v2.ndjson.
//! Usage: cargo run --release --bin gen_training_data

use llm_nature_semantic_transformer::training::TrainingDataWriter;

fn main() {
    let corpus_path = "corpus_v2.ndjson";
    let output_dir  = "training_data";
    let seed        = 42u64;

    if !std::path::Path::new(corpus_path).exists() {
        eprintln!("ERROR: {} not found. Run gen_corpus first.", corpus_path);
        std::process::exit(1);
    }

    println!("Loading corpus from {}...", corpus_path);
    let stats = TrainingDataWriter::run(corpus_path, output_dir, seed)
        .expect("training data generation failed");

    stats.print_report();

    println!("\nOutput files:");
    for f in &["features.bin", "labels.bin", "splits.json", "manifest.json"] {
        let path = format!("{}/{}", output_dir, f);
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        println!("  {:<40} {:>8} bytes", path, size);
    }

    if stats.max_imbalance < 100.0 {
        println!("\nStage 12 gate: ✓ PASSED (imbalance={:.1}×)", stats.max_imbalance);
    } else {
        println!("\nStage 12 gate: ✗ FAILED (imbalance={:.1}×)", stats.max_imbalance);
        std::process::exit(1);
    }
}
