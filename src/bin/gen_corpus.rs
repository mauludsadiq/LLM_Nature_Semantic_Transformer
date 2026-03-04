//! Generate the verified trace corpus v2 (full op vocabulary).
//! Usage: cargo run --release --bin gen_corpus

use llm_nature_semantic_transformer::tower::Tower;
use llm_nature_semantic_transformer::corpus::{CorpusCollector, CorpusConfig};

fn main() {
    println!("Building tower...");
    let tower = Tower::build();
    println!("Tower root: {}", hex::encode(&tower.manifest.root_digest[..16]));

    let config = CorpusConfig::v2();
    println!(
        "Collecting corpus v2: {} phonemes × {} taus × {} top_ks → ~{} records",
        config.n_phonemes,
        config.taus.len(),
        config.top_ks.len(),
        config.expected_records(),
    );

    let t0 = std::time::Instant::now();
    let (records, manifest) = CorpusCollector::collect(&tower, &config);
    let elapsed = t0.elapsed();

    println!("Collected {} records in {:.2}s", records.len(), elapsed.as_secs_f64());

    CorpusCollector::print_summary(&records, &manifest);

    CorpusCollector::write_ndjson(&records, &manifest, &config.output_path)
        .expect("failed to write corpus");

    println!("\nWrote {}", config.output_path);
    println!("Corpus digest: {}", hex::encode(manifest.corpus_digest));

    // Stage 10 gate check
    let entropy = CorpusCollector::op_entropy(&records);
    println!("\nStage 10 gate: op_entropy = {:.3} bits", entropy);
    if entropy > 2.0 {
        println!("✓ PASSED (> 2.0 bits)");
    } else {
        println!("✗ FAILED (<= 2.0 bits) — corpus needs more op diversity");
        std::process::exit(1);
    }
}
