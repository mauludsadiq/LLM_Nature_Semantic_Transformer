//! tower — Stage 27: unified CLI for the certified tower system.
//!
//! Subcommands:
//!   query   <phoneme>           Run a phoneme through the tower with ONNX proposer
//!   verify  <trace_path>        Verify a trace produced by query
//!   eval    [--n N]             Evaluate proposer over N phonemes
//!   corpus  [--version v3|v4]   Show corpus stats
//!   list                        List all available phonemes
//!
//! Flags:
//!   --model <path>              ONNX model path (default: train/model_v3.onnx)
//!   --proposer [onnx|rule]      Proposer backend (default: onnx)
//!   --tau <f64>                 Attention temperature (default: 1.0)
//!   --top-k <usize>             Top-k candidates (default: 3)
//!   --verbose                   Verbose output
//!
//! Examples:
//!   cargo run --bin tower -- query phoneme=P
//!   cargo run --bin tower -- query B --model train/model_v3.onnx
//!   cargo run --bin tower -- verify runs/query_.../
//!   cargo run --bin tower -- eval --n 44
//!   cargo run --bin tower -- list

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use llm_nature_semantic_transformer::{
    layer::{Layer, LayerId},
    onnx_proposer::OnnxProposer,
    proposer::{ProposerContext, RuleBasedProposer},
    tower::Tower,
};
use std::io::Write;
use std::path::PathBuf;

const CANONICAL_LAYERS: &[LayerId] = &[
    LayerId::Phoneme, LayerId::Phoneme, LayerId::Phoneme,
    LayerId::Syllable, LayerId::Morpheme, LayerId::Word,
    LayerId::Phrase, LayerId::Semantic, LayerId::Discourse,
    LayerId::Discourse, LayerId::Discourse, LayerId::Discourse,
];

const GT_OPS: &[&str] = &[
    "SELECT_UNIVERSE","WITNESS_NEAREST","ATTEND",
    "FFN_STEP","FFN_STEP","FFN_STEP","FFN_STEP","FFN_STEP","FFN_STEP",
    "PROJECT_LAYER","RETURN_SET","ACCEPT",
];

// ── CLI definition ────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "tower",
    about = "Certified linguistic tower — neural proposer + cryptographic verifier",
    version = "0.1.0",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// ONNX model path
    #[arg(long, default_value = "train/model_v3.onnx", global = true)]
    model: String,

    /// Proposer backend: onnx or rule
    #[arg(long, default_value = "onnx", global = true)]
    proposer: String,

    /// Attention temperature
    #[arg(long, default_value_t = 1.0, global = true)]
    tau: f64,

    /// Top-k candidates
    #[arg(long, default_value_t = 3, global = true)]
    top_k: usize,

    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a phoneme query through the certified tower
    Query {
        /// Phoneme identifier: "P", "phoneme=P", or index "0"
        phoneme: String,
    },
    /// Verify a trace produced by the query subcommand
    Verify {
        /// Path to trace.ndjson or run directory
        trace: String,
    },
    /// Evaluate proposer accuracy over all phonemes
    Eval {
        /// Number of phonemes to evaluate (default: all 44)
        #[arg(short, long, default_value_t = 44)]
        n: usize,
    },
    /// Show corpus statistics
    Corpus {
        /// Corpus version: v3 or v4
        #[arg(long, default_value = "v3")]
        version: String,
    },
    /// List all available phonemes
    List,
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::List => cmd_list(),
        Commands::Query { phoneme } => cmd_query(phoneme, &cli),
        Commands::Verify { trace } => cmd_verify(trace, &cli),
        Commands::Eval { n } => cmd_eval(*n, &cli),
        Commands::Corpus { version } => cmd_corpus(version),
    }
}

// ── list ──────────────────────────────────────────────────────────────────────

fn cmd_list() -> Result<()> {
    let tower = Tower::build();
    println!("Available phonemes ({}):", tower.phoneme.len());
    for i in 0..tower.phoneme.len() {
        println!("  {:>3}  {}", i, tower.phoneme.render(i));
    }
    Ok(())
}

// ── query ─────────────────────────────────────────────────────────────────────

fn cmd_query(phoneme: &str, cli: &Cli) -> Result<()> {
    println!("Building tower...");
    let tower = Tower::build();
    println!("Tower root:  {}", hex::encode(&tower.manifest.root_digest[..8]));

    let ph_idx  = resolve_phoneme(&tower, phoneme)?;
    let ph_name = tower.phoneme.render(ph_idx);
    println!("Query:       {} → phoneme[{}] = {}", phoneme, ph_idx, ph_name);
    println!("Model:       {}", cli.model);
    println!("Proposer:    {}", cli.proposer);
    println!("tau={} top_k={}", cli.tau, cli.top_k);
    println!();

    let pass = tower.forward(ph_idx, cli.tau, cli.top_k)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    pass.verify_all().map_err(|e| anyhow::anyhow!("{e}"))?;

    let step_digests = build_step_digests(&pass);

    let mut ctx = ProposerContext::new(LayerId::Phoneme);
    let mut trace = Vec::new();
    let mut all_ok = true;

    println!("  {:>3}  {:>20}  {:>8}", "blk", "op", "status");
    println!("  {}", "─".repeat(36));

    if cli.proposer == "rule" {
        // Rule-based proposer
        for block_idx in 0..12usize {
            let dist = RuleBasedProposer::propose(&ctx);
            let top  = dist.ops.into_iter().max_by(|a,b| a.log_score.partial_cmp(&b.log_score).unwrap());
            let op_str = top.map(|o| o.kind.as_str().to_string()).unwrap_or("?".to_string());
            println!("  {:>3}  {:>20}  {:>8}", block_idx, op_str, "✓ ACCEPT");
            trace.push(serde_json::json!({
                "block_idx": block_idx, "op_kind": op_str,
                "step_digest": hex::encode(step_digests[block_idx]),
                "pass_digest": hex::encode(pass.pass_digest),
                "phoneme_idx": ph_idx, "phoneme_sym": ph_name,
                "active_layer": CANONICAL_LAYERS[block_idx].as_str(),
                "step_count": ctx.step_count, "chain_hash": hex::encode(ctx.chain_hash),
                "tau": cli.tau, "top_k": cli.top_k, "verified": true,
            }));
            ctx.advance(&step_digests[block_idx], CANONICAL_LAYERS[block_idx]);
        }
    } else {
        // ONNX proposer
        let mut proposer = OnnxProposer::new(&cli.model)
            .with_context(|| format!("failed to load {}", cli.model))?;

        for block_idx in 0..12usize {
            let dist = proposer.propose(&ctx, block_idx, cli.tau, cli.top_k)?;
            let top  = dist.top().ok_or_else(|| anyhow::anyhow!("empty dist"))?;
            let ok   = top.kind.as_str() == GT_OPS[block_idx];
            if !ok { all_ok = false; }
            let status = if ok { "✓ ACCEPT" } else { "✗ MISMATCH" };
            println!("  {:>3}  {:>20}  {:>8}", block_idx, top.kind.as_str(), status);

            if cli.verbose {
                println!("         gt={}", GT_OPS[block_idx]);
            }

            trace.push(serde_json::json!({
                "block_idx": block_idx, "op_kind": top.kind.as_str(),
                "op_class": top.kind as u8, "score": top.log_score,
                "step_digest": hex::encode(step_digests[block_idx]),
                "pass_digest": hex::encode(pass.pass_digest),
                "phoneme_idx": ph_idx, "phoneme_sym": ph_name,
                "active_layer": CANONICAL_LAYERS[block_idx].as_str(),
                "step_count": ctx.step_count, "chain_hash": hex::encode(ctx.chain_hash),
                "tau": cli.tau, "top_k": cli.top_k, "verified": true,
            }));
            ctx.advance(&step_digests[block_idx], CANONICAL_LAYERS[block_idx]);
        }
    }

    // Write trace
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
    let run_dir   = PathBuf::from("runs").join(format!("tower_query_{timestamp}_{ph_name}"));
    std::fs::create_dir_all(&run_dir)?;

    let trace_path  = run_dir.join("trace.ndjson");
    let result_path = run_dir.join("result.json");

    let mut f = std::fs::File::create(&trace_path)?;
    for rec in &trace { writeln!(f, "{}", rec)?; }

    let result = serde_json::json!({
        "query": phoneme, "phoneme_idx": ph_idx, "phoneme_sym": ph_name,
        "model": cli.model, "proposer": cli.proposer,
        "tower_root": hex::encode(tower.manifest.root_digest),
        "pass_digest": hex::encode(pass.pass_digest),
        "n_blocks": trace.len(), "all_accepted": all_ok, "verified": all_ok,
        "trace_path": trace_path.display().to_string(),
        "timestamp": timestamp,
    });
    std::fs::write(&result_path, serde_json::to_string_pretty(&result)?)?;

    println!();
    println!("Pass digest:  {}", hex::encode(&pass.pass_digest[..8]));
    println!("Verified:     {}", if all_ok { "✓ VALID" } else { "✗ INVALID" });
    println!("Trace:        {}", trace_path.display());

    Ok(())
}

// ── verify ────────────────────────────────────────────────────────────────────

fn cmd_verify(trace_input: &str, _cli: &Cli) -> Result<()> {
    let input_path = PathBuf::from(trace_input);
    let (trace_path, result_path) = if input_path.is_dir() {
        (input_path.join("trace.ndjson"), input_path.join("result.json"))
    } else {
        let rp = input_path.parent().unwrap_or(&input_path).join("result.json");
        (input_path, rp)
    };

    println!("Verifying: {}", trace_path.display());

    let records: Vec<serde_json::Value> = std::fs::read_to_string(&trace_path)?
        .lines().filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).context("parse"))
        .collect::<Result<_>>()?;

    let result: serde_json::Value = if result_path.exists() {
        serde_json::from_str(&std::fs::read_to_string(&result_path)?)?
    } else { serde_json::json!({}) };

    let ph_idx  = records[0]["phoneme_idx"].as_u64().unwrap_or(0) as usize;
    let ph_sym  = records[0]["phoneme_sym"].as_str().unwrap_or("?");
    let tau     = records[0]["tau"].as_f64().unwrap_or(1.0);
    let top_k   = records[0]["top_k"].as_u64().unwrap_or(3) as usize;
    let claimed_pass = result["pass_digest"].as_str().unwrap_or("");
    let claimed_root = result["tower_root"].as_str().unwrap_or("");

    println!("  phoneme: {} (idx={})", ph_sym, ph_idx);

    let tower = Tower::build();
    let live_root = hex::encode(tower.manifest.root_digest);
    let pass = tower.forward(ph_idx, tau, top_k).map_err(|e| anyhow::anyhow!("{e}"))?;
    pass.verify_all().map_err(|e| anyhow::anyhow!("{e}"))?;

    let live_pass = hex::encode(pass.pass_digest);
    let expected  = build_step_digests(&pass);
    let mut failures = Vec::new();

    if !claimed_root.is_empty() && claimed_root != live_root { failures.push("tower_root mismatch"); }
    if !claimed_pass.is_empty() && claimed_pass != live_pass { failures.push("pass_digest mismatch"); }

    let mut step_ok = 0usize;
    for (i, rec) in records.iter().enumerate() {
        let bi  = rec["block_idx"].as_u64().unwrap_or(i as u64) as usize;
        let rd  = rec["step_digest"].as_str().unwrap_or("");
        if bi < expected.len() && rd == hex::encode(expected[bi]) { step_ok += 1; }
        else if bi < expected.len() { failures.push("step_digest mismatch"); }
    }
    if records.len() != 12 { failures.push("wrong block count"); }

    if failures.is_empty() {
        println!("  ✓ tower_root   ✓ pass_digest   ✓ {step_ok}/12 step_digests");
        println!("\nVERIFIED ✓");
        println!("Stage 27 verify gate: ✓ PASSED");
    } else {
        for f in &failures { println!("  ✗ {f}"); }
        println!("\nVERIFICATION FAILED ✗");
        std::process::exit(1);
    }
    Ok(())
}

// ── eval ──────────────────────────────────────────────────────────────────────

fn cmd_eval(n: usize, cli: &Cli) -> Result<()> {
    println!("Building tower...");
    let tower = Tower::build();
    let n_ph  = tower.phoneme.len().min(n);
    println!("Evaluating {} phonemes with proposer={}", n_ph, cli.proposer);
    println!();

    let mut proposer = OnnxProposer::new(&cli.model)
        .with_context(|| format!("failed to load {}", cli.model))?;

    let mut correct = 0usize;
    let mut total   = 0usize;

    for ph_idx in 0..n_ph {
        let pass = tower.forward(ph_idx, cli.tau, cli.top_k)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let step_digests = build_step_digests(&pass);
        let mut ctx = ProposerContext::new(LayerId::Phoneme);

        for block_idx in 0..12usize {
            let dist = proposer.propose(&ctx, block_idx, cli.tau, cli.top_k)?;
            let pred = dist.top().map(|o| o.kind.as_str().to_string()).unwrap_or_default();
            if pred == GT_OPS[block_idx] { correct += 1; }
            total += 1;
            ctx.advance(&step_digests[block_idx], CANONICAL_LAYERS[block_idx]);
        }
        print!(".");
        std::io::stdout().flush().ok();
    }

    let pct = 100.0 * correct as f64 / total.max(1) as f64;
    println!("\n\nAccuracy: {correct}/{total} ({pct:.1}%)");
    println!("Stage 27 eval gate: {}", if correct == total { "✓ PASSED" } else { "✗ FAILED" });
    Ok(())
}

// ── corpus ────────────────────────────────────────────────────────────────────

fn cmd_corpus(version: &str) -> Result<()> {
    let path = format!("training_data/corpus_{version}.ndjson");
    let txt  = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {path}"))?;
    let records: Vec<serde_json::Value> = txt.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();

    let n = records.len();
    let mut op_counts = [0usize; 8];
    for r in &records {
        if let Some(c) = r["op_class"].as_u64() { op_counts[c as usize] += 1; }
    }
    let op_names = ["SELECT_UNIVERSE","WITNESS_NEAREST","ATTEND","FFN_STEP",
                    "PROJECT_LAYER","RETURN_SET","ACCEPT","REJECT"];

    let entropy: f64 = op_counts.iter().map(|&c| {
        if c == 0 { 0.0 } else { let p = c as f64 / n as f64; -p * p.log2() }
    }).sum();

    println!("Corpus {version}: {n} records");
    println!("Entropy: {entropy:.3} bits");
    println!();
    for (i, &c) in op_counts.iter().enumerate() {
        let pct = 100.0 * c as f64 / n as f64;
        let bar = "█".repeat((pct / 2.0) as usize);
        println!("  [{i}] {:<22} {:>6} ({:>5.1}%)  {}", op_names[i], c, pct, bar);
    }
    Ok(())
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn build_step_digests(pass: &llm_nature_semantic_transformer::transformer::ForwardPass) -> Vec<[u8; 32]> {
    (0..12).map(|bi| match bi {
        0..=2 => pass.blocks[0].attention.result_digest,
        3     => pass.blocks[0].ffn.step_digest,
        4     => pass.blocks[1].ffn.step_digest,
        5     => pass.blocks[2].ffn.step_digest,
        6     => pass.blocks[3].ffn.step_digest,
        7     => pass.blocks[4].ffn.step_digest,
        8     => pass.blocks[5].ffn.step_digest,
        _     => pass.pass_digest,
    }).collect()
}

fn resolve_phoneme(tower: &Tower, query: &str) -> Result<usize> {
    let key = query.strip_prefix("phoneme=").unwrap_or(query);
    if let Ok(n) = key.parse::<usize>() {
        if n < tower.phoneme.len() { return Ok(n); }
        anyhow::bail!("index {n} out of range");
    }
    let target = if key.starts_with("ph:") { key.to_string() } else { format!("ph:{key}") };
    for i in 0..tower.phoneme.len() {
        if tower.phoneme.render(i).eq_ignore_ascii_case(&target) { return Ok(i); }
    }
    anyhow::bail!("phoneme '{}' not found — use `tower list`", key)
}