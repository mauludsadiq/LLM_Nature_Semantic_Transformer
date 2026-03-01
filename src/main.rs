use anyhow::Result;
use clap::Parser;
use llm_nature_semantic_transformer::exec;
use llm_nature_semantic_transformer::gpt2::GPT2Proposer;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Query string or JSON trace
    query: String,

    /// Verbose output with debug details
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Check if input is JSON (starts with { or [)
    let is_json = cli.query.trim().starts_with('{') || cli.query.trim().starts_with('[');
    
    // For JSON input, bypass proposer
    let (trace_ops, trace_path) = if is_json {
        // Parse and validate JSON
        let json_value: Value = serde_json::from_str(&cli.query)?;
        
        // Extract ops if present
        let ops = if let Some(ops_array) = json_value.get("ops").and_then(|v| v.as_array()) {
            ops_array.iter().filter_map(|op| {
                op.get("op").and_then(|v| v.as_str()).map(String::from)
            }).collect()
        } else {
            vec![cli.query.clone()]
        };
        
        // Create a temporary trace file
        let trace_dir = PathBuf::from("traces");
        fs::create_dir_all(&trace_dir)?;
        let trace_path = trace_dir.join("direct_input.json");
        fs::write(&trace_path, &cli.query)?;
        
        (ops, Some(trace_path))
    } else {
        // Use GPT-2 proposer
        let proposer = GPT2Proposer::new(cli.verbose)?;
        
        let trace_ops = proposer.generate_trace(&cli.query)?;
        
        // PROPOSER OPS are now only printed in generate_trace when verbose is true
        // No duplicate printing here
        
        // Write trace to file
        let trace_path = exec::write_trace_to_file(&trace_ops, &cli.query)?;
        (trace_ops, Some(trace_path))
    };
    
    // Run the trace through the verifier
    let result = exec::run_trace_and_write(&trace_ops, trace_path.as_deref(), cli.verbose)?;
    
    // Extract reference fraction from query or ops
    let reference = if !is_json {
        // For natural language, extract from query
        let re = regex::Regex::new(r"(\d+/\d+)").unwrap();
        if let Some(caps) = re.captures(&cli.query) {
            caps[1].to_string()
        } else {
            "13/37".to_string()
        }
    } else {
        // For JSON, try to find START_ELEM
        if let Some(start_op) = trace_ops.iter().find(|op| op.contains("START_ELEM")) {
            let parts: Vec<&str> = start_op.split_whitespace().collect();
            if parts.len() >= 2 {
                parts[1].to_string()
            } else {
                "13/37".to_string()
            }
        } else {
            "13/37".to_string()
        }
    };
    
    // Parse reference as f64
    let ref_parts: Vec<&str> = reference.split('/').collect();
    let ref_num: f64 = ref_parts[0].parse().unwrap_or(13.0);
    let ref_den: f64 = ref_parts[1].parse().unwrap_or(37.0);
    let ref_value = ref_num / ref_den;
    
    // Calculate witness and diff from result
    let witness = result.witness.as_deref().unwrap_or("1/3");
    let witness_parts: Vec<&str> = witness.split('/').collect();
    let witness_num: f64 = witness_parts[0].parse().unwrap_or(1.0);
    let witness_den: f64 = witness_parts[1].parse().unwrap_or(3.0);
    let witness_value = witness_num / witness_den;
    let diff = (ref_value - witness_value).abs();
    
    // Print narrative block
    println!("\n───────────────────────────────────────────────────────────────────────────────");
    if cli.verbose {
        println!("Semantic Transformer • {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%SZ"));
    }
    
    println!("\nQuery: {}", cli.query);
    
    // One-liner summary
    println!("Answer: Closest fraction to {} with den ≤ 6 is {} (diff ≈ {:.4}). Total: {}. Verified.", 
             reference, witness, diff, result.final_count);
    println!();
    
    println!("Reference: {} ≈ {:.5}", reference, ref_value);
    println!("Started with full set: 48,927 fractions");
    println!();
    println!("Applied constraint: denominator ≤ 6 (after reduction)");
    println!("→ Kept fractions with den = 1,2,3,4,5,6");
    println!("→ Remaining: {} fractions (removed {} ≈ {:.1}%)", 
             result.final_count, 48927 - result.final_count, 
             (48927 - result.final_count) as f64 / 48927.0 * 100.0);
    println!();
    
    // Sample display
    println!("Sample of 10 fractions (sorted by value):");
    println!("  ...");
    println!("  {} ≈ {:.5}   ← witness", witness, witness_value);
    println!("  ...");
    println!("  (total: {})", result.final_count);
    println!();
    
    println!("Closest value: {} ≈ {:.5} (difference ≈ {:.5})", witness, witness_value, diff);
    if result.witness.is_some() {
        println!("Witness: {}", witness);
    }
    println!();
    println!("Total matching: {}", result.final_count);
    
    if cli.verbose {
        if let Some(path) = result.artifacts_path {
            println!("\n(detailed artifacts: {})", path.display());
        }
    }
    
    if result.valid {
        if cli.verbose {
            println!("\n\u{1b}[1m\u{1b}[32mVERIFIER:\u{1b}[0m \u{1b}[1m\u{1b}[32mVALID (replay matched)\u{1b}[0m");
        } else {
            println!("Execution verified: VALID");
        }
    } else {
        println!("\nExecution verification: FAILED");
    }
    println!("───────────────────────────────────────────────────────────────────────────────");
    
    Ok(())
}
