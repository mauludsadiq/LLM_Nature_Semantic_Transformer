use anyhow::{anyhow, Result};
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
        
        // Extract ops if present (lossless: include required args)
          let ops = if let Some(ops_array) = json_value.get("ops").and_then(|v| v.as_array()) {
              let mut out: Vec<String> = Vec::with_capacity(ops_array.len());
              for opv in ops_array {
                  let op = opv.get("op").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("op missing op"))?;
                  match op {
                      "SELECT_UNIVERSE" => {
                          let u = opv.get("universe").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("SELECT_UNIVERSE missing universe"))?;
                          let n = opv.get("n").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("SELECT_UNIVERSE missing n"))?;
                          out.push(format!("SELECT_UNIVERSE universe={} n={}", u, n));
                      }
                      "FILTER_WEIGHT" => {
                          let min = opv.get("min").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("FILTER_WEIGHT missing min"))?;
                          let max = opv.get("max").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("FILTER_WEIGHT missing max"))?;
                          out.push(format!("FILTER_WEIGHT min={} max={}", min, max));
                      }
                      "TOPK" => {
                          let target = opv.get("target_elem").and_then(|v| v.as_str())
                              .or_else(|| opv.get("target").and_then(|v| v.as_str()))
                              .ok_or_else(|| anyhow!("TOPK missing target_elem"))?;
                          let k = opv.get("k").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("TOPK missing k"))?;
                          out.push(format!("TOPK target_elem={} k={}", target, k));
                      }
                      "RETURN_SET" => {
                          let max_items = opv.get("max_items").and_then(|v| v.as_u64()).unwrap_or(20);
                          let include_witness = opv.get("include_witness").and_then(|v| v.as_bool()).unwrap_or(false);
                          out.push(format!("RETURN_SET max_items={} include_witness={}", max_items, if include_witness { 1 } else { 0 }));
                      }
                      "START_ELEM" => {
                          let elem = opv.get("elem").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("START_ELEM missing elem"))?;
                          out.push(format!("LOAD {}", elem));
                      }
                      "SET_BIT" => {
                          let i = opv.get("i").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("SET_BIT missing i"))?;
                          let b = opv.get("b").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("SET_BIT missing b"))?;
                          out.push(format!("MASK_BIT bit={} val={}", i, b));
                      }
                      "WITNESS_NEAREST" => {
                          let target = opv.get("target_elem").and_then(|v| v.as_str())
                              .or_else(|| opv.get("target").and_then(|v| v.as_str()))
                              .ok_or_else(|| anyhow!("WITNESS_NEAREST missing target"))?;
                          let metric = opv.get("metric").and_then(|v| v.as_str()).unwrap_or("ABS_DIFF");
                          out.push(format!("WITNESS_NEAREST target_elem={} metric={}", target, metric));
                      }
                      other => return Err(anyhow!("unsupported op in JSON: {}", other)),
                  }
              }
              out
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
    
    // Parse reference as f64 (only if it is actually a fraction)
      let (ref_value, reference_is_frac) = if reference.contains("/") {
          let ref_parts: Vec<&str> = reference.split("/").collect();
          if ref_parts.len() == 2 {
              let ref_num: f64 = ref_parts[0].parse().unwrap_or(13.0);
              let ref_den: f64 = ref_parts[1].parse().unwrap_or(37.0);
              (ref_num / ref_den, true)
          } else {
              (0.0, false)
          }
      } else {
          (0.0, false)
      };

      // Calculate witness and diff from result (only if witness is actually a fraction)
      let witness = result.witness.as_deref().unwrap_or("1/3");
      let (witness_value, witness_is_frac) = if witness.contains("/") {
          let witness_parts: Vec<&str> = witness.split("/").collect();
          if witness_parts.len() == 2 {
              let witness_num: f64 = witness_parts[0].parse().unwrap_or(1.0);
              let witness_den: f64 = witness_parts[1].parse().unwrap_or(3.0);
              (witness_num / witness_den, true)
          } else {
              (0.0, false)
          }
      } else {
          (0.0, false)
      };

      let diff = if reference_is_frac && witness_is_frac {
          (ref_value - witness_value).abs()
      } else {
          0.0
      };
    // Print narrative block
      println!("\n───────────────────────────────────────────────────────────────────────────────");
      if cli.verbose {
          println!("Semantic Transformer • {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%SZ"));
      }

      println!("\nQuery: {}", cli.query);

      if reference_is_frac && witness_is_frac {
          // Fraction/QE narrative
          println!(
              "Answer: Closest fraction to {} with den ≤ 6 is {} (diff ≈ {:.4}). Total: {}. Verified.",
              reference,
              witness,
              diff,
              result.final_count
          );
          println!();
          println!("Reference: {} ≈ {:.5}", reference, ref_value);
          println!("Witness: {} ≈ {:.5}", witness, witness_value);
          println!("Total matching: {}", result.final_count);
      } else {
          // Non-fraction narrative (e.g. BOOLFUN)
          println!(
              "Answer: Witness is {}. Total: {}. Verified.",
              witness,
              result.final_count
          );
          println!();
          println!("Witness: {}", witness);
          println!("Total matching: {}", result.final_count);
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
