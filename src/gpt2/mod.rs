use anyhow::{anyhow, Result};
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader, Read};
use std::time::Instant;

pub struct GPT2Proposer;

impl GPT2Proposer {
    pub fn new() -> Result<Self> {
        // Test the Python bridge
        let output = Command::new("python3")
            .arg("scripts/gpt2_proposer.py")
            .arg("--test")
            .output()
            .map_err(|e| anyhow!("Failed to run Python bridge: {}", e))?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("Python bridge test failed: {}", stderr));
        }
        
        println!("  \u{1b}[32m✓\u{1b}[0m GPT-2 (HuggingFace) ready via Python bridge");
        Ok(GPT2Proposer)
    }

    pub fn generate_trace(&self, query: &str) -> Result<Vec<String>> {
        let start = Instant::now();
        
        // Call Python script with the query
        let mut child = Command::new("python3")
            .arg("scripts/gpt2_proposer.py")
            .arg(query)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("Failed to spawn Python process: {}", e))?;

        // Read stdout line by line to capture progress
        let stdout = child.stdout.take().unwrap();
        let reader = BufReader::new(stdout);
        let mut output_lines = Vec::new();
        
        for line in reader.lines() {
            let line = line?;
            if line.starts_with("PROGRESS:") {
                println!("  {}", &line[9..]); // Print progress without PROGRESS: prefix
            } else {
                output_lines.push(line);
            }
        }

        // Wait for the process to finish and capture stderr
        let status = child.wait()?;
        let mut stderr = String::new();
        if let Some(stderr_pipe) = child.stderr {
            BufReader::new(stderr_pipe).read_to_string(&mut stderr)?;
        }

        if !status.success() {
            return Err(anyhow!("Python process failed: {}", stderr));
        }

        // Join output lines and parse JSON
        let output = output_lines.join("\n");
        
        // Try to parse as JSON
        let json: serde_json::Value = match serde_json::from_str(&output) {
            Ok(json) => json,
            Err(_e) => {
                // If JSON parsing fails, treat as raw output (back-compat)
                let wall_ms = start.elapsed().as_millis();
                println!("  PROMPT: (unknown; python did not emit JSON object)");
                
                // Truncate raw output for display
                let mut raw_sum = output.replace("\n", " ");
                if raw_sum.len() > 100 {
                    raw_sum.truncate(100);
                    raw_sum.push_str("... (truncated)");
                }
                println!("  Raw output: {}", if raw_sum.is_empty() { "(empty)".to_string() } else { raw_sum });
                println!("  Inference time: {}ms (bridge)", wall_ms);
                
                // Parse as simple ops
                let ops: Vec<String> = if output.starts_with("[") {
                    serde_json::from_str(&output)
                        .map_err(|e| anyhow!("invalid JSON trace from python: {} (stdout={})", e, output))?
                } else {
                    output.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()
                };
                
                println!("  Parsed: {} operations extracted", ops.len());
                
                if ops.is_empty() {
                    println!("  ⚠️ GPT-2 output was empty or invalid");
                    println!("  ⚠️ Using fallback trace");
                    
                    // Extract fraction from query for fallback
                    let fraction = extract_fraction_from_query(query).unwrap_or_else(|| "7/200".to_string());
                    
                    return Ok(vec![
                        format!("LOAD {}", fraction),
                        "MASK_BIT bit=2 val=1".to_string(),
                        format!("WITNESS_NEAREST target={}", fraction),
                        "RETURN_SET".to_string(),
                    ]);
                }
                
                return Ok(ops);
            }
        };

        // Extract fields from JSON
        let ok = json["ok"].as_bool().unwrap_or(false);
        if !ok {
            let error = json["error"].as_str().unwrap_or("unknown error");
            return Err(anyhow!("Python bridge error: {}", error));
        }

        let raw_text = json["raw_text"].as_str().unwrap_or("");
        let ops: Vec<String> = json["ops"]
            .as_array()
            .ok_or_else(|| anyhow!("missing ops array in JSON"))?
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();

        let _fallback_used = json["fallback_used"].as_bool().unwrap_or(false);
        let meta = &json["meta"];
        let _device = meta["device"].as_str().unwrap_or("unknown");
        let inference_s = meta["inference_s"].as_f64().unwrap_or(0.0);
        let tokens = meta["tokens_generated"].as_u64().unwrap_or(0);
        
        let wall_ms = start.elapsed().as_millis();

        // Display info
        if let Some(prompt) = json["prompt"].as_str() {
            let prompt_preview: String = prompt.chars().take(100).collect();
            println!("  PROMPT: {}...", prompt_preview.replace("\n", "\\n"));
        }
        
        // Truncate raw output for display
        let mut raw_sum = raw_text.replace("\n", " ");
        if raw_sum.len() > 100 {
            raw_sum.truncate(100);
            raw_sum.push_str("... (truncated)");
        }
        println!("  Raw output: {}", if raw_sum.is_empty() { "(empty)".to_string() } else { raw_sum });
        println!("  Inference time: {}s (python) | wall: {}ms (bridge)", inference_s, wall_ms);
        println!("  Tokens generated: {}", tokens);
        println!("  Parsed: {} operations extracted", ops.len());

        if _fallback_used {
            println!("  ⚠️ Fallback trace used (generation failed or invalid)");
        }

        if ops.is_empty() {
            println!("  ⚠️ GPT-2 output was empty or invalid");
            println!("  ⚠️ Using fallback trace");
            
            // Extract fraction from query for fallback
            let fraction = extract_fraction_from_query(query).unwrap_or_else(|| "7/200".to_string());
            
            return Ok(vec![
                format!("LOAD {}", fraction),
                "MASK_BIT bit=2 val=1".to_string(),
                format!("WITNESS_NEAREST target={}", fraction),
                "RETURN_SET".to_string(),
            ]);
        }

        Ok(ops)
    }
}

/// Extract fraction from query string like "Find fractions similar to 13/37 but with denominator ≤ 6"
fn extract_fraction_from_query(query: &str) -> Option<String> {
    // Try to find pattern like "similar to X/Y" or just a standalone fraction
    let re = regex::Regex::new(r"similar to (\d+/\d+)").ok()?;
    if let Some(caps) = re.captures(query) {
        return Some(caps[1].to_string());
    }
    
    // Fallback: look for any fraction pattern
    let re = regex::Regex::new(r"(\d+/\d+)").ok()?;
    if let Some(caps) = re.captures(query) {
        return Some(caps[1].to_string());
    }
    
    None
}

#[allow(dead_code)]

pub fn interpret_trace(ops: &[String]) -> Vec<String> {
    let mut human_readable = Vec::new();
    
    for (i, op) in ops.iter().enumerate() {
        match op.as_str() {
            op if op.starts_with("LOAD ") => {
                let fraction = op.strip_prefix("LOAD ").unwrap_or("");
                human_readable.push(format!("Step {}: LOAD {}", i, fraction));
            }
            "MASK_BIT bit=2 val=1" => {
                human_readable.push(format!("Step {}: MASK_BIT (den≤6 := true)", i));
            }
            op if op.starts_with("WITNESS_NEAREST target=") => {
                let fraction = op.strip_prefix("WITNESS_NEAREST target=").unwrap_or("");
                human_readable.push(format!("Step {}: WITNESS_NEAREST(target={}, metric=ABS_DIFF)", i, fraction));
            }
            "RETURN_SET" => {
                human_readable.push(format!("Step {}: RETURN_SET", i));
            }
            _ => {
                human_readable.push(format!("Step {}: {}", i, op));
            }
        }
    }
    
    human_readable
}
