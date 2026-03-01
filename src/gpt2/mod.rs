use anyhow::{Result, anyhow};
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};

use serde_json::Value;

pub struct GPT2Proposer {
    pub verbose: bool,
}

impl GPT2Proposer {
    pub fn new(verbose: bool) -> Result<Self> {
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
        
        if verbose {
            println!("  \u{1b}[32m✓\u{1b}[0m GPT-2 (HuggingFace) ready via Python bridge");
        }
        
        Ok(GPT2Proposer { verbose })
    }

    pub fn generate_trace(&self, query: &str) -> Result<Vec<String>> {
        if self.verbose {
            println!("📝 PROPOSER OPS:");
        }
        
        // Call the Python bridge
        let mut child = Command::new("python3")
            .arg("scripts/gpt2_proposer.py")
            .arg(query)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("Failed to spawn Python bridge: {}", e))?;
        
        // Read stdout line by line (the bridge outputs one JSON line)
        let stdout = child.stdout.take().expect("Failed to capture stdout");
        let reader = BufReader::new(stdout);
        let mut json_output = String::new();
        
        for line in reader.lines() {
            if let Ok(line) = line {
                json_output = line;
                break;
            }
        }
        
        // Wait for the child to finish
        let _status = child.wait()?;
        
        // Parse the JSON response
        let response: Value = serde_json::from_str(&json_output)
            .map_err(|e| anyhow!("Failed to parse Python bridge output: {}", e))?;
        
        // Check for errors
        let ok = response["ok"].as_bool().unwrap_or(false);
        if !ok {
            let error = response["error"].as_str().unwrap_or("unknown error");
            return Err(anyhow!("Python bridge error: {}", error));
        }
        
        // Extract the operations
        let ops = response["ops"].as_array()
            .ok_or_else(|| anyhow!("No ops array in response"))?
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect::<Vec<String>>();
        
        if ops.is_empty() {
            // This should not happen if ok=true, but handle gracefully
            return Err(anyhow!("Python bridge returned empty ops"));
        }
        
        if self.verbose {
            for op in &ops {
                println!("  {}", op);
            }
            
            // Also print metadata if available
            if let Some(meta) = response.get("meta") {
                if let Some(inference_s) = meta["inference_s"].as_f64() {
                    println!("  (inference: {:.3}s)", inference_s);
                }
                if let Some(tokens) = meta["tokens_generated"].as_u64() {
                    println!("  (tokens: {})", tokens);
                }
            }
        }
        
        Ok(ops)
    }
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
