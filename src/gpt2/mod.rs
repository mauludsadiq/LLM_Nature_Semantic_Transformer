use anyhow::{Result, anyhow};
use std::process::{Command, Stdio};
pub struct GPT2Proposer;

impl GPT2Proposer {
    pub fn new() -> Result<Self> {
        println!("  ✓ GPT-2 (HuggingFace) ready via Python bridge");
        Ok(GPT2Proposer)
    }

    pub fn generate_trace(&self, query: &str) -> Result<Vec<String>> {
        let output = Command::new("python3")
            .arg("-W")
            .arg("ignore")
            .arg("gpt2_proposer.py")
            .arg(query)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| anyhow!("failed to run python: {}", e))?;

        if !output.status.success() {
            return Err(anyhow!(
                "python proposer failed (exit={}): {}",
                output.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        let output = String::from_utf8_lossy(&output.stdout).to_string();

        let s = output.trim();

        // Accept either JSON array (preferred) or newline-delimited ops.
        let ops: Vec<String> = if s.starts_with("[") {
            serde_json::from_str(s)
                .map_err(|e| anyhow!("invalid JSON trace from python: {} (stdout={})", e, s))?
        } else {
            s.lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect()
        };

        if ops.is_empty() {
            return Err(anyhow!("empty trace from GPT-2"));
        }

        Ok(ops)
    }
}

pub fn interpret_trace(trace: &[String]) -> Vec<String> {
    trace.iter().map(|op| {
        match op.as_str() {
            "LOAD 7/200" => "Step 0: LOAD 7/200".to_string(),
            "MASK_BIT bit=2 val=1" => "Step 1: MASK_BIT (den≤6 := true)".to_string(),
            "WITNESS_NEAREST target=7/200" => "Step 2: WITNESS_NEAREST(target=7/200, metric=ABS_DIFF)".to_string(),
            "RETURN_SET" => "Step 3: RETURN_SET".to_string(),
            _ => format!("Unknown op: {}", op),
        }
    }).collect()
}
