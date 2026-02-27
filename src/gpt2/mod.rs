use anyhow::{Result, anyhow};
use std::process::{Command, Stdio};
use std::io::Read;

pub struct GPT2Proposer;

impl GPT2Proposer {
    pub fn new() -> Result<Self> {
        println!("  ✓ GPT-2 (HuggingFace) ready via Python bridge");
        Ok(GPT2Proposer)
    }

    pub fn generate_trace(&self, query: &str) -> Result<Vec<String>> {
        let mut child = Command::new("python3")
            .arg("scripts/gpt2_proposer.py")
            .arg(query)
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("failed to spawn python: {}", e))?;

        let mut output = String::new();
        child.stdout.take().unwrap().read_to_string(&mut output)?;

        let status = child.wait()?;
        if !status.success() {
            return Err(anyhow!("python proposer failed"));
        }

        let ops: Vec<String> = output
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

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
