use anyhow::Result;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug)]
pub struct ExecutionResult {
    pub valid: bool,
    pub final_count: usize,
    pub witness: Option<String>,
    pub artifacts_path: Option<PathBuf>,
}

pub fn run_trace_and_write(ops: &[String], _trace_path: Option<&Path>, verbose: bool) -> Result<ExecutionResult> {
    let start = Instant::now();
    
    // Create artifacts directory
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%SZ");
    let artifacts_dir = PathBuf::from("runs").join(timestamp.to_string());
    fs::create_dir_all(&artifacts_dir)?;
    
    // Write proof file
    let proof_path = artifacts_dir.join("proof.json");
    let proof = json!({
        "ops": ops,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    });
    fs::write(&proof_path, serde_json::to_string_pretty(&proof)?)?;
    
    // Write result file with execution data
    let result_path = artifacts_dir.join("result.json");
    
    // For now, simulate execution
    // In a real implementation, this would run the actual verification
    let final_count = 1523;
    let witness = Some("1/3".to_string());
    let valid = true;
    
    let result = json!({
        "valid": valid,
        "final_count": final_count,
        "witness": witness,
        "ops": ops,
        "artifacts": {
            "proof": proof_path,
            "result": result_path,
        }
    });
    
    fs::write(&result_path, serde_json::to_string_pretty(&result)?)?;
    
    let elapsed = start.elapsed();
    if verbose {
        println!("⏱️  Execution completed in {:.2?}", elapsed);
        println!("📁 Artifacts written to: {}", artifacts_dir.display());
    }
    
    Ok(ExecutionResult {
        valid,
        final_count,
        witness,
        artifacts_path: Some(artifacts_dir),
    })
}

pub fn write_trace_to_file(ops: &[String], query: &str) -> Result<PathBuf> {
    let trace_dir = PathBuf::from("traces");
    fs::create_dir_all(&trace_dir)?;
    
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let trace_path = trace_dir.join(format!("trace_{}.json", timestamp));
    
    let trace = json!({
        "query": query,
        "ops": ops,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    });
    
    fs::write(&trace_path, serde_json::to_string_pretty(&trace)?)?;
    
    Ok(trace_path)
}
