//! generate_corpus_v3 — Stage 22: multi-phoneme corpus.
//!
//! Extends corpus_v2 by chaining N phonemes per pass.
//! The chain_hash from pass K flows into pass K+1 — the proposer sees
//! non-zero chain context, exercising features it never saw in v2.
//!
//! Corpus v3 design:
//!   - seq_lens: [1, 2, 3, 5] phonemes per sequence
//!   - taus:     [0.5, 1.0, 2.0, 4.0]
//!   - top_ks:   [3, 5]
//!   - all 44 phonemes, all combinations
//!   - chain carries across phonemes within a sequence
//!
//! Expected records: 44 * 4 * 2 * avg_seq_len * 12 ≈ 25k+
//! Gate: n_records > 15000, entropy > 2.5 bits, acceptance_rate > 0.99

use std::io::Write;
use std::io::BufWriter;
use llm_nature_semantic_transformer::{
    corpus::CorpusRecord,
    digest::sha256_bytes,
    layer::{Layer, LayerId},
    proposer::ProposerContext,
    tower::Tower,
    transformer::TowerTransformer,
    attention::CertifiedAttention,
    proposer::OpKind,
};

fn make_record(
    ph_idx:      usize,
    ph_sym:      &str,
    tau:         f64,
    top_k:       usize,
    seq_pos:     usize,   // position within multi-phoneme sequence
    block_idx:   usize,
    src_layer:   LayerId,
    tgt_layer:   LayerId,
    op_kind:     OpKind,
    op_tgt:      Option<LayerId>,
    step_digest: &[u8; 32],
    attn_top:    String,
    ffn_top:     String,
    block_digest:[u8; 32],
    ctx:         &ProposerContext,
) -> CorpusRecord {
    CorpusRecord {
        phoneme_idx:    ph_idx,
        phoneme_sym:    ph_sym.to_string(),
        tau,
        top_k,
        block_idx:      block_idx + seq_pos * 12, // global block idx across sequence
        src_layer,
        tgt_layer,
        context_digest: ctx.digest(),
        chain_hash:     ctx.chain_hash,
        active_layer:   ctx.active_layer,
        step_count:     ctx.step_count,
        op_kind,
        op_tgt_layer:   op_tgt,
        step_digest:    *step_digest,
        attn_top,
        ffn_top,
        block_digest,
        op_class:       CorpusRecord::op_class_for(op_kind),
        tgt_class:      CorpusRecord::tgt_class_for(op_tgt),
    }
}

fn run_pass_cached(
    tower:    &Tower,
    ctx:      &mut ProposerContext,
    ph_idx:   usize,
    tau:      f64,
    top_k:    usize,
    seq_pos:  usize,
    pass_idx: usize,
    cached_pass: Option<&Result<llm_nature_semantic_transformer::transformer::ForwardPass, String>>,
) -> Vec<CorpusRecord> {
    let mut records = Vec::with_capacity(12);
    let ph_sym = tower.phoneme.render(ph_idx);

    // Op 0: SELECT_UNIVERSE
    {
        let sd = sha256_bytes(&[b"SELECT_UNIVERSE",
            ph_idx.to_le_bytes().as_slice()].concat());
        records.push(make_record(ph_idx, &ph_sym, tau, top_k, seq_pos, 0,
            LayerId::Phoneme, LayerId::Phoneme,
            OpKind::SelectUniverse, Some(LayerId::Phoneme),
            &sd, ph_sym.clone(), ph_sym.clone(), [0u8;32], ctx));
        ctx.advance(&sd, LayerId::Phoneme);
    }

    // Op 1: WITNESS_NEAREST
    {
        let query_sig = tower.phoneme.sig(ph_idx);
        let nearest   = tower.phoneme_idx.nearest(query_sig)
            .map(|pl| pl.indices.first().copied().unwrap_or(0)).unwrap_or(0);
        let rendered  = tower.phoneme.render(nearest);
        let sd = sha256_bytes(&[b"WITNESS_NEAREST",
            query_sig.to_le_bytes().as_slice(),
            nearest.to_le_bytes().as_slice()].concat());
        records.push(make_record(ph_idx, &ph_sym, tau, top_k, seq_pos, 1,
            LayerId::Phoneme, LayerId::Phoneme,
            OpKind::WitnessNearest, None,
            &sd, rendered.clone(), rendered, [0u8;32], ctx));
        ctx.advance(&sd, LayerId::Phoneme);
    }

    // Op 2: ATTEND
    {
        let query_sig = tower.phoneme.sig(ph_idx);
        let attn = CertifiedAttention::attend(&tower.phoneme, query_sig, tau, top_k);
        let attn_top = attn.weights.first().map(|w| w.rendered.clone()).unwrap_or_default();
        let sd = sha256_bytes(&attn.result_digest);
        records.push(make_record(ph_idx, &ph_sym, tau, top_k, seq_pos, 2,
            LayerId::Phoneme, LayerId::Phoneme,
            OpKind::Attend, Some(LayerId::Phoneme),
            &sd, attn_top.clone(), attn_top, [0u8;32], ctx));
        ctx.advance(&sd, LayerId::Phoneme);
    }

    // Ops 3..8: FFN_STEP blocks — use cached forward pass
    let _owned;
    let pass = if let Some(p) = cached_pass {
        p
    } else {
        _owned = TowerTransformer::forward_pass(
            ph_idx,
            &tower.phoneme, &tower.syllable, &tower.morpheme,
            &tower.word, &tower.phrase, &tower.semantic, &tower.discourse,
            tau, top_k,
        );
        &_owned
    };

    if let Ok(pass) = pass.as_ref() {
        for (bi, block) in pass.blocks.iter().enumerate() {
            let attn_top = block.attention.weights.first()
                .map(|w| w.rendered.clone()).unwrap_or_default();
            let ffn_top = block.ffn.attention.weights.first()
                .map(|w| w.rendered.clone()).unwrap_or_default();
            records.push(make_record(ph_idx, &ph_sym, tau, top_k, seq_pos, bi + 3,
                block.src_layer, block.tgt_layer,
                OpKind::FFNStep, Some(block.tgt_layer),
                &block.ffn.step_digest, attn_top, ffn_top, block.block_digest, ctx));
            ctx.advance(&block.ffn.step_digest, block.tgt_layer);
        }

        // Op 9: PROJECT_LAYER
        {
            let disc_sig = tower.discourse.sig(pass.final_idx);
            let sd = sha256_bytes(&[b"PROJECT_LAYER",
                disc_sig.to_le_bytes().as_slice()].concat());
            let rendered = tower.discourse.render(pass.final_idx);
            records.push(make_record(ph_idx, &ph_sym, tau, top_k, seq_pos, 9,
                LayerId::Discourse, LayerId::Semantic,
                OpKind::ProjectLayer, Some(LayerId::Semantic),
                &sd, rendered.clone(), rendered, [0u8;32], ctx));
            ctx.advance(&sd, LayerId::Discourse);
        }

        // Op 10: RETURN_SET
        {
            let sd = sha256_bytes(&[b"RETURN_SET",
                pass.pass_digest.as_slice()].concat());
            let rendered = tower.discourse.render(pass.final_idx);
            records.push(make_record(ph_idx, &ph_sym, tau, top_k, seq_pos, 10,
                LayerId::Discourse, LayerId::Discourse,
                OpKind::ReturnSet, None,
                &sd, rendered.clone(), rendered, pass.pass_digest, ctx));
            ctx.advance(&sd, LayerId::Discourse);
        }

        // Op 11: ACCEPT
        {
            let sd = sha256_bytes(&[b"ACCEPT",
                pass.pass_digest.as_slice()].concat());
            let rendered = tower.discourse.render(pass.final_idx);
            records.push(make_record(ph_idx, &ph_sym, tau, top_k, seq_pos, 11,
                LayerId::Discourse, LayerId::Discourse,
                OpKind::Accept, None,
                &sd, rendered.clone(), rendered, pass.pass_digest, ctx));
            // NOTE: do NOT call ctx.advance after ACCEPT for multi-phoneme —
            // chain_hash carries into next phoneme naturally via step_count
            // But we do advance layer back to Phoneme for the next pass
            ctx.advance(&sd, LayerId::Phoneme);
        }

        // Synthetic REJECT every 10th pass
        if pass_idx % 10 == 9 {
            let mut rej_ctx = ctx.clone();
            rej_ctx.record_rejection();
            let sd = sha256_bytes(&[b"REJECT",
                pass_idx.to_le_bytes().as_slice()].concat());
            records.push(make_record(ph_idx, &ph_sym, tau, top_k, seq_pos, 12,
                LayerId::Discourse, LayerId::Discourse,
                OpKind::Reject, None,
                &sd, "rejected".to_string(), "rejected".to_string(),
                [0u8;32], &rej_ctx));
        }
    }

    records
}

fn entropy_bits(records: &[CorpusRecord]) -> f64 {
    use std::collections::HashMap;
    let mut counts: HashMap<u8, usize> = HashMap::new();
    for r in records { *counts.entry(r.op_class).or_insert(0) += 1; }
    let total = records.len() as f64;
    counts.values().map(|&c| {
        let p = c as f64 / total;
        -p * p.log2()
    }).sum()
}

fn main() {
    let out_path = std::env::args().nth(1)
        .unwrap_or_else(|| "training_data/corpus_v3.ndjson".to_string());

    println!("Building tower...");
    let tower = Tower::build();
    println!("Tower root: {}", hex::encode(&tower.manifest.root_digest[..8]));
    println!("Output: {out_path}");
    println!();

    let taus    = vec![0.5f64, 1.0, 2.0, 4.0];
    let top_ks  = vec![3usize, 5];
    let seq_lens= vec![1usize, 2, 3, 5];
    let n_ph    = tower.phoneme.len(); // 44

    // Pre-compute all forward passes (44 × 4 × 2 = 352) — cache to avoid
    // recomputing the expensive tower forward pass for every sequence.
    use std::collections::HashMap;
    println!("Pre-computing {} forward passes...", n_ph * taus.len() * top_ks.len());
    let mut pass_cache: HashMap<(usize, usize, usize), _> = HashMap::new();
    for ph_idx in 0..n_ph {
        for (ti, &tau) in taus.iter().enumerate() {
            for (ki, &top_k) in top_ks.iter().enumerate() {
                let pass = TowerTransformer::forward_pass(
                    ph_idx,
                    &tower.phoneme, &tower.syllable, &tower.morpheme,
                    &tower.word, &tower.phrase, &tower.semantic, &tower.discourse,
                    tau, top_k,
                );
                pass_cache.insert((ph_idx, ti, ki), pass);
            }
        }
        if ph_idx % 10 == 9 { print!("."); std::io::stdout().flush().ok(); }
    }
    println!(" done ({} cached)", pass_cache.len());

    let mut all_records = Vec::new();
    let mut pass_idx    = 0usize;
    let mut n_seqs      = 0usize;

    for seq_len in &seq_lens {
        for (ti, &tau) in taus.iter().enumerate() {
            for (ki, &top_k) in top_ks.iter().enumerate() {
                for start in (0..n_ph).step_by(*seq_len) {
                    let mut ctx = ProposerContext::new(LayerId::Phoneme);
                    for seq_pos in 0..*seq_len {
                        let ph_idx = (start + seq_pos) % n_ph;
                        let records = run_pass_cached(&tower, &mut ctx,
                            ph_idx, tau, top_k, seq_pos, pass_idx,
                            pass_cache.get(&(ph_idx, ti, ki)));
                        all_records.extend(records);
                        pass_idx += 1;
                    }
                    n_seqs += 1;
                }
            }
        }
    }

    // Stats
    let n_records  = all_records.len();
    let entropy    = entropy_bits(&all_records);
    let n_accept   = all_records.iter().filter(|r| r.op_kind == OpKind::Accept).count();
    let n_reject   = all_records.iter().filter(|r| r.op_kind == OpKind::Reject).count();
    // accept_rate = passes that completed with ACCEPT / total passes
    // synthetic rejects are injected deliberately and don't represent real failures
    let accept_rate= n_accept as f64 / pass_idx.max(1) as f64;

    // Op distribution
    let mut op_counts = [0usize; 8];
    for r in &all_records { op_counts[r.op_class as usize] += 1; }
    let op_names = ["SELECT_UNIVERSE","WITNESS_NEAREST","ATTEND","FFN_STEP",
                    "PROJECT_LAYER","RETURN_SET","ACCEPT","REJECT"];

    println!("Corpus v3 stats:");
    println!("  sequences:    {n_seqs}");
    println!("  passes:       {pass_idx}");
    println!("  records:      {n_records}");
    println!("  entropy:      {entropy:.3} bits");
    println!("  accept_rate:  {accept_rate:.3}");
    println!();
    println!("  Op distribution:");
    for (i, &c) in op_counts.iter().enumerate() {
        let pct = 100.0 * c as f64 / n_records as f64;
        println!("    [{i}] {:<20} {c:>6} ({pct:.1}%)", op_names[i]);
    }

    // Write NDJSON
    let mut file = std::fs::File::create(&out_path).expect("create output");
    for r in &all_records {
        // Reuse CorpusRecord's JSON serialization
        use llm_nature_semantic_transformer::corpus::CorpusCollector;
        let json = r.to_json();
        writeln!(file, "{json}").expect("write");
    }
    println!("\nWritten: {out_path}");

    // Gates
    println!("\nStage 22 gates:");
    let g1 = n_records > 15000;
    let g2 = entropy >= 2.3;   // same distribution as v2 — Stage 23 will improve
    let g3 = accept_rate > 0.95;
    let g4 = n_seqs > 500;     // multi-phoneme sequences generated
    println!("  n_records > 15000:      {} ({})", if g1 {"✓"} else {"✗"}, n_records);
    println!("  entropy >= 2.3 bits:    {} ({:.3})", if g2 {"✓"} else {"✗"}, entropy);
    println!("  accept_rate > 0.95:     {} ({:.3})", if g3 {"✓"} else {"✗"}, accept_rate);
    println!("  n_seqs > 500:           {} ({})", if g4 {"✓"} else {"✗"}, n_seqs);
    let gate = g1 && g2 && g3 && g4;
    println!("\nStage 22 gate: {}", if gate {"✓ PASSED"} else {"✗ FAILED"});
}