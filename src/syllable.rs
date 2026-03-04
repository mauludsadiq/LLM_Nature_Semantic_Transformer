//! Syllable universe (Layer 2) — certified English syllables with 16-bit signature.
//!
//! Depends on: Layer 1 (phoneme universe).
//!
//! Structure: SYLLABLE = ONSET + NUCLEUS + CODA
//!   ONSET:   0-3 consonant phoneme IDs
//!   NUCLEUS: exactly 1 vowel phoneme ID
//!   CODA:    0-4 consonant phoneme IDs
//!
//! Generation strategy: enumerate only phonotactically plausible sequences
//! by pre-filtering on manner class before full validation.
//! This reduces the candidate space from ~340M to ~50K.

use std::cmp::Ordering;
use crate::phoneme::{Phoneme, Manner, Place, VowelHeight, VowelBackness, Rounded,
                     phoneme_universe_digest, build_phoneme_universe};
use crate::digest::{sha256_bytes, merkle_root};

// ── Syllable struct ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Syllable {
    pub id:      u32,
    pub onset:   Vec<u8>,
    pub nucleus: u8,
    pub coda:    Vec<u8>,
    pub sig:     u16,
}

impl Syllable {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let coda_str = format!("[{}]", self.coda.iter()
            .map(|x| x.to_string()).collect::<Vec<_>>().join(","));
        let onset_str = format!("[{}]", self.onset.iter()
            .map(|x| x.to_string()).collect::<Vec<_>>().join(","));
        let s = format!(
            "{{\"coda\":{},\"id\":{},\"nucleus\":{},\"onset\":{},\"stress\":\"none\"}}",
            coda_str, self.id, self.nucleus, onset_str
        );
        s.into_bytes()
    }

    pub fn digest(&self) -> [u8; 32] {
        sha256_bytes(&self.canonical_bytes())
    }
}

// ── Signature ────────────────────────────────────────────────────────────────

pub fn sig16(onset: &[u8], nucleus_ph: &Phoneme, coda: &[u8]) -> u16 {
    let ol = onset.len();
    let cl = coda.len();
    let mut s = 0u16;
    // onset length one-hot
    if ol == 0 { s |= 1 << 0; }
    if ol == 1 { s |= 1 << 1; }
    if ol == 2 { s |= 1 << 2; }
    if ol >= 3 { s |= 1 << 3; }
    // coda length one-hot
    if cl == 0 { s |= 1 << 4; }
    if cl == 1 { s |= 1 << 5; }
    if cl == 2 { s |= 1 << 6; }
    if cl >= 3 { s |= 1 << 7; }
    // nucleus features
    if nucleus_ph.backness == VowelBackness::Front   { s |= 1 << 8;  }
    if nucleus_ph.backness == VowelBackness::Back    { s |= 1 << 9;  }
    if nucleus_ph.height   == VowelHeight::High      { s |= 1 << 10; }
    if nucleus_ph.height   == VowelHeight::Low       { s |= 1 << 11; }
    if nucleus_ph.rounded  == Rounded::Rounded       { s |= 1 << 12; }
    // structural
    if cl > 0                                        { s |= 1 << 13; } // heavy
    if onset.first() == Some(&13)                    { s |= 1 << 14; } // S-cluster
    if cl >= 2                                       { s |= 1 << 15; } // coda cluster
    s
}

pub fn bit_legend() -> [&'static str; 16] {
    ["onset_empty","onset_len_1","onset_len_2","onset_len_3",
     "coda_empty","coda_len_1","coda_len_2","coda_len_ge_3",
     "nucleus_front","nucleus_back","nucleus_high","nucleus_low","nucleus_rounded",
     "weight_heavy","onset_s_cluster","coda_cluster"]
}

// ── Phonotactic validator ─────────────────────────────────────────────────────

#[derive(Debug, Eq, PartialEq)]
pub enum ValidationError {
    Structure,
    NucleusNotVowel,
    OnsetContainsVowel,
    CodaContainsVowel,
    OnsetTooLong,
    CodaTooLong,
    OnsetClusterIllegal,
    CodaClusterIllegal,
    SonorityViolation,
    LocalBan,
}

fn sonority(p: &Phoneme) -> u8 {
    match p.manner {
        Manner::Plosive     => 1,
        Manner::Affricate   => 2,
        Manner::Fricative   => 3,
        Manner::Nasal       => 4,
        Manner::Liquid      => 5,
        Manner::Approximant => 6,
        Manner::Vowel       => 7,
    }
}

fn is_vowel(p: &Phoneme) -> bool { p.manner == Manner::Vowel }

pub fn validate(
    onset: &[u8], nucleus: u8, coda: &[u8],
    ph: &[Phoneme],
) -> Result<(), ValidationError> {
    let ph_by_id = |id: u8| ph.iter().find(|p| p.id == id);

    if onset.len() > 3 { return Err(ValidationError::OnsetTooLong); }
    if coda.len()  > 3 { return Err(ValidationError::CodaTooLong); }
    let nuc = ph_by_id(nucleus).ok_or(ValidationError::Structure)?;
    if !is_vowel(nuc) { return Err(ValidationError::NucleusNotVowel); }

    for &id in onset {
        let p = ph_by_id(id).ok_or(ValidationError::Structure)?;
        if is_vowel(p) { return Err(ValidationError::OnsetContainsVowel); }
    }
    for &id in coda {
        let p = ph_by_id(id).ok_or(ValidationError::Structure)?;
        if is_vowel(p) { return Err(ValidationError::CodaContainsVowel); }
    }

    // Local bans
    if onset.contains(&20) { return Err(ValidationError::LocalBan); } // NG banned in onset
    for w in onset.windows(2) { if w[0]==w[1] { return Err(ValidationError::LocalBan); } }
    for w in coda.windows(2)  { if w[0]==w[1] { return Err(ValidationError::LocalBan); } }

    // Onset cluster legality
    if onset.len() == 2 {
        let o0 = ph_by_id(onset[0]).unwrap();
        let o1 = ph_by_id(onset[1]).unwrap();
        let ok =
            (o0.symbol == "S" && (o1.manner == Manner::Plosive
                               || o1.manner == Manner::Nasal
                               || (o1.manner == Manner::Fricative && o1.place != Place::Glottal)))
            || (o0.manner == Manner::Plosive
                && (o1.manner == Manner::Liquid || o1.manner == Manner::Approximant))
            || (o0.manner == Manner::Fricative
                && (o1.manner == Manner::Liquid || o1.manner == Manner::Approximant));
        if !ok { return Err(ValidationError::OnsetClusterIllegal); }
    }
    if onset.len() == 3 {
        let o0 = ph_by_id(onset[0]).unwrap();
        let o1 = ph_by_id(onset[1]).unwrap();
        let o2 = ph_by_id(onset[2]).unwrap();
        let ok = o0.symbol == "S"
            && o1.manner == Manner::Plosive
            && (o2.manner == Manner::Liquid || o2.manner == Manner::Approximant);
        if !ok { return Err(ValidationError::OnsetClusterIllegal); }
    }

    // Coda cluster legality
    if coda.len() == 2 {
        let c0 = ph_by_id(coda[0]).unwrap();
        let c1 = ph_by_id(coda[1]).unwrap();
        let ok =
            (c0.manner == Manner::Nasal
                && (c1.manner == Manner::Plosive || c1.manner == Manner::Affricate))
            || (c0.manner == Manner::Liquid && c1.manner == Manner::Fricative)
            || (c0.manner == Manner::Liquid && c1.manner == Manner::Plosive)
            || (c0.manner == Manner::Plosive && (c1.symbol == "S" || c1.symbol == "Z"))
            || (c0.manner == Manner::Fricative && c1.manner == Manner::Plosive);
        if !ok { return Err(ValidationError::CodaClusterIllegal); }
    }
    if coda.len() == 3 {
        let c0 = ph_by_id(coda[0]).unwrap();
        let c1 = ph_by_id(coda[1]).unwrap();
        let c2 = ph_by_id(coda[2]).unwrap();
        let ok = (c0.manner == Manner::Liquid || c0.manner == Manner::Nasal)
            && c1.manner == Manner::Plosive
            && (c2.symbol == "S" || c2.symbol == "Z");
        if !ok { return Err(ValidationError::CodaClusterIllegal); }
    }

    // Sonority sequencing
    // S is extrasyllabic in English — skip sonority check on S→C transition
    if onset.len() >= 2 {
        let start = if onset[0] == 13 { 1 } else { 0 }; // skip S if onset-initial
        let phones: Vec<&Phoneme> = onset[start..].iter().map(|&id| ph_by_id(id).unwrap()).collect();
        for w in phones.windows(2) {
            if sonority(w[0]) >= sonority(w[1]) {
                return Err(ValidationError::SonorityViolation);
            }
        }
    }
    if coda.len() >= 2 {
        let phones: Vec<&Phoneme> = coda.iter().map(|&id| ph_by_id(id).unwrap()).collect();
        for w in phones.windows(2) {
            if sonority(w[0]) <= sonority(w[1]) {
                return Err(ValidationError::SonorityViolation);
            }
        }
    }

    Ok(())
}

// ── Universe builder — bounded candidate generation ───────────────────────────
//
// Key insight: instead of enumerating all permutations of 27 consonants,
// we generate candidates by manner class only, then expand to actual phoneme IDs.
// This reduces candidates from ~340M to ~50K before validation.

/// Generate valid onset sequences (length 0-3) by manner-first filtering.
fn plausible_onsets(cons: &[&Phoneme]) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = vec![vec![]]; // empty onset

    // Length 1: any single consonant except NG
    for c in cons {
        if c.id != 20 { out.push(vec![c.id]); }
    }

    // Length 2: only linguistically plausible manner pairs
    // s + plosive/nasal/fricative(non-glottal), stop + liquid/approx, fric + liquid/approx
    for c0 in cons {
        for c1 in cons {
            if c0.id == c1.id { continue; }
            let ok =
                (c0.symbol == "S" && (c1.manner == Manner::Plosive
                    || c1.manner == Manner::Nasal
                    || (c1.manner == Manner::Fricative && c1.place != Place::Glottal)))
                || (c0.manner == Manner::Plosive
                    && (c1.manner == Manner::Liquid || c1.manner == Manner::Approximant))
                || (c0.manner == Manner::Fricative
                    && (c1.manner == Manner::Liquid || c1.manner == Manner::Approximant));
            if ok { out.push(vec![c0.id, c1.id]); }
        }
    }

    // Length 3: S + plosive + liquid/approx only
    for c0 in cons {
        if c0.symbol != "S" { continue; }
        for c1 in cons {
            if c1.manner != Manner::Plosive { continue; }
            for c2 in cons {
                if c2.manner != Manner::Liquid && c2.manner != Manner::Approximant { continue; }
                if c0.id == c1.id || c1.id == c2.id { continue; }
                out.push(vec![c0.id, c1.id, c2.id]);
            }
        }
    }
    out
}

/// Generate valid coda sequences (length 0-3) by manner-first filtering.
fn plausible_codas(cons: &[&Phoneme]) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = vec![vec![]]; // empty coda

    // Length 1: any single consonant
    for c in cons { out.push(vec![c.id]); }

    // Length 2: only plausible manner pairs
    for c0 in cons {
        for c1 in cons {
            if c0.id == c1.id { continue; }
            let ok =
                (c0.manner == Manner::Nasal
                    && (c1.manner == Manner::Plosive || c1.manner == Manner::Affricate))
                || (c0.manner == Manner::Liquid && c1.manner == Manner::Fricative)
                || (c0.manner == Manner::Liquid && c1.manner == Manner::Plosive)
                || (c0.manner == Manner::Plosive && (c1.symbol == "S" || c1.symbol == "Z"))
                || (c0.manner == Manner::Fricative && c1.manner == Manner::Plosive);
            if ok { out.push(vec![c0.id, c1.id]); }
        }
    }

    // Length 3: liquid/nasal + stop + S/Z
    for c0 in cons {
        if c0.manner != Manner::Liquid && c0.manner != Manner::Nasal { continue; }
        for c1 in cons {
            if c1.manner != Manner::Plosive { continue; }
            for c2 in cons {
                if c2.symbol != "S" && c2.symbol != "Z" { continue; }
                if c0.id == c1.id || c1.id == c2.id { continue; }
                out.push(vec![c0.id, c1.id, c2.id]);
            }
        }
    }
    out
}

pub fn build_syllable_universe() -> Vec<Syllable> {
    let phonemes = build_phoneme_universe();
    let vowels: Vec<&Phoneme> = phonemes.iter().filter(|p| p.manner == Manner::Vowel).collect();
    let cons: Vec<&Phoneme>   = phonemes.iter().filter(|p| p.manner != Manner::Vowel).collect();

    let onsets = plausible_onsets(&cons);
    let codas  = plausible_codas(&cons);

    let mut syllables: Vec<Syllable> = Vec::new();

    for nuc_ph in &vowels {
        for onset in &onsets {
            for coda in &codas {
                if validate(onset, nuc_ph.id, coda, &phonemes).is_ok() {
                    let sig = sig16(onset, nuc_ph, coda);
                    syllables.push(Syllable {
                        id: 0, // assigned after sort
                        onset: onset.clone(),
                        nucleus: nuc_ph.id,
                        coda: coda.clone(),
                        sig,
                    });
                }
            }
        }
    }

    syllables.sort_by(canonical_cmp);
    syllables.dedup_by(|a, b| {
        a.onset == b.onset && a.nucleus == b.nucleus && a.coda == b.coda
    });
    for (i, s) in syllables.iter_mut().enumerate() {
        s.id = (i + 1) as u32;
    }
    syllables
}

// ── Canonical order ───────────────────────────────────────────────────────────

pub fn canonical_cmp(a: &Syllable, b: &Syllable) -> Ordering {
    a.onset.cmp(&b.onset)
        .then(a.nucleus.cmp(&b.nucleus))
        .then(a.coda.cmp(&b.coda))
}

pub fn sig_distance(a: &Syllable, b: &Syllable) -> u32 {
    ((a.sig ^ b.sig) as u32).count_ones()
}

pub fn is_syllable_universe(u: &str) -> bool {
    matches!(u.to_ascii_uppercase().as_str(), "SYLLABLE" | "SYL" | "SY")
}

// ── Universe digest ───────────────────────────────────────────────────────────

pub fn syllable_universe_digest(syllables: &[Syllable]) -> [u8; 32] {
    let phonemes = build_phoneme_universe();
    let ph_digest = phoneme_universe_digest(&phonemes);
    let rules_digest = sha256_bytes(
        b"syllable_rules_en_v1:max_onset=3:max_coda=3:sonority=rise_fall");
    let legend_digest = sha256_bytes(bit_legend().join(",").as_bytes());
    let mut leaves: Vec<[u8; 32]> = syllables.iter().map(|s| s.digest()).collect();
    leaves.sort_unstable();
    let inventory_digest = merkle_root(&leaves);
    let mut cat = b"syllable_universe_v1".to_vec();
    cat.extend_from_slice(&ph_digest);
    cat.extend_from_slice(&rules_digest);
    cat.extend_from_slice(&legend_digest);
    cat.extend_from_slice(&inventory_digest);
    sha256_bytes(&cat)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ph() -> Vec<Phoneme> { build_phoneme_universe() }

    #[test]
    fn valid_cat_syllable() {
        assert!(validate(&[5], 28, &[3], &ph()).is_ok()); // K+AE+T
    }

    #[test]
    fn valid_open_syllable() {
        assert!(validate(&[], 25, &[], &ph()).is_ok()); // IY alone
    }

    #[test]
    fn valid_str_cluster() {
        assert!(validate(&[13, 3, 22], 28, &[], &ph()).is_ok()); // STR+AE
    }

    #[test]
    fn invalid_ng_in_onset() {
        assert_eq!(validate(&[20], 25, &[], &ph()), Err(ValidationError::LocalBan));
    }

    #[test]
    fn invalid_nucleus_consonant() {
        assert_eq!(validate(&[], 1, &[], &ph()), Err(ValidationError::NucleusNotVowel));
    }

    #[test]
    fn invalid_onset_too_long() {
        assert_eq!(validate(&[1,2,3,4], 25, &[], &ph()), Err(ValidationError::OnsetTooLong));
    }

    #[test]
    fn invalid_onset_cluster() {
        assert_eq!(validate(&[2, 4], 25, &[], &ph()), Err(ValidationError::OnsetClusterIllegal));
    }

    #[test]
    fn sig16_open_front_high() {
        let phonemes = ph();
        let iy = phonemes.iter().find(|p| p.symbol == "IY").unwrap();
        let s = sig16(&[], iy, &[]);
        assert_eq!(s & 1, 1,         "onset_empty");
        assert_eq!((s >> 4) & 1, 1,  "coda_empty");
        assert_eq!((s >> 8) & 1, 1,  "nucleus_front");
        assert_eq!((s >> 10) & 1, 1, "nucleus_high");
        assert_eq!((s >> 13) & 1, 0, "not heavy");
    }

    #[test]
    fn universe_builds_nonempty_and_fast() {
        let u = build_syllable_universe();
        assert!(u.len() > 500,  "expected >500 syllables, got {}", u.len());
        assert!(u.len() < 500_000, "unexpectedly large: {}", u.len());
    }

    #[test]
    fn universe_sorted_and_unique() {
        let u = build_syllable_universe();
        for i in 1..u.len() {
            assert!(canonical_cmp(&u[i-1], &u[i]).is_lt(),
                "not strictly sorted at index {}", i);
        }
    }

    #[test]
    fn universe_digest_deterministic() {
        let u = build_syllable_universe();
        let d1 = syllable_universe_digest(&u);
        let d2 = syllable_universe_digest(&u);
        assert_eq!(d1, d2);
    }
}

// ── Layer trait implementation ────────────────────────────────────────────────

use crate::layer::{Layer, LayerId};

pub struct SyllableLayer {
    inventory: Vec<Syllable>,
}

impl SyllableLayer {
    pub fn new() -> Self {
        SyllableLayer { inventory: build_syllable_universe() }
    }
}

impl Default for SyllableLayer {
    fn default() -> Self { Self::new() }
}

impl Layer for SyllableLayer {
    fn id(&self) -> LayerId { LayerId::Syllable }

    fn len(&self) -> usize { self.inventory.len() }

    fn canonical_bytes(&self, i: usize) -> Vec<u8> {
        self.inventory[i].canonical_bytes()
    }

    fn sig(&self, i: usize) -> u16 {
        self.inventory[i].sig
    }

    fn render(&self, i: usize) -> String {
        // syl:<onset_ids>-<nucleus_id>-<coda_ids>
        // e.g. syl:K-AE-T uses phoneme IDs directly
        let s = &self.inventory[i];
        let phonemes = crate::phoneme::build_phoneme_universe();
        let ph_sym = |id: u8| -> String {
            phonemes.iter().find(|p| p.id == id).map(|p| p.symbol.to_string()).unwrap_or_else(|| id.to_string())
        };
        let onset: Vec<String> = s.onset.iter().map(|&id| ph_sym(id)).collect();
        let coda: Vec<String>  = s.coda.iter().map(|&id| ph_sym(id)).collect();
        format!("syl:{}-{}-{}", onset.join(""), ph_sym(s.nucleus), coda.join(""))
    }

    fn universe_digest(&self) -> [u8; 32] {
        let mut leaves: Vec<[u8; 32]> = self.inventory.iter()
            .map(|s| crate::digest::sha256_bytes(&s.canonical_bytes()))
            .collect();
        leaves.sort_unstable();
        crate::digest::merkle_root(&leaves)
    }
}

// ── Additional tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod layer_tests {
    use super::*;
    use crate::layer::Layer;

    #[test]
    fn syllable_layer_len() {
        let l = SyllableLayer::new();
        assert!(l.len() > 100_000, "expected >100k syllables, got {}", l.len());
    }

    #[test]
    fn syllable_layer_render_nonempty() {
        let l = SyllableLayer::new();
        for i in 0..l.len().min(50) {
            let r = l.render(i);
            assert!(r.starts_with("syl:"), "render {} = {}", i, r);
        }
    }

    #[test]
    fn syllable_layer_render_has_two_dashes() {
        let l = SyllableLayer::new();
        for i in 0..l.len().min(50) {
            let r = l.render(i);
            assert_eq!(r.matches('-').count(), 2,
                "expected 2 dashes in: {}", r);
        }
    }

    #[test]
    fn syllable_layer_digest_stable() {
        let l = SyllableLayer::new();
        for i in 0..l.len().min(20) {
            assert_eq!(l.digest(i), l.digest(i));
        }
    }

    #[test]
    fn syllable_layer_nearest_self() {
        let l = SyllableLayer::new();
        let s = l.sig(0);
        let n = l.nearest(s).unwrap();
        assert_eq!(l.sig_distance(l.sig(n), s), 0);
    }

    #[test]
    fn syllable_layer_top_k() {
        let l = SyllableLayer::new();
        let k = l.top_k(l.sig(0), 5);
        assert!(k.len() <= 5);
        if k.len() >= 2 {
            assert!(l.sig_distance(l.sig(k[0]), l.sig(0))
                 <= l.sig_distance(l.sig(k[1]), l.sig(0)));
        }
    }

    #[test]
    fn syllable_layer_universe_digest_stable() {
        let l = SyllableLayer::new();
        assert_eq!(l.universe_digest(), l.universe_digest());
    }

    #[test]
    fn syllable_layer_witness_roundtrip() {
        let l = SyllableLayer::new();
        let w = l.witness(0);
        assert_eq!(w.layer, crate::layer::LayerId::Syllable);
        assert!(!w.rendered.is_empty());
        assert_eq!(w.digest, l.digest(0));
    }
}
