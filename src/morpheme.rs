//! Morpheme universe (Layer 3) — meaning-bearing units with allomorphy.
//!
//! Depends on: Layer 2 (syllable universe) — surface forms reference syllable IDs.
//!
//! Core principle: surface form is NOT identity. meaning_id IS identity.
//! Many surface forms can map to one morpheme (allomorphy).
//!
//! Signature bits (16-bit):
//!   bit0   is_root
//!   bit1   is_prefix
//!   bit2   is_suffix
//!   bit3   is_function
//!   bit4   is_inflectional
//!   bit5   is_derivational
//!   bit6   is_lexical
//!   bit7   role_plural
//!   bit8   role_past
//!   bit9   role_progressive
//!   bit10  role_negation
//!   bit11  multi_form_allomorph
//!   bit12  conditioned_variant
//!   bit13  pos_noun_affinity
//!   bit14  pos_verb_affinity
//!   bit15  pos_det_affinity

use std::cmp::Ordering;
use crate::digest::{sha256_bytes, merkle_root};

// ── Enumerations ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MorphemeType { Root, Prefix, Suffix, Function }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MorphClass { Derivational, Inflectional, Lexical }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GramRole {
    Plural, Past, Progressive, Comparative, Superlative,
    Negation, Agent, Nominalizer, None,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PosAffinity { Noun, Verb, Adj, Adv, Det, Prep, None }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Register { Neutral, Formal, Informal, Slang }

// ── Surface form ──────────────────────────────────────────────────────────────

/// A single surface realization of a morpheme.
/// `label` is human-readable evidence only — not part of identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SurfaceForm {
    pub form_id:   u8,
    pub label:     &'static str,
    pub condition: &'static str,  // e.g. "before_vowel_onset", "unconditional"
}

// ── Morpheme ──────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Morpheme {
    pub meaning_id:     &'static str,   // global stable identity
    pub morpheme_type:  MorphemeType,
    pub morph_class:    MorphClass,
    pub gram_role:      GramRole,
    pub pos_affinity:   PosAffinity,
    pub register:       Register,
    pub surface_forms:  Vec<SurfaceForm>,
    pub sig:            u16,
}

impl Morpheme {
    /// Canonical byte encoding: keys sorted lexicographically, no whitespace.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mtype = match self.morpheme_type {
            MorphemeType::Root     => "root",
            MorphemeType::Prefix   => "prefix",
            MorphemeType::Suffix   => "suffix",
            MorphemeType::Function => "function",
        };
        let mclass = match self.morph_class {
            MorphClass::Derivational => "derivational",
            MorphClass::Inflectional => "inflectional",
            MorphClass::Lexical      => "lexical",
        };
        let role = match self.gram_role {
            GramRole::Plural       => "plural",
            GramRole::Past         => "past",
            GramRole::Progressive  => "progressive",
            GramRole::Comparative  => "comparative",
            GramRole::Superlative  => "superlative",
            GramRole::Negation     => "negation",
            GramRole::Agent        => "agent",
            GramRole::Nominalizer  => "nominalizer",
            GramRole::None         => "none",
        };
        let pos = match self.pos_affinity {
            PosAffinity::Noun => "noun",
            PosAffinity::Verb => "verb",
            PosAffinity::Adj  => "adj",
            PosAffinity::Adv  => "adv",
            PosAffinity::Det  => "det",
            PosAffinity::Prep => "prep",
            PosAffinity::None => "none",
        };
        let reg = match self.register {
            Register::Neutral  => "neutral",
            Register::Formal   => "formal",
            Register::Informal => "informal",
            Register::Slang    => "slang",
        };
        // surface_forms sorted by form_id, serialized compactly
        let mut forms = self.surface_forms.clone();
        forms.sort_by_key(|f| f.form_id);
        let forms_str = forms.iter()
            .map(|f| format!(
                "{{\"condition\":\"{}\",\"form_id\":{},\"label\":\"{}\"}}",
                f.condition, f.form_id, f.label
            ))
            .collect::<Vec<_>>()
            .join(",");

        // Keys sorted lexicographically
        let s = format!(
            "{{\"gram_role\":\"{}\",\"meaning_id\":\"{}\",\"morph_class\":\"{}\",\
\"morpheme_type\":\"{}\",\"pos_affinity\":\"{}\",\"register\":\"{}\",\
\"surface_forms\":[{}]}}",
            role, self.meaning_id, mclass, mtype, pos, reg, forms_str
        );
        s.into_bytes()
    }

    pub fn digest(&self) -> [u8; 32] {
        sha256_bytes(&self.canonical_bytes())
    }
}

// ── Signature ─────────────────────────────────────────────────────────────────

pub fn sig16(m: &Morpheme) -> u16 {
    let mut s = 0u16;
    if m.morpheme_type == MorphemeType::Root     { s |= 1 << 0; }
    if m.morpheme_type == MorphemeType::Prefix   { s |= 1 << 1; }
    if m.morpheme_type == MorphemeType::Suffix   { s |= 1 << 2; }
    if m.morpheme_type == MorphemeType::Function { s |= 1 << 3; }
    if m.morph_class == MorphClass::Inflectional  { s |= 1 << 4; }
    if m.morph_class == MorphClass::Derivational  { s |= 1 << 5; }
    if m.morph_class == MorphClass::Lexical       { s |= 1 << 6; }
    if m.gram_role == GramRole::Plural            { s |= 1 << 7; }
    if m.gram_role == GramRole::Past              { s |= 1 << 8; }
    if m.gram_role == GramRole::Progressive       { s |= 1 << 9; }
    if m.gram_role == GramRole::Negation          { s |= 1 << 10; }
    if m.surface_forms.len() > 1                  { s |= 1 << 11; }
    if m.surface_forms.iter().any(|f| f.condition != "unconditional") { s |= 1 << 12; }
    if m.pos_affinity == PosAffinity::Noun        { s |= 1 << 13; }
    if m.pos_affinity == PosAffinity::Verb        { s |= 1 << 14; }
    if m.pos_affinity == PosAffinity::Det         { s |= 1 << 15; }
    s
}

pub fn bit_legend() -> [&'static str; 16] {
    ["is_root","is_prefix","is_suffix","is_function",
     "is_inflectional","is_derivational","is_lexical",
     "role_plural","role_past","role_progressive","role_negation",
     "multi_form_allomorph","conditioned_variant",
     "pos_noun_affinity","pos_verb_affinity","pos_det_affinity"]
}

// ── Validation ────────────────────────────────────────────────────────────────

#[derive(Debug, Eq, PartialEq)]
pub enum ValidationError {
    MissingMeaningId,
    NoSurfaceForms,
    DuplicateFormId,
    UnknownEnumValue,
    MultiformWithoutCondition,
}

pub fn validate(m: &Morpheme) -> Result<(), ValidationError> {
    if m.meaning_id.is_empty() {
        return Err(ValidationError::MissingMeaningId);
    }
    if m.surface_forms.is_empty() {
        return Err(ValidationError::NoSurfaceForms);
    }
    // form_ids must be unique
    let mut ids: Vec<u8> = m.surface_forms.iter().map(|f| f.form_id).collect();
    let before = ids.len();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() != before {
        return Err(ValidationError::DuplicateFormId);
    }
    // multi-form morphemes must have conditions on at least one form
    if m.surface_forms.len() > 1 {
        let all_unconditional = m.surface_forms.iter()
            .all(|f| f.condition == "unconditional");
        if all_unconditional {
            return Err(ValidationError::MultiformWithoutCondition);
        }
    }
    Ok(())
}

// ── Canonical order ───────────────────────────────────────────────────────────

pub fn canonical_cmp(a: &Morpheme, b: &Morpheme) -> Ordering {
    a.meaning_id.cmp(b.meaning_id)
}

pub fn sig_distance(a: &Morpheme, b: &Morpheme) -> u32 {
    ((a.sig ^ b.sig) as u32).count_ones()
}

// ── Inventory builder ─────────────────────────────────────────────────────────

/// Build the certified English morpheme inventory (v1).
/// Covers: core inflectional suffixes, productive derivational affixes,
/// grammatical function morphemes.
pub fn build_morpheme_universe() -> Vec<Morpheme> {
    let mut morphemes: Vec<Morpheme> = vec![

        // ── Inflectional suffixes ─────────────────────────────────────

        Morpheme {
            meaning_id: "en:morph:plural",
            morpheme_type: MorphemeType::Suffix,
            morph_class:   MorphClass::Inflectional,
            gram_role:     GramRole::Plural,
            pos_affinity:  PosAffinity::Noun,
            register:      Register::Neutral,
            surface_forms: vec![
                SurfaceForm { form_id: 1, label: "s",  condition: "after_voiceless_non_sibilant" },
                SurfaceForm { form_id: 2, label: "z",  condition: "after_voiced_non_sibilant" },
                SurfaceForm { form_id: 3, label: "iz", condition: "after_sibilant" },
            ],
            sig: 0,
        },

        Morpheme {
            meaning_id: "en:morph:past",
            morpheme_type: MorphemeType::Suffix,
            morph_class:   MorphClass::Inflectional,
            gram_role:     GramRole::Past,
            pos_affinity:  PosAffinity::Verb,
            register:      Register::Neutral,
            surface_forms: vec![
                SurfaceForm { form_id: 1, label: "t",  condition: "after_voiceless" },
                SurfaceForm { form_id: 2, label: "d",  condition: "after_voiced_non_dental" },
                SurfaceForm { form_id: 3, label: "id", condition: "after_dental" },
            ],
            sig: 0,
        },

        Morpheme {
            meaning_id: "en:morph:progressive",
            morpheme_type: MorphemeType::Suffix,
            morph_class:   MorphClass::Inflectional,
            gram_role:     GramRole::Progressive,
            pos_affinity:  PosAffinity::Verb,
            register:      Register::Neutral,
            surface_forms: vec![
                SurfaceForm { form_id: 1, label: "ing", condition: "unconditional" },
            ],
            sig: 0,
        },

        Morpheme {
            meaning_id: "en:morph:comparative",
            morpheme_type: MorphemeType::Suffix,
            morph_class:   MorphClass::Inflectional,
            gram_role:     GramRole::Comparative,
            pos_affinity:  PosAffinity::Adj,
            register:      Register::Neutral,
            surface_forms: vec![
                SurfaceForm { form_id: 1, label: "er", condition: "unconditional" },
            ],
            sig: 0,
        },

        Morpheme {
            meaning_id: "en:morph:superlative",
            morpheme_type: MorphemeType::Suffix,
            morph_class:   MorphClass::Inflectional,
            gram_role:     GramRole::Superlative,
            pos_affinity:  PosAffinity::Adj,
            register:      Register::Neutral,
            surface_forms: vec![
                SurfaceForm { form_id: 1, label: "est", condition: "unconditional" },
            ],
            sig: 0,
        },

        // ── Derivational affixes ──────────────────────────────────────

        Morpheme {
            meaning_id: "en:morph:negation_un",
            morpheme_type: MorphemeType::Prefix,
            morph_class:   MorphClass::Derivational,
            gram_role:     GramRole::Negation,
            pos_affinity:  PosAffinity::Adj,
            register:      Register::Neutral,
            surface_forms: vec![
                SurfaceForm { form_id: 1, label: "un", condition: "unconditional" },
            ],
            sig: 0,
        },

        Morpheme {
            meaning_id: "en:morph:negation_in",
            morpheme_type: MorphemeType::Prefix,
            morph_class:   MorphClass::Derivational,
            gram_role:     GramRole::Negation,
            pos_affinity:  PosAffinity::Adj,
            register:      Register::Neutral,
            surface_forms: vec![
                SurfaceForm { form_id: 1, label: "in",  condition: "before_non_labial_non_liquid" },
                SurfaceForm { form_id: 2, label: "im",  condition: "before_bilabial" },
                SurfaceForm { form_id: 3, label: "il",  condition: "before_liquid_l" },
                SurfaceForm { form_id: 4, label: "ir",  condition: "before_liquid_r" },
            ],
            sig: 0,
        },

        Morpheme {
            meaning_id: "en:morph:agent_er",
            morpheme_type: MorphemeType::Suffix,
            morph_class:   MorphClass::Derivational,
            gram_role:     GramRole::Agent,
            pos_affinity:  PosAffinity::Noun,
            register:      Register::Neutral,
            surface_forms: vec![
                SurfaceForm { form_id: 1, label: "er", condition: "unconditional" },
            ],
            sig: 0,
        },

        Morpheme {
            meaning_id: "en:morph:nominalizer_tion",
            morpheme_type: MorphemeType::Suffix,
            morph_class:   MorphClass::Derivational,
            gram_role:     GramRole::Nominalizer,
            pos_affinity:  PosAffinity::Noun,
            register:      Register::Formal,
            surface_forms: vec![
                SurfaceForm { form_id: 1, label: "tion",  condition: "after_consonant" },
                SurfaceForm { form_id: 2, label: "ation", condition: "after_vowel" },
            ],
            sig: 0,
        },

        Morpheme {
            meaning_id: "en:morph:nominalizer_ness",
            morpheme_type: MorphemeType::Suffix,
            morph_class:   MorphClass::Derivational,
            gram_role:     GramRole::Nominalizer,
            pos_affinity:  PosAffinity::Noun,
            register:      Register::Neutral,
            surface_forms: vec![
                SurfaceForm { form_id: 1, label: "ness", condition: "unconditional" },
            ],
            sig: 0,
        },

        Morpheme {
            meaning_id: "en:morph:re_prefix",
            morpheme_type: MorphemeType::Prefix,
            morph_class:   MorphClass::Derivational,
            gram_role:     GramRole::None,
            pos_affinity:  PosAffinity::Verb,
            register:      Register::Neutral,
            surface_forms: vec![
                SurfaceForm { form_id: 1, label: "re", condition: "unconditional" },
            ],
            sig: 0,
        },

        Morpheme {
            meaning_id: "en:morph:pre_prefix",
            morpheme_type: MorphemeType::Prefix,
            morph_class:   MorphClass::Derivational,
            gram_role:     GramRole::None,
            pos_affinity:  PosAffinity::None,
            register:      Register::Neutral,
            surface_forms: vec![
                SurfaceForm { form_id: 1, label: "pre", condition: "unconditional" },
            ],
            sig: 0,
        },

        Morpheme {
            meaning_id: "en:morph:ly_adverb",
            morpheme_type: MorphemeType::Suffix,
            morph_class:   MorphClass::Derivational,
            gram_role:     GramRole::None,
            pos_affinity:  PosAffinity::Adv,
            register:      Register::Neutral,
            surface_forms: vec![
                SurfaceForm { form_id: 1, label: "ly", condition: "unconditional" },
            ],
            sig: 0,
        },

        // ── Function morphemes ────────────────────────────────────────

        Morpheme {
            meaning_id: "en:morph:indef_article",
            morpheme_type: MorphemeType::Function,
            morph_class:   MorphClass::Lexical,
            gram_role:     GramRole::None,
            pos_affinity:  PosAffinity::Det,
            register:      Register::Neutral,
            surface_forms: vec![
                SurfaceForm { form_id: 1, label: "a",  condition: "before_consonant_onset" },
                SurfaceForm { form_id: 2, label: "an", condition: "before_vowel_onset" },
            ],
            sig: 0,
        },

        Morpheme {
            meaning_id: "en:morph:def_article",
            morpheme_type: MorphemeType::Function,
            morph_class:   MorphClass::Lexical,
            gram_role:     GramRole::None,
            pos_affinity:  PosAffinity::Det,
            register:      Register::Neutral,
            surface_forms: vec![
                SurfaceForm { form_id: 1, label: "the", condition: "unconditional" },
            ],
            sig: 0,
        },

        Morpheme {
            meaning_id: "en:morph:negation_not",
            morpheme_type: MorphemeType::Function,
            morph_class:   MorphClass::Lexical,
            gram_role:     GramRole::Negation,
            pos_affinity:  PosAffinity::None,
            register:      Register::Neutral,
            surface_forms: vec![
                SurfaceForm { form_id: 1, label: "not", condition: "unconditional" },
                SurfaceForm { form_id: 2, label: "n't", condition: "clitic_form" },
            ],
            sig: 0,
        },
    ];

    // Compute signatures
    for m in morphemes.iter_mut() {
        m.sig = sig16(m);
    }

    morphemes.sort_by(canonical_cmp);
    morphemes
}

// ── Universe digest ───────────────────────────────────────────────────────────

pub fn morpheme_universe_digest(morphemes: &[Morpheme]) -> [u8; 32] {
    let rules_digest  = sha256_bytes(b"morpheme_rules_en_v1:allomorphy=phonological_conditioned");
    let legend_digest = sha256_bytes(bit_legend().join(",").as_bytes());
    let mut leaves: Vec<[u8; 32]> = morphemes.iter().map(|m| m.digest()).collect();
    leaves.sort_unstable();
    let inventory_digest = merkle_root(&leaves);
    let mut cat = b"morpheme_universe_v1".to_vec();
    cat.extend_from_slice(&rules_digest);
    cat.extend_from_slice(&legend_digest);
    cat.extend_from_slice(&inventory_digest);
    sha256_bytes(&cat)
}

pub fn is_morpheme_universe(u: &str) -> bool {
    matches!(u.to_ascii_uppercase().as_str(), "MORPHEME" | "MORPH" | "MOR")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morpheme_universe_size_reasonable() {
        let u = build_morpheme_universe();
        assert!(u.len() >= 15, "expected at least 15 morphemes, got {}", u.len());
    }

    #[test]
    fn meaning_ids_unique() {
        let u = build_morpheme_universe();
        let mut ids: Vec<&str> = u.iter().map(|m| m.meaning_id).collect();
        let before = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), before, "meaning_ids must be unique");
    }

    #[test]
    fn all_morphemes_validate() {
        let u = build_morpheme_universe();
        for m in &u {
            assert!(validate(m).is_ok(), "morpheme {} failed validation", m.meaning_id);
        }
    }

    #[test]
    fn plural_is_allomorphic() {
        let u = build_morpheme_universe();
        let plural = u.iter().find(|m| m.meaning_id == "en:morph:plural").unwrap();
        assert_eq!(plural.surface_forms.len(), 3);
        assert!((plural.sig >> 11) & 1 == 1, "multi_form bit must be set");
        assert!((plural.sig >> 12) & 1 == 1, "conditioned_variant bit must be set");
    }

    #[test]
    fn progressive_is_single_form() {
        let u = build_morpheme_universe();
        let prog = u.iter().find(|m| m.meaning_id == "en:morph:progressive").unwrap();
        assert_eq!(prog.surface_forms.len(), 1);
        assert!((prog.sig >> 11) & 1 == 0, "multi_form bit must NOT be set");
    }

    #[test]
    fn sig16_plural_bits() {
        let u = build_morpheme_universe();
        let plural = u.iter().find(|m| m.meaning_id == "en:morph:plural").unwrap();
        assert!((plural.sig >> 2) & 1 == 1,  "is_suffix");
        assert!((plural.sig >> 4) & 1 == 1,  "is_inflectional");
        assert!((plural.sig >> 7) & 1 == 1,  "role_plural");
        assert!((plural.sig >> 13) & 1 == 1, "pos_noun_affinity");
    }

    #[test]
    fn canonical_bytes_keys_sorted() {
        let u = build_morpheme_universe();
        let m = &u[0];
        let s = String::from_utf8(m.canonical_bytes()).unwrap();
        let keys = ["gram_role","meaning_id","morph_class","morpheme_type",
                    "pos_affinity","register","surface_forms"];
        let mut last = 0usize;
        for k in &keys {
            let pos = s.find(k).expect("key must exist");
            assert!(pos >= last, "key {} out of order", k);
            last = pos;
        }
    }

    #[test]
    fn universe_sorted_by_meaning_id() {
        let u = build_morpheme_universe();
        for i in 1..u.len() {
            assert!(canonical_cmp(&u[i-1], &u[i]).is_lt(),
                "not sorted at index {}", i);
        }
    }

    #[test]
    fn universe_digest_deterministic() {
        let u = build_morpheme_universe();
        let d1 = morpheme_universe_digest(&u);
        let d2 = morpheme_universe_digest(&u);
        assert_eq!(d1, d2);
    }

    #[test]
    fn indef_article_conditioned() {
        let u = build_morpheme_universe();
        let art = u.iter().find(|m| m.meaning_id == "en:morph:indef_article").unwrap();
        assert_eq!(art.surface_forms.len(), 2);
        let labels: Vec<&str> = art.surface_forms.iter().map(|f| f.label).collect();
        assert!(labels.contains(&"a") && labels.contains(&"an"));
    }

    #[test]
    fn sig_distance_identical() {
        let u = build_morpheme_universe();
        assert_eq!(sig_distance(&u[0], &u[0]), 0);
    }
}

// ── Layer trait implementation ────────────────────────────────────────────────

use crate::layer::{Layer, LayerId};

pub struct MorphemeLayer {
    inventory: Vec<Morpheme>,
}

impl MorphemeLayer {
    pub fn new() -> Self {
        MorphemeLayer { inventory: build_morpheme_universe() }
    }
}

impl Default for MorphemeLayer {
    fn default() -> Self { Self::new() }
}

impl Layer for MorphemeLayer {
    fn id(&self) -> LayerId { LayerId::Morpheme }

    fn len(&self) -> usize { self.inventory.len() }

    fn canonical_bytes(&self, i: usize) -> Vec<u8> {
        self.inventory[i].canonical_bytes()
    }

    fn sig(&self, i: usize) -> u16 {
        self.inventory[i].sig
    }

    fn render(&self, i: usize) -> String {
        let m = &self.inventory[i];
        // morph:<meaning_id>/<primary_surface>
        // e.g. morph:plural/s  morph:en:word:cat/cat
        let primary = m.surface_forms.first()
            .map(|sf| sf.label)
            .unwrap_or("?");
        format!("morph:{}/{}", m.meaning_id, primary)
    }

    fn universe_digest(&self) -> [u8; 32] {
        morpheme_universe_digest(&self.inventory)
    }
}

// ── Additional tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod layer_tests {
    use super::*;
    use crate::layer::Layer;

    #[test]
    fn morpheme_layer_len() {
        let l = MorphemeLayer::new();
        assert_eq!(l.len(), 16);
    }

    #[test]
    fn morpheme_layer_render_nonempty() {
        let l = MorphemeLayer::new();
        for i in 0..l.len() {
            let r = l.render(i);
            assert!(r.starts_with("morph:"), "render {} = {}", i, r);
        }
    }

    #[test]
    fn morpheme_layer_render_has_slash() {
        let l = MorphemeLayer::new();
        for i in 0..l.len() {
            let r = l.render(i);
            assert!(r.contains('/'), "no slash in render {}: {}", i, r);
        }
    }

    #[test]
    fn morpheme_layer_digest_stable() {
        let l = MorphemeLayer::new();
        for i in 0..l.len() {
            assert_eq!(l.digest(i), l.digest(i));
        }
    }

    #[test]
    fn morpheme_layer_sig_matches_inventory() {
        let l = MorphemeLayer::new();
        let inv = build_morpheme_universe();
        for (i, m) in inv.iter().enumerate() {
            assert_eq!(l.sig(i), m.sig);
        }
    }

    #[test]
    fn morpheme_layer_nearest_self() {
        let l = MorphemeLayer::new();
        let s = l.sig(0);
        let n = l.nearest(s).unwrap();
        assert_eq!(l.sig_distance(l.sig(n), s), 0);
    }

    #[test]
    fn morpheme_layer_top_k() {
        let l = MorphemeLayer::new();
        let k = l.top_k(l.sig(0), 3);
        assert!(k.len() <= 3);
        if k.len() >= 2 {
            assert!(l.sig_distance(l.sig(k[0]), l.sig(0))
                 <= l.sig_distance(l.sig(k[1]), l.sig(0)));
        }
    }

    #[test]
    fn morpheme_layer_universe_digest_stable() {
        let l = MorphemeLayer::new();
        assert_eq!(l.universe_digest(), l.universe_digest());
    }

    #[test]
    fn morpheme_layer_witness_roundtrip() {
        let l = MorphemeLayer::new();
        let w = l.witness(0);
        assert_eq!(w.layer, crate::layer::LayerId::Morpheme);
        assert!(!w.rendered.is_empty());
        assert_eq!(w.digest, l.digest(0));
    }

    #[test]
    fn morpheme_layer_plural_rendered() {
        let l = MorphemeLayer::new();
        let has_plural = (0..l.len()).any(|i| l.render(i).contains("plural"));
        assert!(has_plural, "no plural morpheme rendered");
    }
}
