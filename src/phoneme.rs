//! Phoneme universe (Layer 1) — 44 English phonemes with 12-bit structural signature.
//!
//! Sources: IPA / ARPAbet feature assignments (static, no external dependency).
//!
//! Signature bits:
//!   bit0   voiced
//!   bit1   plosive
//!   bit2   nasal
//!   bit3   fricative
//!   bit4   affricate
//!   bit5   approximant
//!   bit6   vowel
//!   bit7   bilabial
//!   bit8   alveolar
//!   bit9   velar
//!   bit10  front_vowel
//!   bit11  rounded

use std::cmp::Ordering;
use crate::digest::sha256_bytes;

/// Voicing
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Voicing { Voiced, Unvoiced }

/// Place of articulation
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Place {
    Bilabial, Labiodental, Dental, Alveolar,
    Postalveolar, Palatal, Velar, Glottal, None,
}

/// Manner of articulation
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Manner {
    Plosive, Nasal, Fricative, Affricate,
    Approximant, Liquid, Vowel,
}

/// Vowel height
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VowelHeight { High, Mid, Low, None }

/// Vowel backness
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VowelBackness { Front, Central, Back, None }

/// Rounding
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Rounded { Rounded, Unrounded, None }

/// A single certified phoneme.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Phoneme {
    pub id:        u8,
    pub symbol:    &'static str,   // ARPAbet symbol
    pub ipa:       &'static str,   // IPA symbol (display only)
    pub voicing:   Voicing,
    pub place:     Place,
    pub manner:    Manner,
    pub height:    VowelHeight,
    pub backness:  VowelBackness,
    pub rounded:   Rounded,
    pub sig:       u16,            // 12-bit predicate signature
}

impl Phoneme {
    /// Canonical byte encoding (keys sorted, explicit fields, no whitespace).
    /// Format mirrors phoneme_feature_schema_v1 JSON canonicalization rules.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let voicing   = match self.voicing   { Voicing::Voiced   => "voiced",   Voicing::Unvoiced => "unvoiced" };
        let place     = match self.place     { Place::Bilabial    => "bilabial", Place::Labiodental => "labiodental",
                                              Place::Dental      => "dental",   Place::Alveolar    => "alveolar",
                                              Place::Postalveolar=> "postalveolar", Place::Palatal => "palatal",
                                              Place::Velar       => "velar",    Place::Glottal     => "glottal",
                                              Place::None        => "none" };
        let manner    = match self.manner    { Manner::Plosive    => "plosive",  Manner::Nasal      => "nasal",
                                              Manner::Fricative  => "fricative",Manner::Affricate  => "affricate",
                                              Manner::Approximant=> "approximant", Manner::Liquid   => "liquid",
                                              Manner::Vowel      => "vowel" };
        let height    = match self.height    { VowelHeight::High  => "high", VowelHeight::Mid  => "mid",
                                              VowelHeight::Low   => "low",  VowelHeight::None => "none" };
        let backness  = match self.backness  { VowelBackness::Front   => "front", VowelBackness::Central => "central",
                                              VowelBackness::Back    => "back",  VowelBackness::None    => "none" };
        let rounded   = match self.rounded   { Rounded::Rounded   => "rounded", Rounded::Unrounded => "unrounded",
                                              Rounded::None      => "none" };
        // Keys sorted lexicographically, no whitespace
        let s = format!(
            "{{\"id\":{},\"manner\":\"{}\",\"place\":\"{}\",\"rounded\":\"{}\",\"symbol\":\"{}\",\"voicing\":\"{}\",\"vowel_backness\":\"{}\",\"vowel_height\":\"{}\"}}",
            self.id, manner, place, rounded, self.symbol, voicing, backness, height
        );
        s.into_bytes()
    }

    /// SHA-256 digest of canonical bytes.
    pub fn digest(&self) -> [u8; 32] {
        sha256_bytes(&self.canonical_bytes())
    }
}

/// Compute 12-bit signature from feature fields.
pub fn sig12(
    voiced: bool, plosive: bool, nasal: bool, fricative: bool,
    affricate: bool, approximant: bool, vowel: bool,
    bilabial: bool, alveolar: bool, velar: bool,
    front_vowel: bool, rounded: bool,
) -> u16 {
    let mut s = 0u16;
    if voiced      { s |= 1 << 0; }
    if plosive     { s |= 1 << 1; }
    if nasal       { s |= 1 << 2; }
    if fricative   { s |= 1 << 3; }
    if affricate   { s |= 1 << 4; }
    if approximant { s |= 1 << 5; }
    if vowel       { s |= 1 << 6; }
    if bilabial    { s |= 1 << 7; }
    if alveolar    { s |= 1 << 8; }
    if velar       { s |= 1 << 9; }
    if front_vowel { s |= 1 << 10; }
    if rounded     { s |= 1 << 11; }
    s
}

/// Canonical total order: id ascending.
pub fn canonical_cmp(a: &Phoneme, b: &Phoneme) -> Ordering {
    a.id.cmp(&b.id)
}

/// Hamming distance on 12-bit signatures.
pub fn sig_distance(a: &Phoneme, b: &Phoneme) -> u32 {
    ((a.sig ^ b.sig) as u32).count_ones()
}

/// Bit legend for the 12-bit signature.
pub fn bit_legend() -> [&'static str; 12] {
    ["voiced","plosive","nasal","fricative","affricate","approximant",
     "vowel","bilabial","alveolar","velar","front_vowel","rounded"]
}

/// The certified English phoneme inventory (44 phonemes).
/// IDs are stable and immutable.
pub fn build_phoneme_universe() -> Vec<Phoneme> {
    use Voicing::*; use Place::*; use Manner::*;
    use VowelHeight as H; use VowelBackness as B; use Rounded as R;

    // Helper closures
    let cons = |id, sym, ipa, v, pl, ma| Phoneme {
        id, symbol: sym, ipa,
        voicing: v, place: pl, manner: ma,
        height: H::None, backness: B::None, rounded: R::None,
        sig: sig12(
            v == Voiced,
            ma == Plosive, ma == Nasal, ma == Fricative, ma == Affricate,
            ma == Approximant || ma == Liquid, false,
            pl == Bilabial, pl == Alveolar, pl == Velar,
            false, false,
        ),
    };
    let vowel = |id, sym, ipa, ht, bk, rd| {
        let front = bk == B::Front;
        let round = rd == R::Rounded;
        Phoneme {
            id, symbol: sym, ipa,
            voicing: Voiced, place: None, manner: Vowel,
            height: ht, backness: bk, rounded: rd,
            sig: sig12(true, false,false,false,false,false, true,
                       false,false,false, front, round),
        }
    };

    vec![
        // ── Stops ──────────────────────────────────────────────────────
        cons( 1, "P",  "p", Unvoiced, Bilabial,     Plosive),
        cons( 2, "B",  "b", Voiced,   Bilabial,     Plosive),
        cons( 3, "T",  "t", Unvoiced, Alveolar,     Plosive),
        cons( 4, "D",  "d", Voiced,   Alveolar,     Plosive),
        cons( 5, "K",  "k", Unvoiced, Velar,        Plosive),
        cons( 6, "G",  "g", Voiced,   Velar,        Plosive),

        // ── Affricates ─────────────────────────────────────────────────
        cons( 7, "CH", "tʃ", Unvoiced, Postalveolar, Affricate),
        cons( 8, "JH", "dʒ", Voiced,   Postalveolar, Affricate),

        // ── Fricatives ─────────────────────────────────────────────────
        cons( 9, "F",  "f", Unvoiced, Labiodental,  Fricative),
        cons(10, "V",  "v", Voiced,   Labiodental,  Fricative),
        cons(11, "TH", "θ", Unvoiced, Dental,       Fricative),
        cons(12, "DH", "ð", Voiced,   Dental,       Fricative),
        cons(13, "S",  "s", Unvoiced, Alveolar,     Fricative),
        cons(14, "Z",  "z", Voiced,   Alveolar,     Fricative),
        cons(15, "SH", "ʃ", Unvoiced, Postalveolar, Fricative),
        cons(16, "ZH", "ʒ", Voiced,   Postalveolar, Fricative),
        cons(17, "HH", "h", Unvoiced, Glottal,      Fricative),

        // ── Nasals ─────────────────────────────────────────────────────
        cons(18, "M",  "m", Voiced, Bilabial,  Nasal),
        cons(19, "N",  "n", Voiced, Alveolar,  Nasal),
        cons(20, "NG", "ŋ", Voiced, Velar,     Nasal),

        // ── Liquids ────────────────────────────────────────────────────
        cons(21, "L",  "l", Voiced, Alveolar,  Liquid),
        cons(22, "R",  "r", Voiced, Alveolar,  Liquid),

        // ── Approximants / Glides ──────────────────────────────────────
        cons(23, "W",  "w", Voiced, Bilabial,  Approximant),
        cons(24, "Y",  "j", Voiced, Palatal,   Approximant),

        // ── Monophthong vowels ─────────────────────────────────────────
        vowel(25, "IY", "iː", H::High, B::Front,   R::Unrounded),
        vowel(26, "IH", "ɪ",  H::High, B::Front,   R::Unrounded),
        vowel(27, "EH", "ɛ",  H::Mid,  B::Front,   R::Unrounded),
        vowel(28, "AE", "æ",  H::Low,  B::Front,   R::Unrounded),
        vowel(29, "AH", "ʌ",  H::Mid,  B::Central, R::Unrounded),
        vowel(30, "AA", "ɑː", H::Low,  B::Back,    R::Unrounded),
        vowel(31, "AO", "ɔː", H::Mid,  B::Back,    R::Rounded),
        vowel(32, "UH", "ʊ",  H::High, B::Back,    R::Rounded),
        vowel(33, "UW", "uː", H::High, B::Back,    R::Rounded),
        vowel(34, "ER", "ɜː", H::Mid,  B::Central, R::Unrounded),

        // ── Diphthongs ─────────────────────────────────────────────────
        vowel(35, "EY", "eɪ", H::Mid,  B::Front,   R::Unrounded),
        vowel(36, "AY", "aɪ", H::Low,  B::Front,   R::Unrounded),
        vowel(37, "OY", "ɔɪ", H::Mid,  B::Back,    R::Rounded),
        vowel(38, "AW", "aʊ", H::Low,  B::Back,    R::Unrounded),
        vowel(39, "OW", "oʊ", H::Mid,  B::Back,    R::Rounded),

        // ── Reduced vowels ─────────────────────────────────────────────
        vowel(40, "AX", "ə",  H::Mid,  B::Central, R::Unrounded),
        vowel(41, "IX", "ɨ",  H::High, B::Central, R::Unrounded),

        // ── Syllabic consonants ────────────────────────────────────────
        Phoneme { id: 42, symbol: "EL", ipa: "l̩",
            voicing: Voiced, place: Alveolar, manner: Liquid,
            height: H::None, backness: B::None, rounded: R::None,
            sig: sig12(true,false,false,false,false,true,false,false,true,false,false,false) },
        Phoneme { id: 43, symbol: "EM", ipa: "m̩",
            voicing: Voiced, place: Bilabial, manner: Nasal,
            height: H::None, backness: B::None, rounded: R::None,
            sig: sig12(true,false,true,false,false,false,false,true,false,false,false,false) },
        Phoneme { id: 44, symbol: "EN", ipa: "n̩",
            voicing: Voiced, place: Alveolar, manner: Nasal,
            height: H::None, backness: B::None, rounded: R::None,
            sig: sig12(true,false,true,false,false,false,false,false,true,false,false,false) },
    ]
}

/// Merkle root of the phoneme universe.
/// universe_digest = merkle_root(sort(phoneme_digest_list))
pub fn phoneme_universe_digest(phonemes: &[Phoneme]) -> [u8; 32] {
    let mut leaves: Vec<[u8; 32]> = phonemes.iter().map(|p| p.digest()).collect();
    leaves.sort_unstable();
    crate::digest::merkle_root(&leaves)
}

/// Check if a universe name refers to the phoneme universe.
pub fn is_phoneme_universe(u: &str) -> bool {
    matches!(u.to_ascii_uppercase().as_str(), "PHONEME" | "PHO" | "PH")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phoneme_universe_size_certified() {
        let u = build_phoneme_universe();
        assert_eq!(u.len(), 44, "phoneme universe must have exactly 44 elements");
    }

    #[test]
    fn phoneme_ids_unique_and_sequential() {
        let u = build_phoneme_universe();
        for (i, p) in u.iter().enumerate() {
            assert_eq!(p.id as usize, i + 1, "id must be 1-indexed sequential");
        }
    }

    #[test]
    fn phoneme_symbols_unique() {
        let u = build_phoneme_universe();
        let mut syms: Vec<&str> = u.iter().map(|p| p.symbol).collect();
        let before = syms.len();
        syms.sort_unstable();
        syms.dedup();
        assert_eq!(syms.len(), before, "all symbols must be unique");
    }

    #[test]
    fn sig12_p_is_unvoiced_bilabial_plosive() {
        let u = build_phoneme_universe();
        let p = u.iter().find(|p| p.symbol == "P").unwrap();
        assert_eq!(p.sig & 1, 0, "P: unvoiced → bit0=0");
        assert_eq!((p.sig >> 1) & 1, 1, "P: plosive → bit1=1");
        assert_eq!((p.sig >> 7) & 1, 1, "P: bilabial → bit7=1");
        assert_eq!((p.sig >> 6) & 1, 0, "P: not vowel → bit6=0");
    }

    #[test]
    fn sig12_iy_is_voiced_front_high_vowel() {
        let u = build_phoneme_universe();
        let iy = u.iter().find(|p| p.symbol == "IY").unwrap();
        assert_eq!(iy.sig & 1, 1,         "IY: voiced → bit0=1");
        assert_eq!((iy.sig >> 6) & 1, 1,  "IY: vowel → bit6=1");
        assert_eq!((iy.sig >> 10) & 1, 1, "IY: front → bit10=1");
        assert_eq!((iy.sig >> 11) & 1, 0, "IY: unrounded → bit11=0");
    }

    #[test]
    fn phoneme_universe_digest_is_deterministic() {
        let u = build_phoneme_universe();
        let d1 = phoneme_universe_digest(&u);
        let d2 = phoneme_universe_digest(&u);
        assert_eq!(d1, d2, "digest must be deterministic");
    }

    #[test]
    fn canonical_bytes_sorted_keys() {
        let u = build_phoneme_universe();
        let p = &u[0]; // P
        let s = String::from_utf8(p.canonical_bytes()).unwrap();
        // Keys must appear in sorted order
        let keys = ["id","manner","place","rounded","symbol","voicing",
                    "vowel_backness","vowel_height"];
        let mut last_pos = 0usize;
        for k in &keys {
            let pos = s.find(k).expect("key must exist");
            assert!(pos >= last_pos, "key {} out of order", k);
            last_pos = pos;
        }
    }

    #[test]
    fn sig_distance_identical() {
        let u = build_phoneme_universe();
        assert_eq!(sig_distance(&u[0], &u[0]), 0);
    }

    #[test]
    fn sig_distance_b_p_is_one() {
        let u = build_phoneme_universe();
        let p = u.iter().find(|p| p.symbol == "P").unwrap();
        let b = u.iter().find(|p| p.symbol == "B").unwrap();
        // P and B differ only in voicing (bit0)
        assert_eq!(sig_distance(p, b), 1, "P/B differ only in voicing");
    }
}

// ── Layer trait implementation ────────────────────────────────────────────────

use crate::layer::{Layer, LayerId};

pub struct PhonemeLayer {
    inventory: Vec<Phoneme>,
}

impl PhonemeLayer {
    pub fn new() -> Self {
        PhonemeLayer { inventory: build_phoneme_universe() }
    }
}

impl Default for PhonemeLayer {
    fn default() -> Self { Self::new() }
}

impl Layer for PhonemeLayer {
    fn id(&self) -> LayerId { LayerId::Phoneme }

    fn len(&self) -> usize { self.inventory.len() }

    fn canonical_bytes(&self, i: usize) -> Vec<u8> {
        self.inventory[i].canonical_bytes()
    }

    fn sig(&self, i: usize) -> u16 {
        self.inventory[i].sig
    }

    fn render(&self, i: usize) -> String {
        // ph:<symbol>
        // e.g. ph:AE  ph:K  ph:IY
        format!("ph:{}", self.inventory[i].symbol)
    }

    fn universe_digest(&self) -> [u8; 32] {
        phoneme_universe_digest(&self.inventory)
    }
}

// ── Additional tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod layer_tests {
    use super::*;
    use crate::layer::Layer;

    #[test]
    fn phoneme_layer_len() {
        let l = PhonemeLayer::new();
        assert_eq!(l.len(), 44);
    }

    #[test]
    fn phoneme_layer_render_nonempty() {
        let l = PhonemeLayer::new();
        for i in 0..l.len() {
            let r = l.render(i);
            assert!(r.starts_with("ph:"), "render {} = {}", i, r);
        }
    }

    #[test]
    fn phoneme_layer_digest_stable() {
        let l = PhonemeLayer::new();
        for i in 0..l.len() {
            assert_eq!(l.digest(i), l.digest(i));
        }
    }

    #[test]
    fn phoneme_layer_sig_matches_inventory() {
        let l = PhonemeLayer::new();
        for i in 0..l.len() {
            assert_eq!(l.sig(i), l.inventory[i].sig);
        }
    }

    #[test]
    fn phoneme_layer_nearest_self() {
        let l = PhonemeLayer::new();
        let s = l.sig(0);
        let n = l.nearest(s).unwrap();
        assert_eq!(l.sig_distance(l.sig(n), s), 0);
    }

    #[test]
    fn phoneme_layer_top_k() {
        let l = PhonemeLayer::new();
        let k = l.top_k(l.sig(0), 5);
        assert!(k.len() <= 5);
        if k.len() >= 2 {
            assert!(l.sig_distance(l.sig(k[0]), l.sig(0))
                 <= l.sig_distance(l.sig(k[1]), l.sig(0)));
        }
    }

    #[test]
    fn phoneme_layer_universe_digest_stable() {
        let l = PhonemeLayer::new();
        assert_eq!(l.universe_digest(), l.universe_digest());
    }

    #[test]
    fn phoneme_layer_witness_roundtrip() {
        let l = PhonemeLayer::new();
        let w = l.witness(0);
        assert_eq!(w.layer, crate::layer::LayerId::Phoneme);
        assert!(!w.rendered.is_empty());
        assert_eq!(w.digest, l.digest(0));
    }

    #[test]
    fn phoneme_layer_renders_known_symbols() {
        let l = PhonemeLayer::new();
        let rendered: Vec<String> = (0..l.len()).map(|i| l.render(i)).collect();
        assert!(rendered.contains(&"ph:AE".to_string()));
        assert!(rendered.contains(&"ph:K".to_string()));
    }
}
