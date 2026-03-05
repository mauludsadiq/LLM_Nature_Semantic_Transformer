//! Phrase universe (Layer 5) — syntactic tree structures over words.
//!
//! Depends on: Layer 4 (word universe).
//!
//! Structure: PHRASE = tree(WORD)
//! Identity is the tree structure, not the string.
//!
//! Grammar (v1):
//!   S  -> NP VP
//!   NP -> DetP N  |  N  |  DetP AP N  |  NP PP
//!   VP -> V  |  V NP  |  V NP PP  |  V PP
//!   PP -> P NP
//!   AP -> Adj  |  Adj AP
//!
//! Signature bits (16-bit):
//!   bit0   is_sentence       (root = S)
//!   bit1   has_subject       (NP child of S)
//!   bit2   has_object        (NP inside VP)
//!   bit3   has_modifier      (AP or AdvP present)
//!   bit4   contains_PP
//!   bit5   contains_AP
//!   bit6   contains_AdvP
//!   bit7   depth_ge_2
//!   bit8   depth_ge_3
//!   bit9   depth_ge_4
//!   bit10  word_count_le_3
//!   bit11  word_count_le_6
//!   bit12  word_count_gt_6
//!   bit13  transitive_vp     (VP contains V + NP)
//!   bit14  determiner_present
//!   bit15  plural_subject

use std::cmp::Ordering;
use crate::digest::{sha256_bytes, merkle_root};

// ── Node types ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum NodeType {
    S, NP, VP, PP, AP, AdvP, DetP,
    N, V, Adj, Adv, P, Det,   // pre-terminal (POS) nodes
}

impl NodeType {
    pub fn as_str(self) -> &'static str {
        match self {
            NodeType::S    => "S",
            NodeType::NP   => "NP",
            NodeType::VP   => "VP",
            NodeType::PP   => "PP",
            NodeType::AP   => "AP",
            NodeType::AdvP => "AdvP",
            NodeType::DetP => "DetP",
            NodeType::N    => "N",
            NodeType::V    => "V",
            NodeType::Adj  => "Adj",
            NodeType::Adv  => "Adv",
            NodeType::P    => "P",
            NodeType::Det  => "Det",
        }
    }

    pub fn is_preterminal(self) -> bool {
        matches!(self, NodeType::N | NodeType::V | NodeType::Adj |
                       NodeType::Adv | NodeType::P | NodeType::Det)
    }
}

// ── Phrase tree ───────────────────────────────────────────────────────────────

/// A node in a phrase tree.
/// Terminal nodes carry a word lemma_id; non-terminal nodes carry children.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PhraseNode {
    /// Non-terminal: typed node with children
    Inner {
        node_type: NodeType,
        children:  Vec<PhraseNode>,
    },
    /// Terminal: pre-terminal POS node with a word
    Terminal {
        node_type: NodeType,   // POS category
        word:      String,     // lemma_id e.g. "en:word:cat"
    },
}

impl PhraseNode {
    pub fn node_type(&self) -> NodeType {
        match self {
            PhraseNode::Inner    { node_type, .. } => *node_type,
            PhraseNode::Terminal { node_type, .. } => *node_type,
        }
    }

    /// Count terminal (word) nodes.
    pub fn word_count(&self) -> usize {
        match self {
            PhraseNode::Terminal { .. } => 1,
            PhraseNode::Inner { children, .. } =>
                children.iter().map(|c| c.word_count()).sum(),
        }
    }

    /// Tree depth (terminal = 1).
    pub fn depth(&self) -> usize {
        match self {
            PhraseNode::Terminal { .. } => 1,
            PhraseNode::Inner { children, .. } =>
                1 + children.iter().map(|c| c.depth()).max().unwrap_or(0),
        }
    }

    /// Collect all node types in tree.
    pub fn collect_types(&self, out: &mut Vec<NodeType>) {
        out.push(self.node_type());
        if let PhraseNode::Inner { children, .. } = self {
            for c in children { c.collect_types(out); }
        }
    }

    /// Collect terminal words in order.
    pub fn collect_words(&self, out: &mut Vec<String>) {
        match self {
            PhraseNode::Terminal { word, .. } => out.push(word.clone()),
            PhraseNode::Inner { children, .. } =>
                children.iter().for_each(|c| c.collect_words(out)),
        }
    }

    /// Canonical JSON — keys sorted, no whitespace, recursive.
    pub fn canonical_json(&self) -> String {
        match self {
            PhraseNode::Terminal { node_type, word } =>
                format!("{{\"node\":\"{}\",\"word\":\"{}\"}}",
                    node_type.as_str(), word),
            PhraseNode::Inner { node_type, children } => {
                let kids: Vec<String> = children.iter()
                    .map(|c| c.canonical_json()).collect();
                format!("{{\"children\":[{}],\"node\":\"{}\"}}",
                    kids.join(","), node_type.as_str())
            }
        }
    }
}

// ── Phrase struct ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Phrase {
    pub phrase_id: u32,
    pub root:      PhraseNode,
    pub sig:       u16,
}

impl Phrase {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        // Keys sorted: node tree is self-describing
        let s = format!("{{\"phrase_id\":{},\"tree\":{}}}",
            self.phrase_id, self.root.canonical_json());
        s.into_bytes()
    }

    pub fn digest(&self) -> [u8; 32] {
        sha256_bytes(&self.canonical_bytes())
    }

    pub fn word_count(&self) -> usize { self.root.word_count() }
    pub fn depth(&self) -> usize      { self.root.depth() }
}

// ── Validation ────────────────────────────────────────────────────────────────

#[derive(Debug, Eq, PartialEq)]
pub enum ValidationError {
    EmptyChildren,
    NodeTypeMismatch,
    RuleMismatch,
    WordMissing,
    PosMappingInvalid,
    MaxDepthExceeded,
    MaxWordsExceeded,
}

const MAX_DEPTH: usize = 8;
const MAX_WORDS: usize = 32;

/// Validate a phrase tree against grammar v1.
pub fn validate(node: &PhraseNode) -> Result<(), ValidationError> {
    validate_node(node, 0)
}

fn validate_node(node: &PhraseNode, depth: usize) -> Result<(), ValidationError> {
    if depth > MAX_DEPTH { return Err(ValidationError::MaxDepthExceeded); }

    match node {
        PhraseNode::Terminal { node_type, word } => {
            if word.is_empty() { return Err(ValidationError::WordMissing); }
            if !node_type.is_preterminal() {
                return Err(ValidationError::PosMappingInvalid);
            }
            Ok(())
        }
        PhraseNode::Inner { node_type, children } => {
            if children.is_empty() { return Err(ValidationError::EmptyChildren); }
            if node.word_count() > MAX_WORDS {
                return Err(ValidationError::MaxWordsExceeded);
            }

            // Validate children recursively first
            for c in children { validate_node(c, depth + 1)?; }

            // Grammar rule check
            let child_types: Vec<NodeType> = children.iter()
                .map(|c| c.node_type()).collect();

            let valid = match node_type {
                NodeType::S => matches!(child_types.as_slice(),
                    [NodeType::NP, NodeType::VP]),

                NodeType::NP => matches!(child_types.as_slice(),
                    [NodeType::N]
                    | [NodeType::Det, NodeType::N]
                    | [NodeType::DetP, NodeType::N]
                    | [NodeType::Det, NodeType::Adj, NodeType::N]
                    | [NodeType::DetP, NodeType::AP, NodeType::N]
                    | [NodeType::NP, NodeType::PP]),

                NodeType::VP => matches!(child_types.as_slice(),
                    [NodeType::V]
                    | [NodeType::V, NodeType::NP]
                    | [NodeType::V, NodeType::NP, NodeType::PP]
                    | [NodeType::V, NodeType::PP]
                    | [NodeType::V, NodeType::AP]),

                NodeType::PP => matches!(child_types.as_slice(),
                    [NodeType::P, NodeType::NP]),

                NodeType::AP => matches!(child_types.as_slice(),
                    [NodeType::Adj]
                    | [NodeType::Adj, NodeType::AP]),

                NodeType::AdvP => matches!(child_types.as_slice(),
                    [NodeType::Adv]),

                NodeType::DetP => matches!(child_types.as_slice(),
                    [NodeType::Det]),

                _ => return Err(ValidationError::NodeTypeMismatch),
            };

            if !valid { Err(ValidationError::RuleMismatch) } else { Ok(()) }
        }
    }
}

// ── Signature ─────────────────────────────────────────────────────────────────

pub fn sig16(phrase: &Phrase) -> u16 {
    let mut types = Vec::new();
    phrase.root.collect_types(&mut types);

    let wc    = phrase.word_count();
    let depth = phrase.depth();

    let is_s   = phrase.root.node_type() == NodeType::S;
    let has_np = types.contains(&NodeType::NP);
    let _has_vp = types.contains(&NodeType::VP);
    let has_pp = types.contains(&NodeType::PP);
    let has_ap = types.contains(&NodeType::AP);
    let has_advp = types.contains(&NodeType::AdvP);
    let has_det = types.contains(&NodeType::Det) || types.contains(&NodeType::DetP);

    // has_object: NP that is a child of VP
    let has_obj = if let PhraseNode::Inner { node_type: NodeType::S, children } = &phrase.root {
        children.iter().any(|c| {
            if let PhraseNode::Inner { node_type: NodeType::VP, children: vp_kids } = c {
                vp_kids.iter().any(|k| k.node_type() == NodeType::NP)
            } else { false }
        })
    } else { false };

    // transitive VP: V + NP
    let transitive = if let PhraseNode::Inner { node_type: NodeType::S, children } = &phrase.root {
        children.iter().any(|c| {
            if let PhraseNode::Inner { node_type: NodeType::VP, children: vp_kids } = c {
                let vp_types: Vec<NodeType> = vp_kids.iter().map(|k| k.node_type()).collect();
                matches!(vp_types.as_slice(),
                    [NodeType::V, NodeType::NP] | [NodeType::V, NodeType::NP, NodeType::PP])
            } else { false }
        })
    } else { false };

    let mut s = 0u16;
    if is_s                  { s |= 1 << 0; }
    if is_s && has_np        { s |= 1 << 1; }
    if has_obj               { s |= 1 << 2; }
    if has_ap || has_advp    { s |= 1 << 3; }
    if has_pp                { s |= 1 << 4; }
    if has_ap                { s |= 1 << 5; }
    if has_advp              { s |= 1 << 6; }
    if depth >= 2            { s |= 1 << 7; }
    if depth >= 3            { s |= 1 << 8; }
    if depth >= 4            { s |= 1 << 9; }
    if wc <= 3               { s |= 1 << 10; }
    if wc <= 6               { s |= 1 << 11; }
    if wc > 6                { s |= 1 << 12; }
    if transitive            { s |= 1 << 13; }
    if has_det               { s |= 1 << 14; }
    // bit15: placeholder for plural subject (requires word feature lookup)
    s
}

pub fn bit_legend() -> [&'static str; 16] {
    ["is_sentence","has_subject","has_object","has_modifier",
     "contains_PP","contains_AP","contains_AdvP",
     "depth_ge_2","depth_ge_3","depth_ge_4",
     "word_count_le_3","word_count_le_6","word_count_gt_6",
     "transitive_vp","determiner_present","plural_subject"]
}

// ── Canonical order ───────────────────────────────────────────────────────────

pub fn canonical_cmp(a: &Phrase, b: &Phrase) -> Ordering {
    a.root.canonical_json().cmp(&b.root.canonical_json())
}

pub fn sig_distance(a: &Phrase, b: &Phrase) -> u32 {
    ((a.sig ^ b.sig) as u32).count_ones()
}

// ── Inventory builder ─────────────────────────────────────────────────────────

/// Build a certified sample phrase inventory (v1).
/// These are canonical example trees — the universe is defined by the
/// validator + grammar rules, not this finite list.
pub fn build_phrase_inventory() -> Vec<Phrase> {
    let mut phrases = vec![

        // "the cat chased the mouse"
        // S -> NP VP
        //   NP -> Det N
        //   VP -> V NP
        //          NP -> Det N
        make_phrase(1, PhraseNode::Inner {
            node_type: NodeType::S,
            children: vec![
                PhraseNode::Inner { node_type: NodeType::NP, children: vec![
                    PhraseNode::Terminal { node_type: NodeType::Det, word: "en:word:the".into() },
                    PhraseNode::Terminal { node_type: NodeType::N,   word: "en:word:cat".into() },
                ]},
                PhraseNode::Inner { node_type: NodeType::VP, children: vec![
                    PhraseNode::Terminal { node_type: NodeType::V, word: "en:word:chased".into() },
                    PhraseNode::Inner { node_type: NodeType::NP, children: vec![
                        PhraseNode::Terminal { node_type: NodeType::Det, word: "en:word:the".into() },
                        PhraseNode::Terminal { node_type: NodeType::N,   word: "en:word:mouse".into() },
                    ]},
                ]},
            ],
        }),

        // "a dog runs"
        // S -> NP VP
        //   NP -> Det N
        //   VP -> V
        make_phrase(2, PhraseNode::Inner {
            node_type: NodeType::S,
            children: vec![
                PhraseNode::Inner { node_type: NodeType::NP, children: vec![
                    PhraseNode::Terminal { node_type: NodeType::Det, word: "en:word:a".into() },
                    PhraseNode::Terminal { node_type: NodeType::N,   word: "en:word:dog".into() },
                ]},
                PhraseNode::Inner { node_type: NodeType::VP, children: vec![
                    PhraseNode::Terminal { node_type: NodeType::V, word: "en:word:runs".into() },
                ]},
            ],
        }),

        // "the small cat sat on the mat"
        // S -> NP VP
        //   NP -> Det Adj N
        //   VP -> V PP
        //          PP -> P NP (Det N)
        make_phrase(3, PhraseNode::Inner {
            node_type: NodeType::S,
            children: vec![
                PhraseNode::Inner { node_type: NodeType::NP, children: vec![
                    PhraseNode::Terminal { node_type: NodeType::Det, word: "en:word:the".into() },
                    PhraseNode::Terminal { node_type: NodeType::Adj, word: "en:word:small".into() },
                    PhraseNode::Terminal { node_type: NodeType::N,   word: "en:word:cat".into() },
                ]},
                PhraseNode::Inner { node_type: NodeType::VP, children: vec![
                    PhraseNode::Terminal { node_type: NodeType::V, word: "en:word:sat".into() },
                    PhraseNode::Inner { node_type: NodeType::PP, children: vec![
                        PhraseNode::Terminal { node_type: NodeType::P, word: "en:word:on".into() },
                        PhraseNode::Inner { node_type: NodeType::NP, children: vec![
                            PhraseNode::Terminal { node_type: NodeType::Det, word: "en:word:the".into() },
                            PhraseNode::Terminal { node_type: NodeType::N,   word: "en:word:mat".into() },
                        ]},
                    ]},
                ]},
            ],
        }),

        // "birds fly"
        // S -> NP VP
        //   NP -> N
        //   VP -> V
        make_phrase(4, PhraseNode::Inner {
            node_type: NodeType::S,
            children: vec![
                PhraseNode::Inner { node_type: NodeType::NP, children: vec![
                    PhraseNode::Terminal { node_type: NodeType::N, word: "en:word:birds".into() },
                ]},
                PhraseNode::Inner { node_type: NodeType::VP, children: vec![
                    PhraseNode::Terminal { node_type: NodeType::V, word: "en:word:fly".into() },
                ]},
            ],
        }),

        // NP only: "the old house"
        make_phrase(5, PhraseNode::Inner {
            node_type: NodeType::NP,
            children: vec![
                PhraseNode::Terminal { node_type: NodeType::Det, word: "en:word:the".into() },
                PhraseNode::Terminal { node_type: NodeType::Adj, word: "en:word:old".into() },
                PhraseNode::Terminal { node_type: NodeType::N,   word: "en:word:house".into() },
            ],
        }),
    ];

    for p in phrases.iter_mut() {
        p.sig = sig16(p);
    }
    phrases.sort_by(canonical_cmp);
    for (i, p) in phrases.iter_mut().enumerate() {
        p.phrase_id = (i + 1) as u32;
    }
    phrases
}

fn make_phrase(id: u32, root: PhraseNode) -> Phrase {
    Phrase { phrase_id: id, root, sig: 0 }
}

// ── Universe digest ───────────────────────────────────────────────────────────

pub fn phrase_universe_digest(phrases: &[Phrase]) -> [u8; 32] {
    let rules_digest  = sha256_bytes(b"phrase_rules_en_v1:grammar=S_NP_VP:max_depth=8:max_words=32");
    let legend_digest = sha256_bytes(bit_legend().join(",").as_bytes());
    let mut leaves: Vec<[u8; 32]> = phrases.iter().map(|p| p.digest()).collect();
    leaves.sort_unstable();
    let inventory_digest = merkle_root(&leaves);
    let mut cat = b"phrase_universe_v1".to_vec();
    cat.extend_from_slice(&rules_digest);
    cat.extend_from_slice(&legend_digest);
    cat.extend_from_slice(&inventory_digest);
    sha256_bytes(&cat)
}

pub fn is_phrase_universe(u: &str) -> bool {
    matches!(u.to_ascii_uppercase().as_str(), "PHRASE" | "PHR" | "PH")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_inventory_phrases_validate() {
        let inv = build_phrase_inventory();
        for p in &inv {
            assert!(validate(&p.root).is_ok(),
                "phrase {} failed validation", p.phrase_id);
        }
    }

    #[test]
    fn cat_chased_mouse_sig() {
        let inv = build_phrase_inventory();
        // First phrase after sort: find by content
        let p = inv.iter().find(|p| {
            let mut words = Vec::new();
            p.root.collect_words(&mut words);
            words.contains(&"en:word:chased".to_string())
        }).unwrap();
        let s = p.sig;
        assert!((s >> 0) & 1 == 1, "is_sentence");
        assert!((s >> 1) & 1 == 1, "has_subject");
        assert!((s >> 2) & 1 == 1, "has_object");
        assert!((s >> 13) & 1 == 1, "transitive_vp");
        assert!((s >> 14) & 1 == 1, "determiner_present");
    }

    #[test]
    fn birds_fly_is_intransitive() {
        let inv = build_phrase_inventory();
        let p = inv.iter().find(|p| {
            let mut words = Vec::new();
            p.root.collect_words(&mut words);
            words.contains(&"en:word:fly".to_string())
        }).unwrap();
        assert!((p.sig >> 13) & 1 == 0, "birds fly is intransitive");
        assert!((p.sig >> 14) & 1 == 0, "no determiner");
    }

    #[test]
    fn pp_phrase_has_pp_bit() {
        let inv = build_phrase_inventory();
        let p = inv.iter().find(|p| {
            let mut words = Vec::new();
            p.root.collect_words(&mut words);
            words.contains(&"en:word:on".to_string())
        }).unwrap();
        assert!((p.sig >> 4) & 1 == 1, "contains_PP");
    }

    #[test]
    fn invalid_empty_children() {
        let node = PhraseNode::Inner {
            node_type: NodeType::NP,
            children: vec![],
        };
        assert_eq!(validate(&node), Err(ValidationError::EmptyChildren));
    }

    #[test]
    fn invalid_rule_mismatch() {
        // NP -> V N is not a valid rule
        let node = PhraseNode::Inner {
            node_type: NodeType::NP,
            children: vec![
                PhraseNode::Terminal { node_type: NodeType::V, word: "en:word:run".into() },
                PhraseNode::Terminal { node_type: NodeType::N, word: "en:word:cat".into() },
            ],
        };
        assert_eq!(validate(&node), Err(ValidationError::RuleMismatch));
    }

    #[test]
    fn canonical_json_keys_sorted() {
        let inv = build_phrase_inventory();
        let p = &inv[0];
        let j = p.root.canonical_json();
        // Inner node must have "children" before "node"
        let ci = j.find("children").unwrap();
        let ni = j.find("node").unwrap();
        assert!(ci < ni, "children must precede node in canonical JSON");
    }

    #[test]
    fn word_count_and_depth() {
        let inv = build_phrase_inventory();
        let p = inv.iter().find(|p| {
            let mut w = Vec::new(); p.root.collect_words(&mut w);
            w.contains(&"en:word:chased".to_string())
        }).unwrap();
        assert_eq!(p.word_count(), 5); // the cat chased the mouse
        assert!(p.depth() >= 3);
    }

    #[test]
    fn inventory_sorted_and_unique() {
        let inv = build_phrase_inventory();
        for i in 1..inv.len() {
            assert!(canonical_cmp(&inv[i-1], &inv[i]).is_lt(),
                "not sorted at index {}", i);
        }
    }

    #[test]
    fn universe_digest_deterministic() {
        let inv = build_phrase_inventory();
        let d1 = phrase_universe_digest(&inv);
        let d2 = phrase_universe_digest(&inv);
        assert_eq!(d1, d2);
    }

    #[test]
    fn sig_distance_identical() {
        let inv = build_phrase_inventory();
        assert_eq!(sig_distance(&inv[0], &inv[0]), 0);
    }
}

// ── Layer trait implementation ────────────────────────────────────────────────

use crate::layer::{Layer, LayerId};

pub struct PhraseLayer {
    inventory: Vec<Phrase>,
}

impl PhraseLayer {
    pub fn new() -> Self {
        let inv = build_phrase_inventory();
        PhraseLayer { inventory: inv }
    }
}

impl Default for PhraseLayer {
    fn default() -> Self { Self::new() }
}

impl Layer for PhraseLayer {
    fn id(&self) -> LayerId { LayerId::Phrase }

    fn len(&self) -> usize { self.inventory.len() }

    fn canonical_bytes(&self, i: usize) -> Vec<u8> {
        self.inventory[i].canonical_bytes()
    }

    fn sig(&self, i: usize) -> u16 {
        self.inventory[i].sig
    }

    fn render(&self, i: usize) -> String {
        let p = &self.inventory[i];
        // phrase:<phrase_id>:<tree_s_expression>
        fn render_node(node: &PhraseNode) -> String {
            match node {
                PhraseNode::Terminal { node_type, word } =>
                    format!("[{}:{}]", node_type.as_str(), word),
                PhraseNode::Inner { node_type, children } => {
                    let kids: Vec<String> = children.iter().map(render_node).collect();
                    format!("[{}{}]", node_type.as_str(), kids.join(""))
                }
            }
        }
        format!("phrase:{}:{}", p.phrase_id, render_node(&p.root))
    }

    fn universe_digest(&self) -> [u8; 32] {
        phrase_universe_digest(&self.inventory)
    }
}

// ── Additional tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod layer_tests {
    use super::*;
    use crate::layer::Layer;

    #[test]
    fn phrase_layer_len() {
        let l = PhraseLayer::new();
        assert_eq!(l.len(), 5);
    }

    #[test]
    fn phrase_layer_render_nonempty() {
        let l = PhraseLayer::new();
        for i in 0..l.len() {
            let r = l.render(i);
            assert!(r.starts_with("phrase:"), "render {} = {}", i, r);
        }
    }

    #[test]
    fn phrase_layer_render_contains_brackets() {
        let l = PhraseLayer::new();
        for i in 0..l.len() {
            let r = l.render(i);
            assert!(r.contains('[') && r.contains(']'),
                "render {} missing brackets: {}", i, r);
        }
    }

    #[test]
    fn phrase_layer_digest_stable() {
        let l = PhraseLayer::new();
        for i in 0..l.len() {
            assert_eq!(l.digest(i), l.digest(i));
        }
    }

    #[test]
    fn phrase_layer_sig_matches_inventory() {
        let l = PhraseLayer::new();
        let inv = build_phrase_inventory();
        for (i, p) in inv.iter().enumerate() {
            assert_eq!(l.sig(i), p.sig);
        }
    }

    #[test]
    fn phrase_layer_nearest_self() {
        let l = PhraseLayer::new();
        let s = l.sig(0);
        let n = l.nearest(s).unwrap();
        assert_eq!(l.sig_distance(l.sig(n), s), 0);
    }

    #[test]
    fn phrase_layer_top_k() {
        let l = PhraseLayer::new();
        let k = l.top_k(l.sig(0), 3);
        assert!(k.len() <= 3);
        if k.len() >= 2 {
            assert!(l.sig_distance(l.sig(k[0]), l.sig(0))
                 <= l.sig_distance(l.sig(k[1]), l.sig(0)));
        }
    }

    #[test]
    fn phrase_layer_universe_digest_stable() {
        let l = PhraseLayer::new();
        assert_eq!(l.universe_digest(), l.universe_digest());
    }

    #[test]
    fn phrase_layer_witness_roundtrip() {
        let l = PhraseLayer::new();
        let w = l.witness(0);
        assert_eq!(w.layer, crate::layer::LayerId::Phrase);
        assert!(!w.rendered.is_empty());
        assert_eq!(w.digest, l.digest(0));
    }

    #[test]
    fn phrase_layer_render_has_s_root() {
        let l = PhraseLayer::new();
        // Most certified phrases are S→NP VP; at least one must have S root
        let has_s = (0..l.len()).any(|i| l.render(i).contains("[S"));
        assert!(has_s, "no phrase with S root found");
        // All renders must start with phrase:
        for i in 0..l.len() {
            assert!(l.render(i).starts_with("phrase:"));
        }
    }
}
