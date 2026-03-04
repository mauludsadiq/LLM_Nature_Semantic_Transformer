//! Discourse universe (Layer 7) — knowledge graphs over semantic graphs.
//!
//! Depends on: Layer 6 (semantic universe).
//!
//! Core principle: discourse is a time-ordered sequence of semantic graphs
//! connected by discourse relations. Ambiguity that cannot be resolved
//! becomes a typed truth_status:"unknown" node — not an open interpretation.
//!
//! Structure:
//!   DISCOURSE = nodes(discourse_units) + edges(discourse_relations)
//!   Nodes: semantic_graph_ref, coreference_chain, unknown_referent
//!   Edges: coreference, causation, temporal_order, contrast,
//!          elaboration, entailment, background, unknown
//!
//! Bounds:
//!   max_nodes = 512
//!   max_edges = 1024
//!
//! Signature bits (16-bit):
//!   bit0   coreference_present
//!   bit1   causation_present
//!   bit2   temporal_order_present
//!   bit3   contrast_present
//!   bit4   elaboration_present
//!   bit5   entailment_present
//!   bit6   background_present
//!   bit7   unknown_relation_present
//!   bit8   unknown_referent_present
//!   bit9   graph_count_le_2
//!   bit10  graph_count_le_5
//!   bit11  graph_count_gt_5
//!   bit12  has_coreference_chain
//!   bit13  multi_event_chain
//!   bit14  negation_present
//!   bit15  resolved_ambiguity

use std::cmp::Ordering;
use crate::digest::{sha256_bytes, merkle_root};

// ── Node and edge types ───────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum DiscourseRelation {
    Coreference,
    Causation,
    TemporalOrder,
    Contrast,
    Elaboration,
    Entailment,
    Background,
    Unknown,
}

impl DiscourseRelation {
    pub fn as_str(self) -> &'static str {
        match self {
            DiscourseRelation::Coreference   => "coreference",
            DiscourseRelation::Causation     => "causation",
            DiscourseRelation::TemporalOrder => "temporal_order",
            DiscourseRelation::Contrast      => "contrast",
            DiscourseRelation::Elaboration   => "elaboration",
            DiscourseRelation::Entailment    => "entailment",
            DiscourseRelation::Background    => "background",
            DiscourseRelation::Unknown       => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DiscourseNodeType {
    SemanticGraphRef,    // reference to a certified semantic graph
    CoreferenceChain,    // resolved coreference cluster
    UnknownReferent,     // typed unknown — ambiguity made explicit
}

impl DiscourseNodeType {
    pub fn as_str(self) -> &'static str {
        match self {
            DiscourseNodeType::SemanticGraphRef  => "semantic_graph_ref",
            DiscourseNodeType::CoreferenceChain  => "coreference_chain",
            DiscourseNodeType::UnknownReferent   => "unknown_referent",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TruthStatus {
    Known,
    Unknown,
    Contested,
}

impl TruthStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            TruthStatus::Known    => "known",
            TruthStatus::Unknown  => "unknown",
            TruthStatus::Contested => "contested",
        }
    }
}

// ── Discourse nodes and edges ─────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscourseNode {
    pub id:           u32,
    pub node_type:    DiscourseNodeType,
    pub label:        String,
    pub truth_status: TruthStatus,
    pub sequence_pos: u32,   // position in discourse (0-indexed, for temporal order)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscourseEdge {
    pub source:   u32,
    pub target:   u32,
    pub relation: DiscourseRelation,
}

// ── Discourse graph ───────────────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscourseGraph {
    pub discourse_id: u32,
    pub nodes:        Vec<DiscourseNode>,
    pub edges:        Vec<DiscourseEdge>,
    pub sig:          u16,
}

impl DiscourseGraph {
    /// Canonical byte encoding: keys sorted, nodes sorted by id,
    /// edges sorted by (source, target, relation).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut nodes = self.nodes.clone();
        nodes.sort_by_key(|n| n.id);
        let nodes_str = nodes.iter()
            .map(|n| format!(
                "{{\"id\":{},\"label\":\"{}\",\"seq\":{},\"truth\":\"{}\",\"type\":\"{}\"}}",
                n.id, n.label, n.sequence_pos,
                n.truth_status.as_str(), n.node_type.as_str()
            ))
            .collect::<Vec<_>>().join(",");

        let mut edges = self.edges.clone();
        edges.sort_by(|a, b| a.source.cmp(&b.source)
            .then(a.target.cmp(&b.target))
            .then(a.relation.cmp(&b.relation)));
        let edges_str = edges.iter()
            .map(|e| format!(
                "{{\"relation\":\"{}\",\"source\":{},\"target\":{}}}",
                e.relation.as_str(), e.source, e.target
            ))
            .collect::<Vec<_>>().join(",");

        // Keys sorted: discourse_id, edges, nodes
        let s = format!(
            "{{\"discourse_id\":{},\"edges\":[{}],\"nodes\":[{}]}}",
            self.discourse_id, edges_str, nodes_str
        );
        s.into_bytes()
    }

    pub fn digest(&self) -> [u8; 32] {
        sha256_bytes(&self.canonical_bytes())
    }
}

// ── Validation ────────────────────────────────────────────────────────────────

#[derive(Debug, Eq, PartialEq)]
pub enum ValidationError {
    EmptyGraph,
    DuplicateNodeId,
    EdgeEndpointMissing,
    MaxNodesExceeded,
    MaxEdgesExceeded,
    OpenAmbiguity,          // ambiguous referent must be typed unknown, not absent
    TemporalOrderCycle,     // temporal_order edges must be acyclic
    SelfLoop,
}

pub const MAX_NODES: usize = 512;
pub const MAX_EDGES: usize = 1024;

pub fn validate(g: &DiscourseGraph) -> Result<(), ValidationError> {
    if g.nodes.is_empty() { return Err(ValidationError::EmptyGraph); }
    if g.nodes.len() > MAX_NODES { return Err(ValidationError::MaxNodesExceeded); }
    if g.edges.len() > MAX_EDGES { return Err(ValidationError::MaxEdgesExceeded); }

    // Node IDs unique
    let mut ids: Vec<u32> = g.nodes.iter().map(|n| n.id).collect();
    let before = ids.len();
    ids.sort_unstable(); ids.dedup();
    if ids.len() != before { return Err(ValidationError::DuplicateNodeId); }

    let node_ids: std::collections::HashSet<u32> =
        g.nodes.iter().map(|n| n.id).collect();

    // No self-loops
    for e in &g.edges {
        if e.source == e.target { return Err(ValidationError::SelfLoop); }
    }

    // Edge endpoints exist
    for e in &g.edges {
        if !node_ids.contains(&e.source) || !node_ids.contains(&e.target) {
            return Err(ValidationError::EdgeEndpointMissing);
        }
    }

    // Temporal order edges must be acyclic (simple cycle detection via DFS)
    let temporal_edges: Vec<(u32, u32)> = g.edges.iter()
        .filter(|e| e.relation == DiscourseRelation::TemporalOrder)
        .map(|e| (e.source, e.target))
        .collect();

    if has_cycle(&temporal_edges, &node_ids) {
        return Err(ValidationError::TemporalOrderCycle);
    }

    Ok(())
}

/// DFS cycle detection on a directed edge set.
fn has_cycle(
    edges: &[(u32, u32)],
    nodes: &std::collections::HashSet<u32>,
) -> bool {
    use std::collections::HashMap;
    let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(s, t) in edges {
        adj.entry(s).or_default().push(t);
    }

    // 0=unvisited, 1=in_stack, 2=done
    let mut state: HashMap<u32, u8> = nodes.iter().map(|&id| (id, 0)).collect();

    fn dfs(
        n: u32,
        adj: &HashMap<u32, Vec<u32>>,
        state: &mut HashMap<u32, u8>,
    ) -> bool {
        *state.get_mut(&n).unwrap() = 1;
        if let Some(neighbors) = adj.get(&n) {
            for &nb in neighbors {
                match state.get(&nb).copied().unwrap_or(0) {
                    1 => return true,
                    0 => if dfs(nb, adj, state) { return true; },
                    _ => {}
                }
            }
        }
        *state.get_mut(&n).unwrap() = 2;
        false
    }

    for &id in nodes {
        if state.get(&id).copied().unwrap_or(0) == 0 {
            if dfs(id, &adj, &mut state) { return true; }
        }
    }
    false
}

// ── Signature ─────────────────────────────────────────────────────────────────

pub fn sig16(g: &DiscourseGraph) -> u16 {
    let relations: Vec<DiscourseRelation> = g.edges.iter().map(|e| e.relation).collect();
    let graph_count = g.nodes.iter()
        .filter(|n| n.node_type == DiscourseNodeType::SemanticGraphRef).count();
    let has_coref_chain = g.nodes.iter()
        .any(|n| n.node_type == DiscourseNodeType::CoreferenceChain);
    let has_unknown = g.nodes.iter()
        .any(|n| n.node_type == DiscourseNodeType::UnknownReferent);
    let multi_event = graph_count > 1;
    let has_neg = g.nodes.iter()
        .any(|n| n.label.contains("neg") || n.label.contains("not"));
    let resolved = g.nodes.iter()
        .any(|n| n.truth_status == TruthStatus::Known
            && n.node_type == DiscourseNodeType::CoreferenceChain);

    let mut s = 0u16;
    if relations.contains(&DiscourseRelation::Coreference)   { s |= 1 << 0; }
    if relations.contains(&DiscourseRelation::Causation)      { s |= 1 << 1; }
    if relations.contains(&DiscourseRelation::TemporalOrder)  { s |= 1 << 2; }
    if relations.contains(&DiscourseRelation::Contrast)       { s |= 1 << 3; }
    if relations.contains(&DiscourseRelation::Elaboration)    { s |= 1 << 4; }
    if relations.contains(&DiscourseRelation::Entailment)     { s |= 1 << 5; }
    if relations.contains(&DiscourseRelation::Background)     { s |= 1 << 6; }
    if relations.contains(&DiscourseRelation::Unknown)        { s |= 1 << 7; }
    if has_unknown                                            { s |= 1 << 8; }
    if graph_count <= 2                                       { s |= 1 << 9; }
    if graph_count <= 5                                       { s |= 1 << 10; }
    if graph_count > 5                                        { s |= 1 << 11; }
    if has_coref_chain                                        { s |= 1 << 12; }
    if multi_event                                            { s |= 1 << 13; }
    if has_neg                                                { s |= 1 << 14; }
    if resolved                                               { s |= 1 << 15; }
    s
}

pub fn bit_legend() -> [&'static str; 16] {
    ["coreference_present","causation_present","temporal_order_present",
     "contrast_present","elaboration_present","entailment_present",
     "background_present","unknown_relation_present",
     "unknown_referent_present",
     "graph_count_le_2","graph_count_le_5","graph_count_gt_5",
     "has_coreference_chain","multi_event_chain",
     "negation_present","resolved_ambiguity"]
}

// ── Canonical order ───────────────────────────────────────────────────────────

pub fn canonical_cmp(a: &DiscourseGraph, b: &DiscourseGraph) -> Ordering {
    a.discourse_id.cmp(&b.discourse_id)
}

pub fn sig_distance(a: &DiscourseGraph, b: &DiscourseGraph) -> u32 {
    ((a.sig ^ b.sig) as u32).count_ones()
}

// ── Inventory builder ─────────────────────────────────────────────────────────

pub fn build_discourse_inventory() -> Vec<DiscourseGraph> {
    let mut graphs = vec![

        // Discourse 1: "The cat chased the mouse. It ran away."
        // Two semantic graph refs connected by temporal_order + coreference
        // "it" is resolved to "mouse" via coreference chain
        DiscourseGraph {
            discourse_id: 1,
            nodes: vec![
                DiscourseNode {
                    id: 1, node_type: DiscourseNodeType::SemanticGraphRef,
                    label: "chase(cat,mouse)".into(),
                    truth_status: TruthStatus::Known, sequence_pos: 0,
                },
                DiscourseNode {
                    id: 2, node_type: DiscourseNodeType::SemanticGraphRef,
                    label: "run_away(mouse)".into(),
                    truth_status: TruthStatus::Known, sequence_pos: 1,
                },
                DiscourseNode {
                    id: 3, node_type: DiscourseNodeType::CoreferenceChain,
                    label: "it=mouse".into(),
                    truth_status: TruthStatus::Known, sequence_pos: 1,
                },
            ],
            edges: vec![
                DiscourseEdge { source: 1, target: 2,
                    relation: DiscourseRelation::TemporalOrder },
                DiscourseEdge { source: 3, target: 2,
                    relation: DiscourseRelation::Coreference },
            ],
            sig: 0,
        },

        // Discourse 2: "The dog barked because it was afraid."
        // Two events connected by causation
        DiscourseGraph {
            discourse_id: 2,
            nodes: vec![
                DiscourseNode {
                    id: 1, node_type: DiscourseNodeType::SemanticGraphRef,
                    label: "bark(dog)".into(),
                    truth_status: TruthStatus::Known, sequence_pos: 0,
                },
                DiscourseNode {
                    id: 2, node_type: DiscourseNodeType::SemanticGraphRef,
                    label: "afraid(dog)".into(),
                    truth_status: TruthStatus::Known, sequence_pos: 0,
                },
            ],
            edges: vec![
                DiscourseEdge { source: 2, target: 1,
                    relation: DiscourseRelation::Causation },
            ],
            sig: 0,
        },

        // Discourse 3: "The cat sat. The dog stood."
        // Contrast between two events
        DiscourseGraph {
            discourse_id: 3,
            nodes: vec![
                DiscourseNode {
                    id: 1, node_type: DiscourseNodeType::SemanticGraphRef,
                    label: "sit(cat)".into(),
                    truth_status: TruthStatus::Known, sequence_pos: 0,
                },
                DiscourseNode {
                    id: 2, node_type: DiscourseNodeType::SemanticGraphRef,
                    label: "stand(dog)".into(),
                    truth_status: TruthStatus::Known, sequence_pos: 1,
                },
            ],
            edges: vec![
                DiscourseEdge { source: 1, target: 2,
                    relation: DiscourseRelation::Contrast },
            ],
            sig: 0,
        },

        // Discourse 4: "Someone took the book. They left quickly."
        // Unresolved referent — typed unknown, not open
        DiscourseGraph {
            discourse_id: 4,
            nodes: vec![
                DiscourseNode {
                    id: 1, node_type: DiscourseNodeType::SemanticGraphRef,
                    label: "take(unknown,book)".into(),
                    truth_status: TruthStatus::Known, sequence_pos: 0,
                },
                DiscourseNode {
                    id: 2, node_type: DiscourseNodeType::SemanticGraphRef,
                    label: "leave_quickly(unknown)".into(),
                    truth_status: TruthStatus::Known, sequence_pos: 1,
                },
                DiscourseNode {
                    id: 3, node_type: DiscourseNodeType::UnknownReferent,
                    label: "someone=unknown".into(),
                    truth_status: TruthStatus::Unknown, sequence_pos: 0,
                },
            ],
            edges: vec![
                DiscourseEdge { source: 1, target: 2,
                    relation: DiscourseRelation::TemporalOrder },
                DiscourseEdge { source: 3, target: 1,
                    relation: DiscourseRelation::Unknown },
            ],
            sig: 0,
        },

        // Discourse 5: Three-event chain with elaboration
        // "The bird sang. It was a robin. The robin had a red breast."
        DiscourseGraph {
            discourse_id: 5,
            nodes: vec![
                DiscourseNode {
                    id: 1, node_type: DiscourseNodeType::SemanticGraphRef,
                    label: "sing(bird)".into(),
                    truth_status: TruthStatus::Known, sequence_pos: 0,
                },
                DiscourseNode {
                    id: 2, node_type: DiscourseNodeType::SemanticGraphRef,
                    label: "is_a(bird,robin)".into(),
                    truth_status: TruthStatus::Known, sequence_pos: 1,
                },
                DiscourseNode {
                    id: 3, node_type: DiscourseNodeType::SemanticGraphRef,
                    label: "has(robin,red_breast)".into(),
                    truth_status: TruthStatus::Known, sequence_pos: 2,
                },
                DiscourseNode {
                    id: 4, node_type: DiscourseNodeType::CoreferenceChain,
                    label: "it=bird=robin".into(),
                    truth_status: TruthStatus::Known, sequence_pos: 1,
                },
            ],
            edges: vec![
                DiscourseEdge { source: 1, target: 2,
                    relation: DiscourseRelation::TemporalOrder },
                DiscourseEdge { source: 2, target: 3,
                    relation: DiscourseRelation::Elaboration },
                DiscourseEdge { source: 4, target: 2,
                    relation: DiscourseRelation::Coreference },
                DiscourseEdge { source: 4, target: 3,
                    relation: DiscourseRelation::Coreference },
            ],
            sig: 0,
        },
    ];

    for g in graphs.iter_mut() {
        g.sig = sig16(g);
    }
    graphs.sort_by(canonical_cmp);
    graphs
}

// ── Universe digest ───────────────────────────────────────────────────────────

pub fn discourse_universe_digest(graphs: &[DiscourseGraph]) -> [u8; 32] {
    let rules_digest  = sha256_bytes(
        b"discourse_rules_v1:max_nodes=512:max_edges=1024:ambiguity=typed_unknown:temporal=acyclic");
    let legend_digest = sha256_bytes(bit_legend().join(",").as_bytes());
    let mut leaves: Vec<[u8; 32]> = graphs.iter().map(|g| g.digest()).collect();
    leaves.sort_unstable();
    let inventory_digest = merkle_root(&leaves);
    let mut cat = b"discourse_universe_v1".to_vec();
    cat.extend_from_slice(&rules_digest);
    cat.extend_from_slice(&legend_digest);
    cat.extend_from_slice(&inventory_digest);
    sha256_bytes(&cat)
}

pub fn is_discourse_universe(u: &str) -> bool {
    matches!(u.to_ascii_uppercase().as_str(),
        "DISCOURSE" | "DIS" | "DISC")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_inventory_graphs_validate() {
        let inv = build_discourse_inventory();
        for g in &inv {
            assert!(validate(g).is_ok(),
                "discourse {} failed validation", g.discourse_id);
        }
    }

    #[test]
    fn coreference_discourse_sig() {
        let inv = build_discourse_inventory();
        let g = inv.iter().find(|g| {
            g.edges.iter().any(|e| e.relation == DiscourseRelation::Coreference)
        }).unwrap();
        assert!((g.sig >> 0) & 1 == 1, "coreference_present");
        assert!((g.sig >> 12) & 1 == 1, "has_coreference_chain");
    }

    #[test]
    fn causation_discourse_sig() {
        let inv = build_discourse_inventory();
        let g = inv.iter().find(|g| {
            g.edges.iter().any(|e| e.relation == DiscourseRelation::Causation)
        }).unwrap();
        assert!((g.sig >> 1) & 1 == 1, "causation_present");
    }

    #[test]
    fn unknown_referent_sig() {
        let inv = build_discourse_inventory();
        let g = inv.iter().find(|g| {
            g.nodes.iter().any(|n| n.node_type == DiscourseNodeType::UnknownReferent)
        }).unwrap();
        assert!((g.sig >> 8) & 1 == 1, "unknown_referent_present");
        assert!((g.sig >> 7) & 1 == 1, "unknown_relation_present");
    }

    #[test]
    fn temporal_order_is_acyclic() {
        // Valid: 1 -> 2 -> 3
        let g = DiscourseGraph {
            discourse_id: 99,
            nodes: vec![
                DiscourseNode { id: 1, node_type: DiscourseNodeType::SemanticGraphRef,
                    label: "e1".into(), truth_status: TruthStatus::Known, sequence_pos: 0 },
                DiscourseNode { id: 2, node_type: DiscourseNodeType::SemanticGraphRef,
                    label: "e2".into(), truth_status: TruthStatus::Known, sequence_pos: 1 },
                DiscourseNode { id: 3, node_type: DiscourseNodeType::SemanticGraphRef,
                    label: "e3".into(), truth_status: TruthStatus::Known, sequence_pos: 2 },
            ],
            edges: vec![
                DiscourseEdge { source: 1, target: 2, relation: DiscourseRelation::TemporalOrder },
                DiscourseEdge { source: 2, target: 3, relation: DiscourseRelation::TemporalOrder },
            ],
            sig: 0,
        };
        assert!(validate(&g).is_ok());
    }

    #[test]
    fn temporal_order_cycle_rejected() {
        // Cycle: 1 -> 2 -> 1
        let g = DiscourseGraph {
            discourse_id: 99,
            nodes: vec![
                DiscourseNode { id: 1, node_type: DiscourseNodeType::SemanticGraphRef,
                    label: "e1".into(), truth_status: TruthStatus::Known, sequence_pos: 0 },
                DiscourseNode { id: 2, node_type: DiscourseNodeType::SemanticGraphRef,
                    label: "e2".into(), truth_status: TruthStatus::Known, sequence_pos: 1 },
            ],
            edges: vec![
                DiscourseEdge { source: 1, target: 2, relation: DiscourseRelation::TemporalOrder },
                DiscourseEdge { source: 2, target: 1, relation: DiscourseRelation::TemporalOrder },
            ],
            sig: 0,
        };
        assert_eq!(validate(&g), Err(ValidationError::TemporalOrderCycle));
    }

    #[test]
    fn self_loop_rejected() {
        let g = DiscourseGraph {
            discourse_id: 99,
            nodes: vec![
                DiscourseNode { id: 1, node_type: DiscourseNodeType::SemanticGraphRef,
                    label: "e1".into(), truth_status: TruthStatus::Known, sequence_pos: 0 },
            ],
            edges: vec![
                DiscourseEdge { source: 1, target: 1, relation: DiscourseRelation::Elaboration },
            ],
            sig: 0,
        };
        assert_eq!(validate(&g), Err(ValidationError::SelfLoop));
    }

    #[test]
    fn invalid_edge_endpoint() {
        let g = DiscourseGraph {
            discourse_id: 99,
            nodes: vec![
                DiscourseNode { id: 1, node_type: DiscourseNodeType::SemanticGraphRef,
                    label: "e1".into(), truth_status: TruthStatus::Known, sequence_pos: 0 },
            ],
            edges: vec![
                DiscourseEdge { source: 1, target: 99, relation: DiscourseRelation::Causation },
            ],
            sig: 0,
        };
        assert_eq!(validate(&g), Err(ValidationError::EdgeEndpointMissing));
    }

    #[test]
    fn canonical_bytes_keys_sorted() {
        let inv = build_discourse_inventory();
        let g = &inv[0];
        let s = String::from_utf8(g.canonical_bytes()).unwrap();
        let di = s.find("discourse_id").unwrap();
        let ei = s.find("edges").unwrap();
        let ni = s.find("nodes").unwrap();
        assert!(di < ei && ei < ni, "keys out of order");
    }

    #[test]
    fn universe_digest_deterministic() {
        let inv = build_discourse_inventory();
        let d1 = discourse_universe_digest(&inv);
        let d2 = discourse_universe_digest(&inv);
        assert_eq!(d1, d2);
    }

    #[test]
    fn sig_distance_identical() {
        let inv = build_discourse_inventory();
        assert_eq!(sig_distance(&inv[0], &inv[0]), 0);
    }

    #[test]
    fn elaboration_chain_has_multi_event() {
        let inv = build_discourse_inventory();
        let g = inv.iter().find(|g| {
            g.edges.iter().any(|e| e.relation == DiscourseRelation::Elaboration)
        }).unwrap();
        assert!((g.sig >> 13) & 1 == 1, "multi_event_chain");
        assert!((g.sig >> 4) & 1 == 1, "elaboration_present");
    }
}
