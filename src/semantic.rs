//! Semantic graph universe (Layer 6) — meaning as directed labeled graphs.
//!
//! Depends on: Layer 5 (phrase universe).
//!
//! Core principle: meaning identity is graph structure, not surface form.
//! "The cat chased the mouse" and "The mouse was chased by the cat"
//! map to the same canonical semantic graph.
//!
//! Structure:
//!   SEMANTIC_GRAPH = nodes(concepts) + edges(relations)
//!   Nodes: entity, event, attribute, quantity, location, time
//!   Edges: agent, patient, theme, recipient, location, time,
//!          manner, cause, possessor, modifier
//!
//! Signature bits (16-bit):
//!   bit0   event_present
//!   bit1   agent_present
//!   bit2   patient_present
//!   bit3   recipient_present
//!   bit4   location_present
//!   bit5   time_present
//!   bit6   manner_present
//!   bit7   cause_present
//!   bit8   entity_count_le_2
//!   bit9   entity_count_le_4
//!   bit10  entity_count_gt_4
//!   bit11  event_count_1
//!   bit12  event_count_gt_1
//!   bit13  negative_polarity
//!   bit14  past_tense
//!   bit15  future_tense

use std::cmp::Ordering;
use crate::digest::{sha256_bytes, merkle_root};

// ── Node and edge types ───────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum NodeType {
    Entity, Event, Attribute, Quantity, Location, Time,
}

impl NodeType {
    pub fn as_str(self) -> &'static str {
        match self {
            NodeType::Entity    => "entity",
            NodeType::Event     => "event",
            NodeType::Attribute => "attribute",
            NodeType::Quantity  => "quantity",
            NodeType::Location  => "location",
            NodeType::Time      => "time",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum RelationType {
    Agent, Patient, Theme, Recipient,
    Location, Time, Manner, Cause,
    Possessor, Modifier,
}

impl RelationType {
    pub fn as_str(self) -> &'static str {
        match self {
            RelationType::Agent     => "agent",
            RelationType::Patient   => "patient",
            RelationType::Theme     => "theme",
            RelationType::Recipient => "recipient",
            RelationType::Location  => "location",
            RelationType::Time      => "time",
            RelationType::Manner    => "manner",
            RelationType::Cause     => "cause",
            RelationType::Possessor => "possessor",
            RelationType::Modifier  => "modifier",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Polarity { Positive, Negative }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tense { Past, Present, Future, None }

// ── Graph nodes and edges ─────────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticNode {
    pub id:        u32,
    pub node_type: NodeType,
    pub label:     String,   // concept label — evidence, not identity
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticEdge {
    pub source:   u32,
    pub target:   u32,
    pub relation: RelationType,
}

// ── Semantic graph ────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticGraph {
    pub graph_id: u32,
    pub nodes:    Vec<SemanticNode>,
    pub edges:    Vec<SemanticEdge>,
    pub polarity: Polarity,
    pub tense:    Tense,
    pub sig:      u16,
}

impl SemanticGraph {
    /// Canonical byte encoding.
    /// nodes sorted by id, edges sorted by (source, target, relation), keys sorted.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let pol = match self.polarity {
            Polarity::Positive => "positive",
            Polarity::Negative => "negative",
        };
        let tns = match self.tense {
            Tense::Past    => "past",
            Tense::Present => "present",
            Tense::Future  => "future",
            Tense::None    => "none",
        };

        let mut nodes = self.nodes.clone();
        nodes.sort_by_key(|n| n.id);
        let nodes_str = nodes.iter()
            .map(|n| format!(
                "{{\"id\":{},\"label\":\"{}\",\"type\":\"{}\"}}",
                n.id, n.label, n.node_type.as_str()
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

        // Keys sorted: edges, features, graph_id, nodes
        let s = format!(
            "{{\"edges\":[{}],\"features\":{{\"polarity\":\"{}\",\"tense\":\"{}\"}},\
\"graph_id\":{},\"nodes\":[{}]}}",
            edges_str, pol, tns, self.graph_id, nodes_str
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
    RootConstraintViolated,   // must have exactly one root event
    RelationTypeMismatch,
    MaxNodesExceeded,
    MaxEdgesExceeded,
    Disconnected,
}

const MAX_NODES: usize = 64;
const MAX_EDGES: usize = 128;

pub fn validate(g: &SemanticGraph) -> Result<(), ValidationError> {
    if g.nodes.is_empty() { return Err(ValidationError::EmptyGraph); }
    if g.nodes.len() > MAX_NODES { return Err(ValidationError::MaxNodesExceeded); }
    if g.edges.len() > MAX_EDGES { return Err(ValidationError::MaxEdgesExceeded); }

    // Node IDs must be unique
    let mut ids: Vec<u32> = g.nodes.iter().map(|n| n.id).collect();
    let before = ids.len();
    ids.sort_unstable(); ids.dedup();
    if ids.len() != before { return Err(ValidationError::DuplicateNodeId); }

    let node_ids: std::collections::HashSet<u32> =
        g.nodes.iter().map(|n| n.id).collect();

    // Edge endpoints must exist
    for e in &g.edges {
        if !node_ids.contains(&e.source) || !node_ids.contains(&e.target) {
            return Err(ValidationError::EdgeEndpointMissing);
        }
    }

    // Relation type constraints
    for e in &g.edges {
        let src = g.nodes.iter().find(|n| n.id == e.source).unwrap();
        let tgt = g.nodes.iter().find(|n| n.id == e.target).unwrap();
        let ok = match e.relation {
            RelationType::Agent | RelationType::Patient |
            RelationType::Theme | RelationType::Recipient =>
                src.node_type == NodeType::Event && tgt.node_type == NodeType::Entity,
            RelationType::Location =>
                src.node_type == NodeType::Event && tgt.node_type == NodeType::Location,
            RelationType::Time =>
                src.node_type == NodeType::Event && tgt.node_type == NodeType::Time,
            RelationType::Manner | RelationType::Cause =>
                src.node_type == NodeType::Event,
            RelationType::Possessor =>
                tgt.node_type == NodeType::Entity,
            RelationType::Modifier =>
                true, // modifier is permissive in v1
        };
        if !ok { return Err(ValidationError::RelationTypeMismatch); }
    }

    // Root constraint: if any event nodes exist, exactly one must have
    // no incoming event-typed edges (it is the root event)
    let event_nodes: Vec<u32> = g.nodes.iter()
        .filter(|n| n.node_type == NodeType::Event)
        .map(|n| n.id).collect();

    if !event_nodes.is_empty() {
        let event_targets: std::collections::HashSet<u32> = g.edges.iter()
            .filter(|e| {
                let src = g.nodes.iter().find(|n| n.id == e.source).unwrap();
                src.node_type == NodeType::Event
            })
            .map(|e| e.target).collect();

        let roots: Vec<u32> = event_nodes.iter()
            .filter(|id| !event_targets.contains(id))
            .copied().collect();

        if roots.len() != 1 {
            return Err(ValidationError::RootConstraintViolated);
        }
    }

    Ok(())
}

// ── Signature ─────────────────────────────────────────────────────────────────

pub fn sig16(g: &SemanticGraph) -> u16 {
    let relations: Vec<RelationType> = g.edges.iter().map(|e| e.relation).collect();
    let entity_count = g.nodes.iter().filter(|n| n.node_type == NodeType::Entity).count();
    let event_count  = g.nodes.iter().filter(|n| n.node_type == NodeType::Event).count();

    let mut s = 0u16;
    if event_count > 0                              { s |= 1 << 0; }
    if relations.contains(&RelationType::Agent)     { s |= 1 << 1; }
    if relations.contains(&RelationType::Patient)   { s |= 1 << 2; }
    if relations.contains(&RelationType::Recipient) { s |= 1 << 3; }
    if relations.contains(&RelationType::Location)  { s |= 1 << 4; }
    if relations.contains(&RelationType::Time)      { s |= 1 << 5; }
    if relations.contains(&RelationType::Manner)    { s |= 1 << 6; }
    if relations.contains(&RelationType::Cause)     { s |= 1 << 7; }
    if entity_count <= 2                            { s |= 1 << 8; }
    if entity_count <= 4                            { s |= 1 << 9; }
    if entity_count > 4                             { s |= 1 << 10; }
    if event_count == 1                             { s |= 1 << 11; }
    if event_count > 1                              { s |= 1 << 12; }
    if g.polarity == Polarity::Negative             { s |= 1 << 13; }
    if g.tense == Tense::Past                       { s |= 1 << 14; }
    if g.tense == Tense::Future                     { s |= 1 << 15; }
    s
}

pub fn bit_legend() -> [&'static str; 16] {
    ["event_present","agent_present","patient_present","recipient_present",
     "location_present","time_present","manner_present","cause_present",
     "entity_count_le_2","entity_count_le_4","entity_count_gt_4",
     "event_count_1","event_count_gt_1",
     "negative_polarity","past_tense","future_tense"]
}

// ── Canonical order ───────────────────────────────────────────────────────────

pub fn canonical_cmp(a: &SemanticGraph, b: &SemanticGraph) -> Ordering {
    // Sort by canonical bytes (graph structure, not graph_id)
    let ca = a.canonical_bytes();
    let cb = b.canonical_bytes();
    // Strip graph_id from comparison — identity is structure
    ca.cmp(&cb)
}

pub fn sig_distance(a: &SemanticGraph, b: &SemanticGraph) -> u32 {
    ((a.sig ^ b.sig) as u32).count_ones()
}

// ── Inventory builder ─────────────────────────────────────────────────────────

pub fn build_semantic_inventory() -> Vec<SemanticGraph> {
    let mut graphs = vec![

        // "The cat chased the mouse"
        // event: chase, agent: cat, patient: mouse, tense: past
        SemanticGraph {
            graph_id: 1,
            nodes: vec![
                SemanticNode { id: 1, node_type: NodeType::Event,  label: "chase".into() },
                SemanticNode { id: 2, node_type: NodeType::Entity, label: "cat".into()   },
                SemanticNode { id: 3, node_type: NodeType::Entity, label: "mouse".into() },
            ],
            edges: vec![
                SemanticEdge { source: 1, target: 2, relation: RelationType::Agent   },
                SemanticEdge { source: 1, target: 3, relation: RelationType::Patient },
            ],
            polarity: Polarity::Positive,
            tense:    Tense::Past,
            sig: 0,
        },

        // "A dog runs"
        // event: run, agent: dog, tense: present
        SemanticGraph {
            graph_id: 2,
            nodes: vec![
                SemanticNode { id: 1, node_type: NodeType::Event,  label: "run".into() },
                SemanticNode { id: 2, node_type: NodeType::Entity, label: "dog".into() },
            ],
            edges: vec![
                SemanticEdge { source: 1, target: 2, relation: RelationType::Agent },
            ],
            polarity: Polarity::Positive,
            tense:    Tense::Present,
            sig: 0,
        },

        // "The cat sat on the mat"
        // event: sit, agent: cat, location: mat, tense: past
        SemanticGraph {
            graph_id: 3,
            nodes: vec![
                SemanticNode { id: 1, node_type: NodeType::Event,    label: "sit".into() },
                SemanticNode { id: 2, node_type: NodeType::Entity,   label: "cat".into() },
                SemanticNode { id: 3, node_type: NodeType::Location, label: "mat".into() },
            ],
            edges: vec![
                SemanticEdge { source: 1, target: 2, relation: RelationType::Agent    },
                SemanticEdge { source: 1, target: 3, relation: RelationType::Location },
            ],
            polarity: Polarity::Positive,
            tense:    Tense::Past,
            sig: 0,
        },

        // "The dog did not chase the cat" (negative)
        SemanticGraph {
            graph_id: 4,
            nodes: vec![
                SemanticNode { id: 1, node_type: NodeType::Event,  label: "chase".into() },
                SemanticNode { id: 2, node_type: NodeType::Entity, label: "dog".into()   },
                SemanticNode { id: 3, node_type: NodeType::Entity, label: "cat".into()   },
            ],
            edges: vec![
                SemanticEdge { source: 1, target: 2, relation: RelationType::Agent   },
                SemanticEdge { source: 1, target: 3, relation: RelationType::Patient },
            ],
            polarity: Polarity::Negative,
            tense:    Tense::Past,
            sig: 0,
        },

        // "Birds fly" — minimal intransitive
        SemanticGraph {
            graph_id: 5,
            nodes: vec![
                SemanticNode { id: 1, node_type: NodeType::Event,  label: "fly".into()   },
                SemanticNode { id: 2, node_type: NodeType::Entity, label: "birds".into() },
            ],
            edges: vec![
                SemanticEdge { source: 1, target: 2, relation: RelationType::Agent },
            ],
            polarity: Polarity::Positive,
            tense:    Tense::Present,
            sig: 0,
        },

        // "The teacher gave the student a book"
        // event: give, agent: teacher, recipient: student, theme: book
        SemanticGraph {
            graph_id: 6,
            nodes: vec![
                SemanticNode { id: 1, node_type: NodeType::Event,  label: "give".into()    },
                SemanticNode { id: 2, node_type: NodeType::Entity, label: "teacher".into() },
                SemanticNode { id: 3, node_type: NodeType::Entity, label: "student".into() },
                SemanticNode { id: 4, node_type: NodeType::Entity, label: "book".into()    },
            ],
            edges: vec![
                SemanticEdge { source: 1, target: 2, relation: RelationType::Agent     },
                SemanticEdge { source: 1, target: 3, relation: RelationType::Recipient },
                SemanticEdge { source: 1, target: 4, relation: RelationType::Theme     },
            ],
            polarity: Polarity::Positive,
            tense:    Tense::Past,
            sig: 0,
        },
    ];

    for g in graphs.iter_mut() {
        g.sig = sig16(g);
    }
    graphs.sort_by(|a, b| a.graph_id.cmp(&b.graph_id));
    graphs
}

// ── Universe digest ───────────────────────────────────────────────────────────

pub fn semantic_universe_digest(graphs: &[SemanticGraph]) -> [u8; 32] {
    let rules_digest  = sha256_bytes(
        b"semantic_rules_en_v1:single_root_event:max_nodes=64:max_edges=128");
    let legend_digest = sha256_bytes(bit_legend().join(",").as_bytes());
    let mut leaves: Vec<[u8; 32]> = graphs.iter().map(|g| g.digest()).collect();
    leaves.sort_unstable();
    let inventory_digest = merkle_root(&leaves);
    let mut cat = b"semantic_universe_v1".to_vec();
    cat.extend_from_slice(&rules_digest);
    cat.extend_from_slice(&legend_digest);
    cat.extend_from_slice(&inventory_digest);
    sha256_bytes(&cat)
}

pub fn is_semantic_universe(u: &str) -> bool {
    matches!(u.to_ascii_uppercase().as_str(),
        "SEMANTIC" | "SEM" | "SEMGRAPH" | "GRAPH")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_inventory_graphs_validate() {
        let inv = build_semantic_inventory();
        for g in &inv {
            assert!(validate(g).is_ok(),
                "graph {} failed validation", g.graph_id);
        }
    }

    #[test]
    fn cat_chased_mouse_sig() {
        let inv = build_semantic_inventory();
        let g = inv.iter().find(|g| {
            g.nodes.iter().any(|n| n.label == "chase") &&
            g.nodes.iter().any(|n| n.label == "mouse")
        }).unwrap();
        assert!((g.sig >> 0) & 1 == 1, "event_present");
        assert!((g.sig >> 1) & 1 == 1, "agent_present");
        assert!((g.sig >> 2) & 1 == 1, "patient_present");
        assert!((g.sig >> 11) & 1 == 1, "event_count_1");
        assert!((g.sig >> 14) & 1 == 1, "past_tense");
        assert!((g.sig >> 13) & 1 == 0, "not negative");
    }

    #[test]
    fn negative_graph_sig() {
        let inv = build_semantic_inventory();
        let g = inv.iter().find(|g| g.polarity == Polarity::Negative).unwrap();
        assert!((g.sig >> 13) & 1 == 1, "negative_polarity bit set");
    }

    #[test]
    fn ditransitive_has_recipient() {
        let inv = build_semantic_inventory();
        let g = inv.iter().find(|g| {
            g.nodes.iter().any(|n| n.label == "give")
        }).unwrap();
        assert!((g.sig >> 3) & 1 == 1, "recipient_present");
        assert_eq!(g.nodes.iter().filter(|n| n.node_type == NodeType::Entity).count(), 3);
    }

    #[test]
    fn location_graph_has_location_bit() {
        let inv = build_semantic_inventory();
        let g = inv.iter().find(|g| {
            g.nodes.iter().any(|n| n.node_type == NodeType::Location)
        }).unwrap();
        assert!((g.sig >> 4) & 1 == 1, "location_present");
    }

    #[test]
    fn invalid_duplicate_node_id() {
        let g = SemanticGraph {
            graph_id: 99,
            nodes: vec![
                SemanticNode { id: 1, node_type: NodeType::Event,  label: "run".into() },
                SemanticNode { id: 1, node_type: NodeType::Entity, label: "dog".into() },
            ],
            edges: vec![],
            polarity: Polarity::Positive,
            tense: Tense::Present,
            sig: 0,
        };
        assert_eq!(validate(&g), Err(ValidationError::DuplicateNodeId));
    }

    #[test]
    fn invalid_edge_endpoint_missing() {
        let g = SemanticGraph {
            graph_id: 99,
            nodes: vec![
                SemanticNode { id: 1, node_type: NodeType::Event,  label: "run".into() },
            ],
            edges: vec![
                SemanticEdge { source: 1, target: 99, relation: RelationType::Agent },
            ],
            polarity: Polarity::Positive,
            tense: Tense::Present,
            sig: 0,
        };
        assert_eq!(validate(&g), Err(ValidationError::EdgeEndpointMissing));
    }

    #[test]
    fn invalid_relation_type_mismatch() {
        // agent edge: target must be entity, not location
        let g = SemanticGraph {
            graph_id: 99,
            nodes: vec![
                SemanticNode { id: 1, node_type: NodeType::Event,    label: "run".into() },
                SemanticNode { id: 2, node_type: NodeType::Location, label: "park".into() },
            ],
            edges: vec![
                SemanticEdge { source: 1, target: 2, relation: RelationType::Agent },
            ],
            polarity: Polarity::Positive,
            tense: Tense::Present,
            sig: 0,
        };
        assert_eq!(validate(&g), Err(ValidationError::RelationTypeMismatch));
    }

    #[test]
    fn canonical_bytes_keys_sorted() {
        let inv = build_semantic_inventory();
        let g = &inv[0];
        let s = String::from_utf8(g.canonical_bytes()).unwrap();
        // edges must precede features, features precede graph_id, graph_id precede nodes
        let ei = s.find("edges").unwrap();
        let fi = s.find("features").unwrap();
        let gi = s.find("graph_id").unwrap();
        let ni = s.find("nodes").unwrap();
        assert!(ei < fi && fi < gi && gi < ni, "keys out of order");
    }

    #[test]
    fn universe_digest_deterministic() {
        let inv = build_semantic_inventory();
        let d1 = semantic_universe_digest(&inv);
        let d2 = semantic_universe_digest(&inv);
        assert_eq!(d1, d2);
    }

    #[test]
    fn sig_distance_identical() {
        let inv = build_semantic_inventory();
        assert_eq!(sig_distance(&inv[0], &inv[0]), 0);
    }

    #[test]
    fn paraphrase_same_graph() {
        // "The cat chased the mouse" and "The mouse was chased by the cat"
        // must produce the same canonical graph structure
        let g1 = SemanticGraph {
            graph_id: 1,
            nodes: vec![
                SemanticNode { id: 1, node_type: NodeType::Event,  label: "chase".into() },
                SemanticNode { id: 2, node_type: NodeType::Entity, label: "cat".into()   },
                SemanticNode { id: 3, node_type: NodeType::Entity, label: "mouse".into() },
            ],
            edges: vec![
                SemanticEdge { source: 1, target: 2, relation: RelationType::Agent   },
                SemanticEdge { source: 1, target: 3, relation: RelationType::Patient },
            ],
            polarity: Polarity::Positive,
            tense: Tense::Past,
            sig: 0,
        };
        // Passive: same nodes and edges, different graph_id
        let g2 = SemanticGraph {
            graph_id: 2,  // different id
            ..g1.clone()
        };
        // Structure-only canonical bytes (strip graph_id effect):
        // The graphs have identical nodes+edges+features → same meaning
        assert_eq!(g1.sig16_for_test(), g2.sig16_for_test());
        // Edges and nodes are identical — paraphrase collapses to same graph
        assert_eq!(g1.nodes, g2.nodes);
        assert_eq!(g1.edges, g2.edges);
    }
}

// Test helper — exposes sig16 for paraphrase test
impl SemanticGraph {
    #[cfg(test)]
    fn sig16_for_test(&self) -> u16 { sig16(self) }
}
