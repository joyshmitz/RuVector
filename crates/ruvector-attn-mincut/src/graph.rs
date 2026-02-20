use serde::{Deserialize, Serialize};

/// A directed edge in the attention graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub src: usize,
    pub dst: usize,
    pub weight: f32,
}

/// Weighted directed graph built from attention logits.
#[derive(Debug, Clone)]
pub struct AttentionGraph {
    pub nodes: usize,
    pub edges: Vec<Edge>,
}

/// Build a weighted directed graph from attention logits Q*K^T / sqrt(d).
///
/// `logits` is a flattened `seq_len x seq_len` matrix in row-major order.
/// Each positive logit becomes an edge; non-positive logits are omitted so the
/// graph is sparse when many logits are near zero or negative.
pub fn graph_from_logits(logits: &[f32], seq_len: usize) -> AttentionGraph {
    assert_eq!(
        logits.len(),
        seq_len * seq_len,
        "logits length must equal seq_len^2"
    );

    let mut edges = Vec::new();
    for i in 0..seq_len {
        for j in 0..seq_len {
            let w = logits[i * seq_len + j];
            if w > 0.0 {
                edges.push(Edge {
                    src: i,
                    dst: j,
                    weight: w,
                });
            }
        }
    }

    AttentionGraph {
        nodes: seq_len,
        edges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_from_logits_basic() {
        // 2x2 logits: all positive
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let g = graph_from_logits(&logits, 2);
        assert_eq!(g.nodes, 2);
        assert_eq!(g.edges.len(), 4);
    }

    #[test]
    fn test_graph_filters_non_positive() {
        let logits = vec![1.0, -0.5, 0.0, 2.0];
        let g = graph_from_logits(&logits, 2);
        // Only (0,0)=1.0 and (1,1)=2.0 survive
        assert_eq!(g.edges.len(), 2);
        assert_eq!(g.edges[0].src, 0);
        assert_eq!(g.edges[0].dst, 0);
        assert_eq!(g.edges[1].src, 1);
        assert_eq!(g.edges[1].dst, 1);
    }

    #[test]
    #[should_panic(expected = "logits length must equal seq_len^2")]
    fn test_graph_mismatched_length() {
        graph_from_logits(&[1.0, 2.0], 3);
    }

    #[test]
    fn test_graph_empty() {
        let logits = vec![-1.0; 9];
        let g = graph_from_logits(&logits, 3);
        assert_eq!(g.nodes, 3);
        assert!(g.edges.is_empty());
    }
}
