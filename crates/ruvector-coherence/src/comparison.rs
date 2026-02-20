//! Side-by-side comparison utilities for attention masks.

use serde::{Deserialize, Serialize};

/// Result of comparing two attention masks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Jaccard similarity coefficient between the two masks.
    pub jaccard: f64,
    /// Number of positions where one mask has an edge and the other does not.
    pub edge_flips: usize,
    /// Total active edges in the baseline mask.
    pub baseline_edges: usize,
    /// Total active edges in the gated mask.
    pub gated_edges: usize,
    /// Ratio of gated sparsity to baseline sparsity.
    /// Values > 1.0 mean the gated mask is denser; < 1.0 means sparser.
    pub sparsity_ratio: f64,
}

/// Computes the Jaccard similarity coefficient between two boolean masks.
///
/// `J(A, B) = |A intersection B| / |A union B|`
///
/// Returns `1.0` when both masks are empty (vacuously similar).
pub fn jaccard_similarity(mask_a: &[bool], mask_b: &[bool]) -> f64 {
    let n = mask_a.len().min(mask_b.len());
    let mut intersection = 0usize;
    let mut union = 0usize;
    for i in 0..n {
        let a = mask_a[i];
        let b = mask_b[i];
        if a || b {
            union += 1;
        }
        if a && b {
            intersection += 1;
        }
    }
    // Count remaining elements beyond the shorter slice as union-only.
    if mask_a.len() > n {
        union += mask_a[n..].iter().filter(|&&v| v).count();
    }
    if mask_b.len() > n {
        union += mask_b[n..].iter().filter(|&&v| v).count();
    }
    if union == 0 {
        return 1.0;
    }
    intersection as f64 / union as f64
}

/// Counts positions where the two masks disagree (one true, the other false).
pub fn edge_flip_count(mask_a: &[bool], mask_b: &[bool]) -> usize {
    let n = mask_a.len().min(mask_b.len());
    let mut flips = 0usize;
    for i in 0..n {
        if mask_a[i] != mask_b[i] {
            flips += 1;
        }
    }
    // Positions beyond the shorter mask count as flips if the longer mask is true.
    if mask_a.len() > n {
        flips += mask_a[n..].iter().filter(|&&v| v).count();
    }
    if mask_b.len() > n {
        flips += mask_b[n..].iter().filter(|&&v| v).count();
    }
    flips
}

/// Performs a full comparison of two attention masks.
pub fn compare_attention_masks(baseline: &[bool], gated: &[bool]) -> ComparisonResult {
    let jaccard = jaccard_similarity(baseline, gated);
    let edge_flips = edge_flip_count(baseline, gated);
    let baseline_edges = baseline.iter().filter(|&&v| v).count();
    let gated_edges = gated.iter().filter(|&&v| v).count();
    let total = baseline.len().max(gated.len());
    let baseline_sparsity = if total > 0 {
        1.0 - (baseline_edges as f64 / total as f64)
    } else {
        1.0
    };
    let gated_sparsity = if total > 0 {
        1.0 - (gated_edges as f64 / total as f64)
    } else {
        1.0
    };
    let sparsity_ratio = if baseline_sparsity > f64::EPSILON {
        gated_sparsity / baseline_sparsity
    } else {
        gated_sparsity
    };
    ComparisonResult {
        jaccard,
        edge_flips,
        baseline_edges,
        gated_edges,
        sparsity_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jaccard_identical() {
        let mask = vec![true, false, true, true];
        assert!((jaccard_similarity(&mask, &mask) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn jaccard_disjoint() {
        let a = vec![true, false, true, false];
        let b = vec![false, true, false, true];
        assert!(jaccard_similarity(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn jaccard_empty() {
        let empty: Vec<bool> = vec![];
        assert_eq!(jaccard_similarity(&empty, &empty), 1.0);
    }

    #[test]
    fn jaccard_all_false() {
        let a = vec![false, false, false];
        let b = vec![false, false, false];
        assert_eq!(jaccard_similarity(&a, &b), 1.0);
    }

    #[test]
    fn jaccard_partial_overlap() {
        let a = vec![true, true, false, false];
        let b = vec![true, false, true, false];
        // intersection = 1 (pos 0), union = 3 (pos 0, 1, 2)
        assert!((jaccard_similarity(&a, &b) - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn edge_flip_count_identical() {
        let mask = vec![true, false, true];
        assert_eq!(edge_flip_count(&mask, &mask), 0);
    }

    #[test]
    fn edge_flip_count_all_flipped() {
        let a = vec![true, false, true];
        let b = vec![false, true, false];
        assert_eq!(edge_flip_count(&a, &b), 3);
    }

    #[test]
    fn edge_flip_count_different_lengths() {
        let a = vec![true, false];
        let b = vec![true, false, true, true];
        // pos 0: same, pos 1: same, pos 2: flip, pos 3: flip
        assert_eq!(edge_flip_count(&a, &b), 2);
    }

    #[test]
    fn compare_attention_masks_basic() {
        let baseline = vec![true, true, false, false, true];
        let gated = vec![true, false, false, true, true];
        let result = compare_attention_masks(&baseline, &gated);
        assert_eq!(result.baseline_edges, 3);
        assert_eq!(result.gated_edges, 3);
        assert_eq!(result.edge_flips, 2);
        // intersection = 2 (pos 0, 4), union = 4 (pos 0, 1, 3, 4)
        assert!((result.jaccard - 0.5).abs() < 1e-10);
    }

    #[test]
    fn compare_sparser_gated() {
        let baseline = vec![true, true, true, true];
        let gated = vec![true, false, false, false];
        let result = compare_attention_masks(&baseline, &gated);
        assert_eq!(result.baseline_edges, 4);
        assert_eq!(result.gated_edges, 1);
        // baseline_sparsity = 0, so sparsity_ratio = gated_sparsity
        assert!(result.sparsity_ratio > 0.0);
    }
}
