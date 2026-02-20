//! Quality guardrails for attention mechanism output comparison.

use serde::{Deserialize, Serialize};

/// Result of a quality check comparing baseline and gated outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityResult {
    /// Cosine similarity between the two output vectors.
    pub cosine_sim: f64,
    /// Euclidean (L2) distance between the two output vectors.
    pub l2_dist: f64,
    /// Whether the cosine similarity meets or exceeds the threshold.
    pub passes_threshold: bool,
}

/// Computes cosine similarity between two vectors.
///
/// Returns `0.0` when either vector has zero magnitude.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0_f64;
    let mut norm_a_sq = 0.0_f64;
    let mut norm_b_sq = 0.0_f64;
    for i in 0..n {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a_sq += ai * ai;
        norm_b_sq += bi * bi;
    }
    let denom = norm_a_sq.sqrt() * norm_b_sq.sqrt();
    if denom < f64::EPSILON {
        return 0.0;
    }
    dot / denom
}

/// Computes the Euclidean (L2) distance between two vectors.
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut sum_sq = 0.0_f64;
    for i in 0..n {
        let diff = (a[i] as f64) - (b[i] as f64);
        sum_sq += diff * diff;
    }
    // Account for extra dimensions in the longer vector.
    if a.len() > n {
        for &v in &a[n..] {
            sum_sq += (v as f64) * (v as f64);
        }
    }
    if b.len() > n {
        for &v in &b[n..] {
            sum_sq += (v as f64) * (v as f64);
        }
    }
    sum_sq.sqrt()
}

/// Checks whether gated output quality is acceptable relative to the baseline.
///
/// The check passes when `cosine_similarity(baseline, gated) >= threshold`.
pub fn quality_check(
    baseline_output: &[f32],
    gated_output: &[f32],
    threshold: f64,
) -> QualityResult {
    let cosine_sim = cosine_similarity(baseline_output, gated_output);
    let l2_dist = l2_distance(baseline_output, gated_output);
    QualityResult {
        cosine_sim,
        l2_dist,
        passes_threshold: cosine_sim >= threshold,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn l2_distance_zero() {
        let v = vec![1.0, 2.0, 3.0];
        assert!(l2_distance(&v, &v) < 1e-10);
    }

    #[test]
    fn l2_distance_known() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((l2_distance(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn l2_distance_different_lengths() {
        let a = vec![1.0];
        let b = vec![1.0, 3.0];
        // diff at pos 0: 0, extra in b: 3.0 => sqrt(0 + 9) = 3.0
        assert!((l2_distance(&a, &b) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn quality_check_passes() {
        let baseline = vec![1.0, 2.0, 3.0];
        let gated = vec![1.1, 2.1, 3.1];
        let result = quality_check(&baseline, &gated, 0.99);
        assert!(result.passes_threshold);
        assert!(result.cosine_sim > 0.99);
        assert!(result.l2_dist > 0.0);
    }

    #[test]
    fn quality_check_fails() {
        let baseline = vec![1.0, 0.0];
        let gated = vec![0.0, 1.0];
        let result = quality_check(&baseline, &gated, 0.5);
        assert!(!result.passes_threshold);
        assert!(result.cosine_sim < 0.5);
    }

    #[test]
    fn quality_result_serializable() {
        let result = QualityResult {
            cosine_sim: 0.95,
            l2_dist: 0.32,
            passes_threshold: true,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deser: QualityResult = serde_json::from_str(&json).unwrap();
        assert!((deser.cosine_sim - 0.95).abs() < 1e-10);
        assert!(deser.passes_threshold);
    }
}
