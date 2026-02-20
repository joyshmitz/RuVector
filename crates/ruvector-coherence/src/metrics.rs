//! Core coherence metrics for attention mechanism evaluation.

use serde::{Deserialize, Serialize};

/// Result of comparing baseline vs. gated attention outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaMetric {
    /// Change in coherence score (gated minus baseline).
    pub coherence_delta: f64,
    /// Number of positions where the top-1 decision flipped.
    pub decision_flips: usize,
    /// Relative change in mean path length (L2 norm ratio).
    pub path_length_change: f64,
}

/// Measures the rate of contradictory outputs between predictions and references.
///
/// For each pair `(prediction, reference)`, a contradiction is detected when
/// the cosine similarity between the two vectors is negative (opposing directions).
///
/// Returns a value in `[0.0, 1.0]` representing the fraction of contradictory pairs.
pub fn contradiction_rate(predictions: &[Vec<f32>], references: &[Vec<f32>]) -> f64 {
    if predictions.is_empty() || references.is_empty() {
        return 0.0;
    }
    let n = predictions.len().min(references.len());
    let contradictions = predictions[..n]
        .iter()
        .zip(references[..n].iter())
        .filter(|(pred, refv)| {
            let dot: f64 = pred
                .iter()
                .zip(refv.iter())
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
            dot < 0.0
        })
        .count();
    contradictions as f64 / n as f64
}

/// Checks consistency of outputs across sequence positions.
///
/// Computes pairwise cosine similarities between consecutive output vectors
/// and returns the mean similarity. A value near `1.0` means highly consistent;
/// near `0.0` means largely independent outputs.
pub fn entailment_consistency(outputs: &[Vec<f32>]) -> f64 {
    if outputs.len() < 2 {
        return 1.0;
    }
    let mut total_sim = 0.0;
    let pairs = outputs.len() - 1;
    for i in 0..pairs {
        total_sim += pairwise_cosine(&outputs[i], &outputs[i + 1]);
    }
    total_sim / pairs as f64
}

/// Computes the behavioral delta between baseline and gated attention outputs.
pub fn delta_behavior(baseline_outputs: &[f32], gated_outputs: &[f32]) -> DeltaMetric {
    let n = baseline_outputs.len().min(gated_outputs.len());
    if n == 0 {
        return DeltaMetric {
            coherence_delta: 0.0,
            decision_flips: 0,
            path_length_change: 0.0,
        };
    }

    let baseline_slice = &baseline_outputs[..n];
    let gated_slice = &gated_outputs[..n];

    // Coherence delta: cosine similarity between the two output vectors.
    let coherence_delta = pairwise_cosine(baseline_slice, gated_slice) - 1.0;

    // Decision flips: positions where the sign of the value changes.
    let decision_flips = baseline_slice
        .iter()
        .zip(gated_slice.iter())
        .filter(|(b, g)| b.is_sign_positive() != g.is_sign_positive())
        .count();

    // Path length change: ratio of L2 norms (gated / baseline) - 1.
    let baseline_norm = l2_norm(baseline_slice);
    let gated_norm = l2_norm(gated_slice);
    let path_length_change = if baseline_norm > f64::EPSILON {
        (gated_norm / baseline_norm) - 1.0
    } else {
        0.0
    };

    DeltaMetric {
        coherence_delta,
        decision_flips,
        path_length_change,
    }
}

fn pairwise_cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    let denom = norm_a * norm_b;
    if denom < f64::EPSILON {
        return 0.0;
    }
    dot / denom
}

fn l2_norm(v: &[f32]) -> f64 {
    v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contradiction_rate_no_contradictions() {
        let preds = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let refs = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        assert_eq!(contradiction_rate(&preds, &refs), 0.0);
    }

    #[test]
    fn contradiction_rate_all_contradictions() {
        let preds = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let refs = vec![vec![-1.0, -1.0], vec![-1.0, -1.0]];
        assert_eq!(contradiction_rate(&preds, &refs), 1.0);
    }

    #[test]
    fn contradiction_rate_empty() {
        let empty: Vec<Vec<f32>> = vec![];
        assert_eq!(contradiction_rate(&empty, &empty), 0.0);
    }

    #[test]
    fn entailment_consistency_identical() {
        let outputs = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0]];
        assert!((entailment_consistency(&outputs) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn entailment_consistency_single() {
        let outputs = vec![vec![1.0, 0.0]];
        assert_eq!(entailment_consistency(&outputs), 1.0);
    }

    #[test]
    fn entailment_consistency_orthogonal() {
        let outputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!(entailment_consistency(&outputs).abs() < 1e-10);
    }

    #[test]
    fn delta_behavior_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let d = delta_behavior(&v, &v);
        assert!(d.coherence_delta.abs() < 1e-10);
        assert_eq!(d.decision_flips, 0);
        assert!(d.path_length_change.abs() < 1e-10);
    }

    #[test]
    fn delta_behavior_flips() {
        let baseline = vec![1.0, -1.0, 1.0];
        let gated = vec![-1.0, 1.0, 1.0];
        let d = delta_behavior(&baseline, &gated);
        assert_eq!(d.decision_flips, 2);
    }

    #[test]
    fn delta_behavior_empty() {
        let d = delta_behavior(&[], &[]);
        assert_eq!(d.decision_flips, 0);
        assert_eq!(d.coherence_delta, 0.0);
    }
}
