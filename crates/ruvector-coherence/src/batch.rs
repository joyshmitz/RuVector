//! Batched evaluation over multiple samples.

use serde::{Deserialize, Serialize};

use crate::metrics::delta_behavior;
use crate::quality::quality_check;

/// Aggregated results from evaluating a batch of baseline/gated output pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    /// Mean coherence delta across all samples.
    pub mean_coherence_delta: f64,
    /// Standard deviation of coherence delta values.
    pub std_coherence_delta: f64,
    /// Lower bound of 95% confidence interval for coherence delta.
    pub ci_95_lower: f64,
    /// Upper bound of 95% confidence interval for coherence delta.
    pub ci_95_upper: f64,
    /// Number of samples evaluated.
    pub n_samples: usize,
    /// Fraction of samples that pass the quality threshold.
    pub pass_rate: f64,
}

/// Evaluates a batch of baseline/gated output pairs and produces aggregate statistics.
///
/// Each pair `(baseline_outputs[i], gated_outputs[i])` is evaluated for its
/// coherence delta (via [`delta_behavior`]) and quality (via [`quality_check`]).
///
/// # Panics
///
/// Does not panic; returns zeroed results when inputs are empty.
pub fn evaluate_batch(
    baseline_outputs: &[Vec<f32>],
    gated_outputs: &[Vec<f32>],
    threshold: f64,
) -> BatchResult {
    let n = baseline_outputs.len().min(gated_outputs.len());
    if n == 0 {
        return BatchResult {
            mean_coherence_delta: 0.0,
            std_coherence_delta: 0.0,
            ci_95_lower: 0.0,
            ci_95_upper: 0.0,
            n_samples: 0,
            pass_rate: 0.0,
        };
    }

    let mut deltas = Vec::with_capacity(n);
    let mut passes = 0usize;

    for i in 0..n {
        let dm = delta_behavior(&baseline_outputs[i], &gated_outputs[i]);
        deltas.push(dm.coherence_delta);

        let qr = quality_check(&baseline_outputs[i], &gated_outputs[i], threshold);
        if qr.passes_threshold {
            passes += 1;
        }
    }

    let mean = deltas.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        deltas.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let std_dev = variance.sqrt();

    // 95% CI using z = 1.96 (normal approximation).
    let margin = 1.96 * std_dev / (n as f64).sqrt();

    BatchResult {
        mean_coherence_delta: mean,
        std_coherence_delta: std_dev,
        ci_95_lower: mean - margin,
        ci_95_upper: mean + margin,
        n_samples: n,
        pass_rate: passes as f64 / n as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_empty() {
        let result = evaluate_batch(&[], &[], 0.9);
        assert_eq!(result.n_samples, 0);
        assert_eq!(result.pass_rate, 0.0);
    }

    #[test]
    fn batch_identical_outputs() {
        let baselines = vec![vec![1.0, 2.0, 3.0]; 10];
        let gated = baselines.clone();
        let result = evaluate_batch(&baselines, &gated, 0.9);
        assert_eq!(result.n_samples, 10);
        assert!((result.mean_coherence_delta).abs() < 1e-10);
        assert!((result.std_coherence_delta).abs() < 1e-10);
        assert!((result.pass_rate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn batch_ci_contains_mean() {
        let baselines = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 3.0],
        ];
        let gated = vec![
            vec![1.1, 0.1],
            vec![0.1, 1.1],
            vec![1.2, 0.9],
            vec![2.1, 2.9],
        ];
        let result = evaluate_batch(&baselines, &gated, 0.9);
        assert_eq!(result.n_samples, 4);
        assert!(result.ci_95_lower <= result.mean_coherence_delta);
        assert!(result.ci_95_upper >= result.mean_coherence_delta);
    }

    #[test]
    fn batch_single_sample() {
        let baselines = vec![vec![1.0, 2.0]];
        let gated = vec![vec![1.0, 2.0]];
        let result = evaluate_batch(&baselines, &gated, 0.5);
        assert_eq!(result.n_samples, 1);
        assert!((result.std_coherence_delta).abs() < 1e-10);
        assert!((result.pass_rate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn batch_pass_rate_partial() {
        // Two samples: one passes (similar), one fails (orthogonal).
        let baselines = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let gated = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = evaluate_batch(&baselines, &gated, 0.5);
        assert_eq!(result.n_samples, 2);
        assert!((result.pass_rate - 0.5).abs() < 1e-10);
    }

    #[test]
    fn batch_result_serializable() {
        let result = BatchResult {
            mean_coherence_delta: -0.05,
            std_coherence_delta: 0.02,
            ci_95_lower: -0.07,
            ci_95_upper: -0.03,
            n_samples: 100,
            pass_rate: 0.95,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deser: BatchResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.n_samples, 100);
        assert!((deser.pass_rate - 0.95).abs() < 1e-10);
    }
}
