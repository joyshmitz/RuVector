use serde::{Deserialize, Serialize};

/// Configuration for the min-cut gating attention operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinCutConfig {
    /// Regularization weight balancing cut cost vs. edge retention.
    pub lambda: f32,
    /// Hysteresis window: edges must be consistently gated for `tau` steps before flipping.
    pub tau: usize,
    /// Convergence tolerance for iterative refinement.
    pub eps: f32,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Whether to emit witness JSONL entries for determinism verification.
    pub witness_enabled: bool,
}

impl Default for MinCutConfig {
    fn default() -> Self {
        Self {
            lambda: 0.5,
            tau: 2,
            eps: 0.01,
            seed: 42,
            witness_enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = MinCutConfig::default();
        assert!((cfg.lambda - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.tau, 2);
        assert!((cfg.eps - 0.01).abs() < f32::EPSILON);
        assert_eq!(cfg.seed, 42);
        assert!(cfg.witness_enabled);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let cfg = MinCutConfig {
            lambda: 0.3,
            tau: 5,
            eps: 0.001,
            seed: 99,
            witness_enabled: false,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let restored: MinCutConfig = serde_json::from_str(&json).unwrap();
        assert!((restored.lambda - 0.3).abs() < f32::EPSILON);
        assert_eq!(restored.tau, 5);
        assert!(!restored.witness_enabled);
    }
}
