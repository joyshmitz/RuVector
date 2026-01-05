//! Sparse Feed-Forward Network implementation.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use tracing::{debug, trace};

use crate::backend::{get_backend, Backend};
use crate::config::ActivationType;
use crate::error::{InferenceError, Result};

/// Sparse Feed-Forward Network computation.
///
/// This implements a two-layer FFN that can compute using only a subset of neurons:
/// - W1: [hidden_dim, input_dim] - first projection (row-major for neuron access)
/// - W2: [output_dim, hidden_dim] - second projection (column-major for accumulation)
/// - Activation function applied between layers
///
/// The sparse forward pass:
/// 1. Sparse first layer: only compute active neurons
/// 2. Apply activation function
/// 3. Sparse second layer: accumulate only active neuron contributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseFfn {
    /// W1: [hidden_dim, input_dim] - first projection.
    /// Row-major layout for efficient neuron access.
    w1: Array2<f32>,

    /// W2: [output_dim, hidden_dim] - second projection.
    /// Column-major layout for efficient accumulation.
    #[serde(with = "w2_serde")]
    w2: Array2<f32>,

    /// Bias for first layer.
    b1: Array1<f32>,

    /// Bias for second layer.
    b2: Array1<f32>,

    /// Activation function type.
    activation: ActivationType,
}

// Custom serialization for w2 to handle layout
mod w2_serde {
    use super::*;
    use ndarray::Array2;

    pub fn serialize<S>(w2: &Array2<f32>, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Convert to standard layout for serialization
        let standard = w2.as_standard_layout();
        standard.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<Array2<f32>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let standard = Array2::<f32>::deserialize(deserializer)?;
        Ok(standard)
    }
}

impl SparseFfn {
    /// Create a new sparse FFN with given dimensions.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        activation: ActivationType,
    ) -> Result<Self> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Initialize with small random values
        let w1 = Array2::from_shape_fn((hidden_dim, input_dim), |_| {
            rng.gen::<f32>() * 0.01
        });

        let w2 = Array2::from_shape_fn((output_dim, hidden_dim), |_| {
            rng.gen::<f32>() * 0.01
        });

        let b1 = Array1::zeros(hidden_dim);
        let b2 = Array1::zeros(output_dim);

        Ok(Self {
            w1,
            w2,
            b1,
            b2,
            activation,
        })
    }

    /// Create from existing weights.
    pub fn from_weights(
        w1: Array2<f32>,
        w2: Array2<f32>,
        b1: Array1<f32>,
        b2: Array1<f32>,
        activation: ActivationType,
    ) -> Result<Self> {
        let (hidden_dim, input_dim) = w1.dim();
        let (output_dim, w2_hidden) = w2.dim();

        if hidden_dim != w2_hidden {
            return Err(InferenceError::Failed(
                format!("Hidden dimension mismatch: W1 has {}, W2 has {}",
                    hidden_dim, w2_hidden)
            ).into());
        }

        if b1.len() != hidden_dim {
            return Err(InferenceError::Failed(
                format!("b1 dimension mismatch: expected {}, got {}",
                    hidden_dim, b1.len())
            ).into());
        }

        if b2.len() != output_dim {
            return Err(InferenceError::Failed(
                format!("b2 dimension mismatch: expected {}, got {}",
                    output_dim, b2.len())
            ).into());
        }

        Ok(Self {
            w1,
            w2,
            b1,
            b2,
            activation,
        })
    }

    /// Get input dimension.
    pub fn input_dim(&self) -> usize {
        self.w1.ncols()
    }

    /// Get hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.w1.nrows()
    }

    /// Get output dimension.
    pub fn output_dim(&self) -> usize {
        self.w2.nrows()
    }

    /// Compute FFN using only active neurons (sparse computation).
    ///
    /// This is the main optimization: only compute activations for predicted neurons.
    pub fn forward_sparse(&self, input: &[f32], active_neurons: &[usize]) -> Result<Vec<f32>> {
        if input.len() != self.input_dim() {
            return Err(InferenceError::InputDimensionMismatch {
                expected: self.input_dim(),
                actual: input.len(),
            }.into());
        }

        if active_neurons.is_empty() {
            return Err(InferenceError::NoActiveNeurons.into());
        }

        trace!("Sparse forward: {} active neurons ({:.1}% sparsity)",
            active_neurons.len(),
            100.0 * (1.0 - active_neurons.len() as f32 / self.hidden_dim() as f32)
        );

        let backend = get_backend();

        // 1. Sparse first layer: only compute active neurons
        let mut hidden = Vec::with_capacity(active_neurons.len());
        for &neuron_idx in active_neurons {
            if neuron_idx >= self.hidden_dim() {
                return Err(InferenceError::Failed(
                    format!("Invalid neuron index: {}", neuron_idx)
                ).into());
            }

            let row = self.w1.row(neuron_idx);
            let dot = backend.dot_product(row.as_slice().unwrap(), input);
            hidden.push(dot + self.b1[neuron_idx]);
        }

        // 2. Apply activation function
        backend.activation(&mut hidden, self.activation);

        // 3. Sparse second layer: accumulate only active neuron contributions
        let mut output = self.b2.to_vec();

        for (i, &neuron_idx) in active_neurons.iter().enumerate() {
            let col = self.w2.column(neuron_idx);
            let h_val = hidden[i];

            for (j, &w) in col.iter().enumerate() {
                output[j] += h_val * w;
            }
        }

        Ok(output)
    }

    /// Compute FFN using all neurons (dense computation).
    ///
    /// This is the baseline for comparison and correctness checking.
    pub fn forward_dense(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input_dim() {
            return Err(InferenceError::InputDimensionMismatch {
                expected: self.input_dim(),
                actual: input.len(),
            }.into());
        }

        let backend = get_backend();
        let input_arr = Array1::from_vec(input.to_vec());

        // 1. First layer: hidden = activation(W1 · input + b1)
        let mut hidden = self.w1.dot(&input_arr) + &self.b1;
        backend.activation(hidden.as_slice_mut().unwrap(), self.activation);

        // 2. Second layer: output = W2 · hidden + b2
        let output = self.w2.dot(&hidden) + &self.b2;

        Ok(output.to_vec())
    }

    /// Compute both sparse and dense, returning the difference for validation.
    #[cfg(test)]
    pub fn validate_sparse(&self, input: &[f32], active_neurons: &[usize]) -> Result<f32> {
        let sparse_output = self.forward_sparse(input, active_neurons)?;
        let dense_output = self.forward_dense(input)?;

        // Compute mean absolute error
        let mae: f32 = sparse_output.iter()
            .zip(dense_output.iter())
            .map(|(s, d)| (s - d).abs())
            .sum::<f32>() / sparse_output.len() as f32;

        Ok(mae)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffn_creation() {
        let ffn = SparseFfn::new(128, 512, 128, ActivationType::Gelu).unwrap();

        assert_eq!(ffn.input_dim(), 128);
        assert_eq!(ffn.hidden_dim(), 512);
        assert_eq!(ffn.output_dim(), 128);
    }

    #[test]
    fn test_dense_forward() {
        let ffn = SparseFfn::new(64, 256, 64, ActivationType::Relu).unwrap();
        let input = vec![0.1; 64];

        let output = ffn.forward_dense(&input).unwrap();
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_sparse_forward() {
        let ffn = SparseFfn::new(64, 256, 64, ActivationType::Relu).unwrap();
        let input = vec![0.1; 64];
        let active_neurons: Vec<usize> = (0..64).collect(); // First 64 neurons

        let output = ffn.forward_sparse(&input, &active_neurons).unwrap();
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_sparse_vs_dense() {
        let ffn = SparseFfn::new(32, 128, 32, ActivationType::Relu).unwrap();
        let input = vec![0.5; 32];

        // Use all neurons - should match dense computation
        let all_neurons: Vec<usize> = (0..128).collect();
        let mae = ffn.validate_sparse(&input, &all_neurons).unwrap();

        // Should be very close (allowing for floating point precision)
        assert!(mae < 1e-5, "MAE too large: {}", mae);
    }

    #[test]
    fn test_empty_active_neurons() {
        let ffn = SparseFfn::new(32, 128, 32, ActivationType::Relu).unwrap();
        let input = vec![0.1; 32];
        let empty: Vec<usize> = vec![];

        let result = ffn.forward_sparse(&input, &empty);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_neuron_index() {
        let ffn = SparseFfn::new(32, 128, 32, ActivationType::Relu).unwrap();
        let input = vec![0.1; 32];
        let invalid = vec![200]; // Out of bounds

        let result = ffn.forward_sparse(&input, &invalid);
        assert!(result.is_err());
    }
}
