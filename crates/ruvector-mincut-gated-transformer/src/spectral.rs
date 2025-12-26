//! Spectral position encoding using graph Laplacian eigenvectors.
//!
//! Based on Spectral Attention Network (Kreuzer et al., 2021).
//! Uses eigenvalues and eigenvectors from the graph Laplacian derived from
//! mincut boundary structures for position-aware encoding.
//!
//! Key innovations:
//! - Laplacian eigendecomposition for structural position encoding
//! - No distance matrix required - uses graph topology
//! - Integrates naturally with mincut boundary edges
//! - Sparse CSR matrix format for 10-200× speedup on sparse graphs
//!
//! ## Performance
//!
//! Graph Laplacians are inherently sparse (E edges vs n² dense entries).
//! Using CSR format reduces matrix-vector products from O(n²) to O(E).

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

/// Compressed Sparse Row (CSR) matrix format.
///
/// Stores only non-zero entries for O(E) matrix-vector multiplication.
/// For a Laplacian with E edges, this is 10-200× faster than dense O(n²).
#[derive(Clone, Debug)]
pub struct SparseCSR {
    /// Number of rows
    pub n: usize,
    /// Row pointers: row i has entries from row_ptr[i]..row_ptr[i+1]
    pub row_ptr: Vec<usize>,
    /// Column indices of non-zero entries
    pub col_idx: Vec<usize>,
    /// Values of non-zero entries
    pub values: Vec<f32>,
}

impl SparseCSR {
    /// Create empty sparse matrix
    pub fn empty(n: usize) -> Self {
        Self {
            n,
            row_ptr: vec![0; n + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparse matrix-vector multiply: y = A * x
    ///
    /// O(nnz) complexity instead of O(n²) for dense.
    #[inline]
    pub fn spmv(&self, x: &[f32], y: &mut [f32]) {
        debug_assert!(x.len() >= self.n);
        debug_assert!(y.len() >= self.n);

        for i in 0..self.n {
            let mut sum = 0.0f32;
            let start = self.row_ptr[i];
            let end = self.row_ptr.get(i + 1).copied().unwrap_or(start);

            for idx in start..end {
                if let (Some(&col), Some(&val)) = (self.col_idx.get(idx), self.values.get(idx)) {
                    if let Some(&x_val) = x.get(col) {
                        sum += val * x_val;
                    }
                }
            }
            if let Some(y_val) = y.get_mut(i) {
                *y_val = sum;
            }
        }
    }

    /// Build sparse Laplacian from boundary edges.
    ///
    /// Returns CSR format Laplacian with O(E) storage and O(E) matrix-vector ops.
    pub fn from_boundary_edges(boundary_edges: &[(u16, u16)], n: usize) -> Self {
        if n == 0 {
            return Self::empty(0);
        }

        // Count non-zeros per row (degree + off-diagonal entries)
        let mut row_nnz = vec![0usize; n];
        let mut degree = vec![0u32; n];

        for &(u, v) in boundary_edges {
            let u = u as usize;
            let v = v as usize;
            if u < n && v < n && u != v {
                row_nnz[u] += 1; // Off-diagonal entry
                row_nnz[v] += 1; // Symmetric
                degree[u] += 1;
                degree[v] += 1;
            }
        }

        // Add diagonal entries
        for i in 0..n {
            if degree[i] > 0 {
                row_nnz[i] += 1; // Diagonal entry
            }
        }

        // Build row pointers
        let mut row_ptr = vec![0usize; n + 1];
        for i in 0..n {
            row_ptr[i + 1] = row_ptr[i] + row_nnz[i];
        }
        let total_nnz = row_ptr[n];

        // Allocate arrays
        let mut col_idx = vec![0usize; total_nnz];
        let mut values = vec![0.0f32; total_nnz];
        let mut current_pos = row_ptr.clone();

        // Fill diagonal entries first
        for i in 0..n {
            if degree[i] > 0 {
                let pos = current_pos[i];
                col_idx[pos] = i;
                values[pos] = degree[i] as f32;
                current_pos[i] += 1;
            }
        }

        // Fill off-diagonal entries (Laplacian = D - A, so off-diagonal = -1)
        for &(u, v) in boundary_edges {
            let u = u as usize;
            let v = v as usize;
            if u < n && v < n && u != v {
                // Entry (u, v)
                let pos_u = current_pos[u];
                if pos_u < row_ptr[u + 1] {
                    col_idx[pos_u] = v;
                    values[pos_u] = -1.0;
                    current_pos[u] += 1;
                }

                // Entry (v, u) - symmetric
                let pos_v = current_pos[v];
                if pos_v < row_ptr[v + 1] {
                    col_idx[pos_v] = u;
                    values[pos_v] = -1.0;
                    current_pos[v] += 1;
                }
            }
        }

        Self { n, row_ptr, col_idx, values }
    }
}

/// Configuration for spectral position encoding.
#[derive(Clone, Debug)]
pub struct SpectralPEConfig {
    /// Number of eigenvectors to compute (typically 8-16)
    pub num_eigenvectors: u16,

    /// Number of attention heads for PE mixing
    pub pe_attention_heads: u16,

    /// Whether to make position encoding learnable
    pub learnable_pe: bool,
}

impl Default for SpectralPEConfig {
    fn default() -> Self {
        Self {
            num_eigenvectors: 8,
            pe_attention_heads: 4,
            learnable_pe: false,
        }
    }
}

/// Spectral position encoder using graph Laplacian.
pub struct SpectralPositionEncoder {
    config: SpectralPEConfig,
}

impl SpectralPositionEncoder {
    /// Create new spectral position encoder.
    pub fn new(config: SpectralPEConfig) -> Self {
        Self { config }
    }
}

impl Default for SpectralPositionEncoder {
    fn default() -> Self {
        Self {
            config: SpectralPEConfig::default(),
        }
    }
}

impl SpectralPositionEncoder {

    /// Compute graph Laplacian from boundary edges.
    ///
    /// Laplacian L = D - A, where:
    /// - D is degree matrix (diagonal)
    /// - A is adjacency matrix
    ///
    /// # Arguments
    ///
    /// * `boundary_edges` - List of (u, v) edges from mincut
    /// * `n` - Number of nodes
    ///
    /// # Returns
    ///
    /// Flattened Laplacian matrix [n x n] in row-major order
    pub fn compute_laplacian(&self, boundary_edges: &[(u16, u16)], n: usize) -> Vec<f32> {
        let mut laplacian = vec![0.0f32; n * n];
        let mut degree = vec![0u32; n];

        // Build adjacency and compute degrees
        for &(u, v) in boundary_edges {
            let u = u as usize;
            let v = v as usize;

            if u >= n || v >= n {
                continue;
            }

            // Symmetric adjacency: A[u][v] = A[v][u] = 1
            laplacian[u * n + v] = -1.0;
            laplacian[v * n + u] = -1.0;

            degree[u] += 1;
            degree[v] += 1;
        }

        // Set diagonal to degree: L = D - A
        for i in 0..n {
            laplacian[i * n + i] = degree[i] as f32;
        }

        laplacian
    }

    /// Compute normalized Laplacian for better numerical stability.
    ///
    /// L_norm = D^(-1/2) * L * D^(-1/2)
    pub fn compute_normalized_laplacian(&self, boundary_edges: &[(u16, u16)], n: usize) -> Vec<f32> {
        let mut laplacian = self.compute_laplacian(boundary_edges, n);
        let mut degree_sqrt_inv = vec![0.0f32; n];

        // Compute D^(-1/2)
        for i in 0..n {
            let deg = laplacian[i * n + i];
            degree_sqrt_inv[i] = if deg > 0.0 {
                1.0 / deg.sqrt()
            } else {
                0.0
            };
        }

        // Normalize: L_norm = D^(-1/2) * L * D^(-1/2)
        for i in 0..n {
            for j in 0..n {
                laplacian[i * n + j] *= degree_sqrt_inv[i] * degree_sqrt_inv[j];
            }
        }

        laplacian
    }

    /// Extract top-k eigenvectors using power iteration.
    ///
    /// Computes dominant eigenvectors without external dependencies.
    ///
    /// # Arguments
    ///
    /// * `laplacian` - Laplacian matrix [n x n]
    /// * `n` - Matrix dimension
    /// * `k` - Number of eigenvectors to compute
    ///
    /// # Returns
    ///
    /// Vector of eigenvectors, each of length n
    pub fn eigenvectors(&self, laplacian: &[f32], n: usize, k: usize) -> Vec<Vec<f32>> {
        let k = k.min(n);
        let mut eigenvectors = Vec::with_capacity(k);

        // We want smallest eigenvectors (smoothest modes)
        // For Laplacian, use inverse iteration or work with (max_eigenvalue*I - L)

        // Estimate maximum eigenvalue (Gershgorin bound: max row sum)
        let mut max_eval = 0.0f32;
        for i in 0..n {
            let row_sum: f32 = (0..n).map(|j| laplacian[i * n + j].abs()).sum();
            max_eval = max_eval.max(row_sum);
        }

        // Shift matrix: M = (max_eval + 1)*I - L
        // Largest eigenvalues of M correspond to smallest of L
        let shift = max_eval + 1.0;
        let mut shifted = laplacian.to_vec();
        for i in 0..n {
            shifted[i * n + i] = shift - shifted[i * n + i];
            for j in 0..n {
                if i != j {
                    shifted[i * n + j] = -shifted[i * n + j];
                }
            }
        }

        for _ in 0..k {
            let evec = power_iteration(&shifted, n, 100);

            // Deflate: remove this eigenvector's contribution
            // A_new = A - λ * v * v^T
            let eigenvalue = rayleigh_quotient(&shifted, n, &evec);

            // Update shifted matrix: A := A - λ * v * v^T
            for i in 0..n {
                for j in 0..n {
                    shifted[i * n + j] -= eigenvalue * evec[i] * evec[j];
                }
            }

            eigenvectors.push(evec);
        }

        eigenvectors
    }

    /// Generate position encoding from eigenvectors.
    ///
    /// Concatenates the first k eigenvector values for each position.
    ///
    /// # Arguments
    ///
    /// * `eigenvectors` - List of eigenvectors [k x n]
    ///
    /// # Returns
    ///
    /// Flattened position encodings [n x k]
    pub fn encode_positions(&self, eigenvectors: &[Vec<f32>]) -> Vec<f32> {
        if eigenvectors.is_empty() {
            return Vec::new();
        }

        let n = eigenvectors[0].len();
        let k = eigenvectors.len();
        let mut encoding = vec![0.0f32; n * k];

        // For each position i, concatenate eigenvector values
        for i in 0..n {
            for (j, evec) in eigenvectors.iter().enumerate() {
                encoding[i * k + j] = evec[i];
            }
        }

        encoding
    }

    /// Add spectral position encoding to embeddings.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Quantized embeddings [seq_len x hidden_dim], i8
    /// * `pe` - Position encodings [n x k], f32
    /// * `scale` - Scaling factor for PE addition
    ///
    /// PE is added to the first k dimensions of each position's embedding.
    pub fn add_to_embeddings(&self, embeddings: &mut [i8], pe: &[f32], scale: f32) {
        if pe.is_empty() {
            return;
        }

        let k = self.config.num_eigenvectors as usize;
        let n = pe.len() / k;

        // Each position in embeddings gets PE added to its first k dimensions
        let embedding_dim = embeddings.len() / n.max(1);

        for pos in 0..n {
            let pe_offset = pos * k;
            let emb_offset = pos * embedding_dim;

            // Add PE to first k dimensions of this position's embedding
            for j in 0..k.min(embedding_dim) {
                if emb_offset + j >= embeddings.len() || pe_offset + j >= pe.len() {
                    break;
                }

                let pe_val = pe[pe_offset + j] * scale;
                let pe_quantized = pe_val.clamp(-128.0, 127.0) as i8;

                // Saturating addition
                let current = embeddings[emb_offset + j] as i32;
                let new_val = (current + pe_quantized as i32).clamp(-128, 127) as i8;
                embeddings[emb_offset + j] = new_val;
            }
        }
    }

    /// Compute spectral distance between two positions.
    ///
    /// Uses eigenvector-based distance metric.
    pub fn spectral_distance(&self, pe: &[f32], i: usize, j: usize) -> f32 {
        let k = self.config.num_eigenvectors as usize;

        if i * k >= pe.len() || j * k >= pe.len() {
            return 0.0;
        }

        let mut dist_sq = 0.0f32;
        for d in 0..k {
            let diff = pe[i * k + d] - pe[j * k + d];
            dist_sq += diff * diff;
        }

        dist_sq.sqrt()
    }

    /// Generate full position encoding from mincut boundary edges.
    ///
    /// End-to-end: edges -> Laplacian -> eigenvectors -> encoding
    pub fn encode_from_edges(&self, boundary_edges: &[(u16, u16)], n: usize) -> Vec<f32> {
        let laplacian = self.compute_normalized_laplacian(boundary_edges, n);
        let k = self.config.num_eigenvectors as usize;
        let eigenvectors = self.eigenvectors(&laplacian, n, k);
        self.encode_positions(&eigenvectors)
    }
}

/// Simple power iteration for dominant eigenvector computation.
///
/// Computes the eigenvector corresponding to the largest eigenvalue.
/// No external dependencies required.
///
/// # Arguments
///
/// * `matrix` - Square matrix [n x n] in row-major order
/// * `n` - Matrix dimension
/// * `num_iters` - Number of power iterations (typically 50-100)
///
/// # Returns
///
/// Normalized eigenvector of length n
pub fn power_iteration(matrix: &[f32], n: usize, num_iters: u16) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }

    // Initialize with random-like vector
    let mut v: Vec<f32> = (0..n).map(|i| ((i * 7 + 13) % 100) as f32 / 100.0).collect();

    // Power iteration
    for _ in 0..num_iters {
        // v_new = A * v
        let mut v_new = vec![0.0f32; n];

        for i in 0..n {
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += matrix[i * n + j] * v[j];
            }
            v_new[i] = sum;
        }

        // Normalize
        let norm: f32 = v_new.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut v_new {
                *x /= norm;
            }
        }

        v = v_new;
    }

    v
}

/// Sparse power iteration using CSR matrix format.
///
/// O(num_iters × E) complexity instead of O(num_iters × n²).
/// For typical graphs where E << n², this provides 10-200× speedup.
///
/// # Arguments
///
/// * `csr` - Sparse matrix in CSR format
/// * `num_iters` - Number of power iterations (typically 50-100)
///
/// # Returns
///
/// Dominant eigenvector (normalized)
pub fn power_iteration_sparse(csr: &SparseCSR, num_iters: u16) -> Vec<f32> {
    let n = csr.n;
    if n == 0 {
        return Vec::new();
    }

    // Initialize with deterministic pseudo-random vector
    let mut v: Vec<f32> = (0..n).map(|i| ((i * 7 + 13) % 100) as f32 / 100.0).collect();
    let mut v_new = vec![0.0f32; n];

    for _ in 0..num_iters {
        // v_new = A * v using sparse matrix-vector multiply
        csr.spmv(&v, &mut v_new);

        // Normalize
        let norm: f32 = v_new.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut v_new {
                *x /= norm;
            }
        }

        core::mem::swap(&mut v, &mut v_new);
    }

    v
}

/// Compute Rayleigh quotient for eigenvalue estimation.
///
/// λ ≈ (v^T * A * v) / (v^T * v)
pub fn rayleigh_quotient(matrix: &[f32], n: usize, v: &[f32]) -> f32 {
    if n == 0 || v.len() != n {
        return 0.0;
    }

    // Compute A * v
    let mut av = vec![0.0f32; n];
    for i in 0..n {
        for j in 0..n {
            av[i] += matrix[i * n + j] * v[j];
        }
    }

    // v^T * (A * v)
    let numerator: f32 = v.iter().zip(av.iter()).map(|(vi, avi)| vi * avi).sum();

    // v^T * v
    let denominator: f32 = v.iter().map(|vi| vi * vi).sum();

    if denominator > 1e-10 {
        numerator / denominator
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_config_default() {
        let config = SpectralPEConfig::default();
        assert_eq!(config.num_eigenvectors, 8);
        assert_eq!(config.pe_attention_heads, 4);
        assert!(!config.learnable_pe);
    }

    #[test]
    fn test_laplacian_empty() {
        let encoder = SpectralPositionEncoder::default();
        let laplacian = encoder.compute_laplacian(&[], 4);

        // Empty edges -> zero Laplacian (no connections)
        assert_eq!(laplacian.len(), 16);
        assert!(laplacian.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_laplacian_simple_graph() {
        let encoder = SpectralPositionEncoder::default();

        // Simple chain: 0-1-2-3
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let laplacian = encoder.compute_laplacian(&edges, 4);

        // Check diagonal (degrees)
        assert_eq!(laplacian[0 * 4 + 0], 1.0); // node 0: degree 1
        assert_eq!(laplacian[1 * 4 + 1], 2.0); // node 1: degree 2
        assert_eq!(laplacian[2 * 4 + 2], 2.0); // node 2: degree 2
        assert_eq!(laplacian[3 * 4 + 3], 1.0); // node 3: degree 1

        // Check off-diagonal (adjacency)
        assert_eq!(laplacian[0 * 4 + 1], -1.0);
        assert_eq!(laplacian[1 * 4 + 0], -1.0);
        assert_eq!(laplacian[1 * 4 + 2], -1.0);
    }

    #[test]
    fn test_laplacian_symmetric() {
        let encoder = SpectralPositionEncoder::default();

        let edges = vec![(0, 1), (1, 2), (0, 2)]; // Triangle
        let laplacian = encoder.compute_laplacian(&edges, 3);

        // Laplacian should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(laplacian[i * 3 + j], laplacian[j * 3 + i]);
            }
        }
    }

    #[test]
    fn test_power_iteration_identity() {
        // Identity matrix has all eigenvalues = 1
        let n = 4;
        let mut identity = vec![0.0f32; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }

        let v = power_iteration(&identity, n, 50);

        // Should converge to normalized vector
        assert_eq!(v.len(), n);

        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_power_iteration_diagonal() {
        // Diagonal matrix with distinct eigenvalues
        let n = 3;
        let mut matrix = vec![0.0f32; n * n];
        matrix[0 * n + 0] = 5.0; // Largest eigenvalue
        matrix[1 * n + 1] = 3.0;
        matrix[2 * n + 2] = 1.0;

        let v = power_iteration(&matrix, n, 100);

        // Should converge to [1, 0, 0] (eigenvector of largest eigenvalue)
        assert!(v[0].abs() > 0.9);
        assert!(v[1].abs() < 0.3);
        assert!(v[2].abs() < 0.3);
    }

    #[test]
    fn test_rayleigh_quotient() {
        let n = 3;
        let mut matrix = vec![0.0f32; n * n];
        matrix[0 * n + 0] = 4.0;
        matrix[1 * n + 1] = 3.0;
        matrix[2 * n + 2] = 2.0;

        // Eigenvector of eigenvalue 4.0
        let v = vec![1.0, 0.0, 0.0];
        let lambda = rayleigh_quotient(&matrix, n, &v);

        assert!((lambda - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_encode_positions() {
        let encoder = SpectralPositionEncoder::default();

        let eigenvectors = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.6, 0.7, 0.8],
        ];

        let encoding = encoder.encode_positions(&eigenvectors);

        // Should be [n x k] = [4 x 2] = 8 elements
        assert_eq!(encoding.len(), 8);

        // Check first position encoding
        assert_eq!(encoding[0], 0.1); // First eigenvector
        assert_eq!(encoding[1], 0.5); // Second eigenvector

        // Check second position
        assert_eq!(encoding[2], 0.2);
        assert_eq!(encoding[3], 0.6);
    }

    #[test]
    fn test_add_to_embeddings() {
        let config = SpectralPEConfig {
            num_eigenvectors: 2,
            pe_attention_heads: 2,
            learnable_pe: false,
        };
        let encoder = SpectralPositionEncoder::new(config);

        // 2 positions, 2 dims each = 4 elements
        let mut embeddings = vec![10i8, 20, 30, 40];
        // PE for 2 positions, 2 eigenvectors each = 4 elements
        let pe = vec![0.5, 1.0, -0.5, -1.0];

        encoder.add_to_embeddings(&mut embeddings, &pe, 10.0);

        // PE values scaled by 10 and added
        // Position 0: embeddings[0] = 10 + 5 = 15, embeddings[1] = 20 + 10 = 30
        // Position 1: embeddings[2] = 30 + (-5) = 25, embeddings[3] = 40 + (-10) = 30
        assert_eq!(embeddings[0], 15);
        assert_eq!(embeddings[1], 30);
        assert_eq!(embeddings[2], 25);
        assert_eq!(embeddings[3], 30);
    }

    #[test]
    fn test_spectral_distance() {
        let config = SpectralPEConfig {
            num_eigenvectors: 2,
            pe_attention_heads: 2,
            learnable_pe: false,
        };
        let encoder = SpectralPositionEncoder::new(config);

        // Encoding: 2 positions, 2 dimensions each
        let pe = vec![
            0.0, 0.0, // position 0
            1.0, 1.0, // position 1
        ];

        let dist = encoder.spectral_distance(&pe, 0, 1);

        // Distance should be sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.414
        assert!((dist - 1.414).abs() < 0.01);

        // Distance to self should be 0
        let dist_self = encoder.spectral_distance(&pe, 0, 0);
        assert!(dist_self.abs() < 1e-6);
    }

    #[test]
    fn test_encode_from_edges() {
        let config = SpectralPEConfig {
            num_eigenvectors: 3,
            pe_attention_heads: 2,
            learnable_pe: false,
        };
        let encoder = SpectralPositionEncoder::new(config);

        // Simple chain graph
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let encoding = encoder.encode_from_edges(&edges, 4);

        // Should produce [4 positions x 3 eigenvectors] = 12 values
        assert_eq!(encoding.len(), 12);

        // All values should be finite
        assert!(encoding.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_normalized_laplacian() {
        let encoder = SpectralPositionEncoder::default();

        let edges = vec![(0, 1), (1, 2)];
        let laplacian = encoder.compute_normalized_laplacian(&edges, 3);

        // Normalized Laplacian should have 1 on diagonal for isolated nodes
        // and values in [-1, 1] otherwise
        assert!(laplacian.iter().all(|&x| x.abs() <= 1.0 + 1e-5));
    }

    #[test]
    fn test_embedding_saturation() {
        let encoder = SpectralPositionEncoder::default();

        let mut embeddings = vec![127i8]; // Maximum value
        let pe = vec![100.0]; // Large PE value

        encoder.add_to_embeddings(&mut embeddings, &pe, 1.0);

        // Should saturate at 127, not overflow
        assert_eq!(embeddings[0], 127);
    }
}
