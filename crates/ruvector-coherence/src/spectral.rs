//! Spectral Coherence Score for graph index health monitoring.
//!
//! Provides a composite metric measuring structural health of graph indices
//! using spectral graph theory properties. All spectral computation is
//! self-contained with no external solver dependencies.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CsrMatrixView – lightweight sparse matrix
// ---------------------------------------------------------------------------

/// Compressed Sparse Row matrix for Laplacian representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrMatrixView {
    pub row_ptr: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl CsrMatrixView {
    /// Create a CSR matrix from raw components.
    pub fn new(
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
        rows: usize,
        cols: usize,
    ) -> Self {
        Self { row_ptr, col_indices, values, rows, cols }
    }

    /// Build a CSR matrix from an edge list. Edges are `(u, v, weight)`.
    /// The resulting matrix is the adjacency matrix (symmetric).
    pub fn from_edges(n: usize, edges: &[(usize, usize, f64)]) -> Self {
        // Collect (row, col, val) entries sorted by (row, col).
        let mut entries: Vec<(usize, usize, f64)> = Vec::with_capacity(edges.len() * 2);
        for &(u, v, w) in edges {
            entries.push((u, v, w));
            if u != v {
                entries.push((v, u, w));
            }
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut row_ptr = vec![0usize; n + 1];
        let mut col_indices = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());

        for &(r, c, v) in &entries {
            row_ptr[r + 1] += 1;
            col_indices.push(c);
            values.push(v);
        }
        for i in 0..n {
            row_ptr[i + 1] += row_ptr[i];
        }

        Self { row_ptr, col_indices, values, rows: n, cols: n }
    }

    /// Sparse matrix-vector product: y = A * x.
    pub fn spmv(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.rows];
        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let mut s = 0.0;
            for idx in start..end {
                s += self.values[idx] * x[self.col_indices[idx]];
            }
            y[i] = s;
        }
        y
    }

    /// Build the graph Laplacian L = D - A from an edge list.
    pub fn build_laplacian(n: usize, edges: &[(usize, usize, f64)]) -> Self {
        // Accumulate degree for each vertex.
        let mut degree = vec![0.0_f64; n];
        let mut sym_edges: Vec<(usize, usize, f64)> = Vec::with_capacity(edges.len() * 2);
        for &(u, v, w) in edges {
            degree[u] += w;
            if u != v {
                degree[v] += w;
                sym_edges.push((u, v, -w));
                sym_edges.push((v, u, -w));
            }
        }
        // Add diagonal entries.
        for i in 0..n {
            sym_edges.push((i, i, degree[i]));
        }
        sym_edges.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut row_ptr = vec![0usize; n + 1];
        let mut col_indices = Vec::with_capacity(sym_edges.len());
        let mut values = Vec::with_capacity(sym_edges.len());

        for &(r, c, v) in &sym_edges {
            row_ptr[r + 1] += 1;
            col_indices.push(c);
            values.push(v);
        }
        for i in 0..n {
            row_ptr[i + 1] += row_ptr[i];
        }

        Self { row_ptr, col_indices, values, rows: n, cols: n }
    }
}

// ---------------------------------------------------------------------------
// SpectralConfig
// ---------------------------------------------------------------------------

/// Configuration for spectral coherence computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralConfig {
    /// Weight for Fiedler value component (default 0.3).
    pub alpha: f64,
    /// Weight for spectral gap component (default 0.3).
    pub beta: f64,
    /// Weight for effective resistance component (default 0.2).
    pub gamma: f64,
    /// Weight for degree regularity component (default 0.2).
    pub delta: f64,
    /// Maximum iterations for power iteration (default 50).
    pub max_iterations: usize,
    /// Convergence tolerance (default 1e-6).
    pub tolerance: f64,
    /// Number of edge updates before a full recompute (default 100).
    pub refresh_threshold: usize,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            beta: 0.3,
            gamma: 0.2,
            delta: 0.2,
            max_iterations: 50,
            tolerance: 1e-6,
            refresh_threshold: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// SpectralCoherenceScore
// ---------------------------------------------------------------------------

/// Composite spectral coherence score with individual components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralCoherenceScore {
    /// Normalized Fiedler value in [0, 1].
    pub fiedler: f64,
    /// Spectral gap ratio in [0, 1].
    pub spectral_gap: f64,
    /// Effective resistance score in [0, 1].
    pub effective_resistance: f64,
    /// Degree regularity score in [0, 1].
    pub degree_regularity: f64,
    /// Weighted composite SCS in [0, 1].
    pub composite: f64,
}

// ---------------------------------------------------------------------------
// Spectral computation functions
// ---------------------------------------------------------------------------

/// Dot product of two slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a slice.
fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Solve L * x = b using conjugate gradient (L must be SPD on the subspace
/// orthogonal to the all-ones vector). We deflate the null space.
fn cg_solve(laplacian: &CsrMatrixView, b: &[f64], max_iter: usize, tol: f64) -> Vec<f64> {
    let n = laplacian.rows;
    let inv_n = 1.0 / n as f64;

    // Deflate b: remove component along all-ones vector.
    let b_mean: f64 = b.iter().sum::<f64>() * inv_n;
    let b_def: Vec<f64> = b.iter().map(|v| v - b_mean).collect();

    let mut x = vec![0.0; n];
    let mut r = b_def.clone();
    let mut p = r.clone();
    let mut rs_old = dot(&r, &r);

    if rs_old < tol * tol {
        return x;
    }

    for _ in 0..max_iter {
        let mut ap = laplacian.spmv(&p);
        // Deflate ap.
        let ap_mean: f64 = ap.iter().sum::<f64>() * inv_n;
        for v in ap.iter_mut() {
            *v -= ap_mean;
        }

        let pap = dot(&p, &ap);
        if pap.abs() < 1e-30 {
            break;
        }
        let alpha = rs_old / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new = dot(&r, &r);
        if rs_new.sqrt() < tol {
            break;
        }
        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }
    x
}

/// Estimate the Fiedler value (second smallest eigenvalue of the Laplacian)
/// and its eigenvector using inverse iteration with deflation.
///
/// Returns `(eigenvalue, eigenvector)`.
pub fn estimate_fiedler(
    laplacian: &CsrMatrixView,
    max_iter: usize,
    tol: f64,
) -> (f64, Vec<f64>) {
    let n = laplacian.rows;
    if n <= 1 {
        return (0.0, vec![0.0; n]);
    }

    let inv_sqrt_n = 1.0 / (n as f64).sqrt();

    // Start with a vector that is orthogonal to all-ones.
    let mut v: Vec<f64> = (0..n).map(|i| (i as f64) - (n as f64 - 1.0) / 2.0).collect();
    let v_norm = norm(&v);
    if v_norm > 1e-30 {
        for x in v.iter_mut() {
            *x /= v_norm;
        }
    }

    // Remove component along all-ones.
    let proj: f64 = v.iter().sum::<f64>() * inv_sqrt_n;
    for x in v.iter_mut() {
        *x -= proj * inv_sqrt_n;
    }
    let v_norm = norm(&v);
    if v_norm > 1e-30 {
        for x in v.iter_mut() {
            *x /= v_norm;
        }
    }

    let mut eigenvalue = 0.0;

    for _ in 0..max_iter {
        // Inverse iteration: solve L * w = v using CG.
        let mut w = cg_solve(laplacian, &v, max_iter * 2, tol * 0.1);

        // Deflate: remove component along all-ones.
        let w_proj: f64 = w.iter().sum::<f64>() * inv_sqrt_n;
        for x in w.iter_mut() {
            *x -= w_proj * inv_sqrt_n;
        }

        let w_norm = norm(&w);
        if w_norm < 1e-30 {
            break;
        }
        for x in w.iter_mut() {
            *x /= w_norm;
        }

        // Rayleigh quotient: eigenvalue = v^T L v.
        let lv = laplacian.spmv(&w);
        eigenvalue = dot(&w, &lv);

        // Check convergence: ||Lw - eigenvalue * w||.
        let residual: f64 = lv
            .iter()
            .zip(w.iter())
            .map(|(lvi, wi)| (lvi - eigenvalue * wi).powi(2))
            .sum::<f64>()
            .sqrt();

        v = w;

        if residual < tol {
            break;
        }
    }

    (eigenvalue.max(0.0), v)
}

/// Estimate the largest eigenvalue of the Laplacian via power iteration.
pub fn estimate_largest_eigenvalue(laplacian: &CsrMatrixView, max_iter: usize) -> f64 {
    let n = laplacian.rows;
    if n == 0 {
        return 0.0;
    }
    let mut v: Vec<f64> = vec![1.0 / (n as f64).sqrt(); n];
    let mut eigenvalue = 0.0;

    for _ in 0..max_iter {
        let w = laplacian.spmv(&v);
        let w_norm = norm(&w);
        if w_norm < 1e-30 {
            return 0.0;
        }
        eigenvalue = dot(&v, &w);
        for (vi, wi) in v.iter_mut().zip(w.iter()) {
            *vi = wi / w_norm;
        }
    }
    eigenvalue.max(0.0)
}

/// Compute spectral gap ratio: fiedler / largest.
pub fn estimate_spectral_gap(fiedler: f64, largest: f64) -> f64 {
    if largest < 1e-30 {
        return 0.0;
    }
    (fiedler / largest).clamp(0.0, 1.0)
}

/// Compute degree regularity: 1 - (std_dev / mean) of vertex degrees.
/// Returns 1.0 for perfectly regular graphs, lower for irregular.
pub fn compute_degree_regularity(laplacian: &CsrMatrixView) -> f64 {
    let n = laplacian.rows;
    if n == 0 {
        return 1.0;
    }

    // Extract degrees from the diagonal of the Laplacian.
    let mut degrees = Vec::with_capacity(n);
    for i in 0..n {
        let start = laplacian.row_ptr[i];
        let end = laplacian.row_ptr[i + 1];
        let mut diag = 0.0;
        for idx in start..end {
            if laplacian.col_indices[idx] == i {
                diag = laplacian.values[idx];
                break;
            }
        }
        degrees.push(diag);
    }

    let mean = degrees.iter().sum::<f64>() / n as f64;
    if mean < 1e-30 {
        return 1.0;
    }
    let variance = degrees.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    (1.0 - std_dev / mean).clamp(0.0, 1.0)
}

/// Estimate average effective resistance by sampling vertex pairs
/// deterministically (every k-th pair) and solving L * x = e_u - e_v via CG.
pub fn estimate_effective_resistance_sampled(
    laplacian: &CsrMatrixView,
    n_samples: usize,
) -> f64 {
    let n = laplacian.rows;
    if n < 2 {
        return 0.0;
    }

    let total_pairs = n * (n - 1) / 2;
    let step = if total_pairs <= n_samples { 1 } else { total_pairs / n_samples };
    let max_samples = n_samples.min(total_pairs);

    let mut total_resistance = 0.0;
    let mut sampled = 0usize;
    let mut pair_idx = 0usize;

    'outer: for u in 0..n {
        for v in (u + 1)..n {
            if pair_idx % step == 0 {
                // Build RHS: e_u - e_v.
                let mut rhs = vec![0.0; n];
                rhs[u] = 1.0;
                rhs[v] = -1.0;

                let x = cg_solve(laplacian, &rhs, 100, 1e-8);
                let resistance = (x[u] - x[v]).abs();
                total_resistance += resistance;
                sampled += 1;

                if sampled >= max_samples {
                    break 'outer;
                }
            }
            pair_idx += 1;
        }
    }

    if sampled == 0 {
        return 0.0;
    }
    total_resistance / sampled as f64
}

// ---------------------------------------------------------------------------
// SpectralTracker – incremental maintenance
// ---------------------------------------------------------------------------

/// Tracks spectral coherence incrementally, recomputing fully when needed.
pub struct SpectralTracker {
    config: SpectralConfig,
    fiedler_estimate: f64,
    gap_estimate: f64,
    resistance_estimate: f64,
    regularity: f64,
    updates_since_refresh: usize,
    /// Cached Fiedler vector for perturbation-based updates.
    fiedler_vector: Option<Vec<f64>>,
}

impl SpectralTracker {
    /// Create a new tracker with the given configuration.
    pub fn new(config: SpectralConfig) -> Self {
        Self {
            config,
            fiedler_estimate: 0.0,
            gap_estimate: 0.0,
            resistance_estimate: 0.0,
            regularity: 1.0,
            updates_since_refresh: 0,
            fiedler_vector: None,
        }
    }

    /// Full spectral computation from a Laplacian.
    pub fn compute(&mut self, laplacian: &CsrMatrixView) -> SpectralCoherenceScore {
        self.full_recompute(laplacian);
        self.build_score()
    }

    /// Incremental update after an edge weight change between vertices u and v.
    /// Uses first-order eigenvalue perturbation: delta_lambda ~= v^T (delta_L) v.
    pub fn update_edge(
        &mut self,
        laplacian: &CsrMatrixView,
        u: usize,
        v: usize,
        weight_delta: f64,
    ) {
        self.updates_since_refresh += 1;

        if self.needs_refresh() || self.fiedler_vector.is_none() {
            self.full_recompute(laplacian);
            return;
        }

        // First-order perturbation of the Fiedler value.
        if let Some(ref fv) = self.fiedler_vector {
            if u < fv.len() && v < fv.len() {
                let diff = fv[u] - fv[v];
                let delta_lambda = weight_delta * diff * diff;
                self.fiedler_estimate = (self.fiedler_estimate + delta_lambda).max(0.0);

                // Update spectral gap estimate.
                let largest = estimate_largest_eigenvalue(laplacian, self.config.max_iterations);
                self.gap_estimate = estimate_spectral_gap(self.fiedler_estimate, largest);
            }
        }

        // Update degree regularity (cheap to recompute).
        self.regularity = compute_degree_regularity(laplacian);
    }

    /// Get the current composite score.
    pub fn score(&self) -> f64 {
        self.build_score().composite
    }

    /// Force a full recomputation from the Laplacian.
    pub fn full_recompute(&mut self, laplacian: &CsrMatrixView) {
        let (fiedler_raw, fiedler_vec) = estimate_fiedler(
            laplacian,
            self.config.max_iterations,
            self.config.tolerance,
        );
        let largest = estimate_largest_eigenvalue(laplacian, self.config.max_iterations);
        let n = laplacian.rows;

        // Normalize fiedler to [0, 1]. For a graph on n vertices the max
        // Fiedler value of L is n (complete graph), so divide by n.
        let fiedler_norm = if n > 0 {
            (fiedler_raw / n as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let gap = estimate_spectral_gap(fiedler_raw, largest);
        let resistance_raw =
            estimate_effective_resistance_sampled(laplacian, 50.min(n * (n - 1) / 2));
        // Normalize resistance: lower is better. Use 1 / (1 + R) mapping.
        let resistance_score = 1.0 / (1.0 + resistance_raw);
        let regularity = compute_degree_regularity(laplacian);

        self.fiedler_estimate = fiedler_norm;
        self.gap_estimate = gap;
        self.resistance_estimate = resistance_score;
        self.regularity = regularity;
        self.fiedler_vector = Some(fiedler_vec);
        self.updates_since_refresh = 0;
    }

    /// Whether the tracker has accumulated enough incremental updates
    /// to warrant a full recomputation.
    pub fn needs_refresh(&self) -> bool {
        self.updates_since_refresh >= self.config.refresh_threshold
    }

    fn build_score(&self) -> SpectralCoherenceScore {
        let composite = self.config.alpha * self.fiedler_estimate
            + self.config.beta * self.gap_estimate
            + self.config.gamma * self.resistance_estimate
            + self.config.delta * self.regularity;
        SpectralCoherenceScore {
            fiedler: self.fiedler_estimate,
            spectral_gap: self.gap_estimate,
            effective_resistance: self.resistance_estimate,
            degree_regularity: self.regularity,
            composite: composite.clamp(0.0, 1.0),
        }
    }
}

// ---------------------------------------------------------------------------
// HealthAlert & HnswHealthMonitor
// ---------------------------------------------------------------------------

/// Alert types for graph index health degradation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthAlert {
    /// The Fiedler value is dangerously low, indicating a fragile index.
    FragileIndex { fiedler: f64 },
    /// The spectral gap is too small, indicating poor expansion.
    PoorExpansion { gap: f64 },
    /// Effective resistance is too high.
    HighResistance { resistance: f64 },
    /// The overall SCS is below the acceptable threshold.
    LowCoherence { scs: f64 },
    /// A full index rebuild is recommended.
    RebuildRecommended { reason: String },
}

/// Health monitor for HNSW graph indices using spectral coherence.
pub struct HnswHealthMonitor {
    tracker: SpectralTracker,
    min_fiedler: f64,
    min_spectral_gap: f64,
    max_resistance: f64,
    min_composite_scs: f64,
}

impl HnswHealthMonitor {
    /// Create a new health monitor with the given configuration.
    pub fn new(config: SpectralConfig) -> Self {
        Self {
            tracker: SpectralTracker::new(config),
            min_fiedler: 0.05,
            min_spectral_gap: 0.01,
            max_resistance: 0.95,
            min_composite_scs: 0.3,
        }
    }

    /// Update the monitor after an edge change in the graph.
    pub fn update(
        &mut self,
        laplacian: &CsrMatrixView,
        edge_change: Option<(usize, usize, f64)>,
    ) {
        if let Some((u, v, delta)) = edge_change {
            self.tracker.update_edge(laplacian, u, v, delta);
        } else {
            self.tracker.full_recompute(laplacian);
        }
    }

    /// Check current health and return any alerts.
    pub fn check_health(&self) -> Vec<HealthAlert> {
        let score = self.tracker.build_score();
        let mut alerts = Vec::new();

        if score.fiedler < self.min_fiedler {
            alerts.push(HealthAlert::FragileIndex { fiedler: score.fiedler });
        }
        if score.spectral_gap < self.min_spectral_gap {
            alerts.push(HealthAlert::PoorExpansion { gap: score.spectral_gap });
        }
        if score.effective_resistance > self.max_resistance {
            alerts.push(HealthAlert::HighResistance {
                resistance: score.effective_resistance,
            });
        }
        if score.composite < self.min_composite_scs {
            alerts.push(HealthAlert::LowCoherence { scs: score.composite });
        }

        // Recommend rebuild if multiple issues detected.
        if alerts.len() >= 2 {
            alerts.push(HealthAlert::RebuildRecommended {
                reason: format!(
                    "Multiple health issues detected ({}). Full rebuild recommended.",
                    alerts.len()
                ),
            });
        }

        alerts
    }

    /// Get the current spectral coherence score.
    pub fn score(&self) -> SpectralCoherenceScore {
        self.tracker.build_score()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Triangle graph: 3 vertices, 3 edges with weight 1.
    fn triangle_edges() -> Vec<(usize, usize, f64)> {
        vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
    }

    /// Path graph: 0 -- 1 -- 2 -- 3
    fn path_edges() -> Vec<(usize, usize, f64)> {
        vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    }

    /// Square (cycle) graph: 0-1-2-3-0 (4-regular cycle).
    fn cycle4_edges() -> Vec<(usize, usize, f64)> {
        vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)]
    }

    #[test]
    fn test_laplacian_construction() {
        let lap = CsrMatrixView::build_laplacian(3, &triangle_edges());
        assert_eq!(lap.rows, 3);
        assert_eq!(lap.cols, 3);

        // Each row of the Laplacian should sum to zero.
        for i in 0..3 {
            let start = lap.row_ptr[i];
            let end = lap.row_ptr[i + 1];
            let row_sum: f64 = lap.values[start..end].iter().sum();
            assert!(
                row_sum.abs() < 1e-10,
                "Row {} sum = {} (expected 0.0)",
                i,
                row_sum
            );
        }

        // Diagonal should be 2.0 for each vertex in a triangle.
        for i in 0..3 {
            let start = lap.row_ptr[i];
            let end = lap.row_ptr[i + 1];
            let mut diag = 0.0;
            for idx in start..end {
                if lap.col_indices[idx] == i {
                    diag = lap.values[idx];
                }
            }
            assert!(
                (diag - 2.0).abs() < 1e-10,
                "Diagonal[{}] = {} (expected 2.0)",
                i,
                diag
            );
        }
    }

    #[test]
    fn test_fiedler_value_triangle() {
        // Triangle (K3): eigenvalues of Laplacian are 0, 3, 3.
        // Fiedler value (second smallest) = 3.0.
        let lap = CsrMatrixView::build_laplacian(3, &triangle_edges());
        let (fiedler, _vec) = estimate_fiedler(&lap, 200, 1e-8);
        assert!(
            (fiedler - 3.0).abs() < 0.15,
            "Triangle Fiedler = {} (expected ~3.0)",
            fiedler
        );
    }

    #[test]
    fn test_fiedler_value_path() {
        // Path graph on 4 vertices: eigenvalues are 0, 2-sqrt(2), 2, 2+sqrt(2).
        // Fiedler = 2 - sqrt(2) ~= 0.5858.
        let lap = CsrMatrixView::build_laplacian(4, &path_edges());
        let (fiedler, _vec) = estimate_fiedler(&lap, 200, 1e-8);
        let expected = 2.0 - std::f64::consts::SQRT_2;
        assert!(
            (fiedler - expected).abs() < 0.15,
            "Path Fiedler = {} (expected ~{})",
            fiedler,
            expected
        );
    }

    #[test]
    fn test_degree_regularity_regular_graph() {
        // Cycle graph C4: all degrees = 2, perfectly regular.
        let lap = CsrMatrixView::build_laplacian(4, &cycle4_edges());
        let reg = compute_degree_regularity(&lap);
        assert!(
            (reg - 1.0).abs() < 1e-10,
            "Regularity of C4 = {} (expected 1.0)",
            reg
        );
    }

    #[test]
    fn test_scs_bounds() {
        let mut tracker = SpectralTracker::new(SpectralConfig::default());
        let lap = CsrMatrixView::build_laplacian(4, &cycle4_edges());
        let score = tracker.compute(&lap);

        assert!(
            score.composite >= 0.0 && score.composite <= 1.0,
            "SCS composite {} out of [0, 1]",
            score.composite
        );
        assert!(score.fiedler >= 0.0 && score.fiedler <= 1.0);
        assert!(score.spectral_gap >= 0.0 && score.spectral_gap <= 1.0);
        assert!(score.effective_resistance >= 0.0 && score.effective_resistance <= 1.0);
        assert!(score.degree_regularity >= 0.0 && score.degree_regularity <= 1.0);
    }

    #[test]
    fn test_scs_monotonicity() {
        // Full graph should have higher SCS than a sparser subgraph.
        let full_edges = vec![
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 2, 1.0),
            (1, 3, 1.0),
            (2, 3, 1.0),
        ];
        let sparse_edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)];

        let mut tracker_full = SpectralTracker::new(SpectralConfig::default());
        let mut tracker_sparse = SpectralTracker::new(SpectralConfig::default());

        let lap_full = CsrMatrixView::build_laplacian(4, &full_edges);
        let lap_sparse = CsrMatrixView::build_laplacian(4, &sparse_edges);

        let score_full = tracker_full.compute(&lap_full);
        let score_sparse = tracker_sparse.compute(&lap_sparse);

        assert!(
            score_full.composite >= score_sparse.composite,
            "Full SCS {} should >= sparse SCS {}",
            score_full.composite,
            score_sparse.composite
        );
    }

    #[test]
    fn test_tracker_incremental() {
        // Start with a well-connected base graph.
        let edges = vec![
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 0, 1.0),
            (0, 2, 1.0),
            (1, 3, 1.0),
        ];
        let mut tracker = SpectralTracker::new(SpectralConfig::default());
        let lap = CsrMatrixView::build_laplacian(4, &edges);
        tracker.compute(&lap);
        let score_before = tracker.score();

        // Small perturbation: slightly increase one edge weight.
        // First-order perturbation theory is accurate for small changes.
        let delta = 0.05;
        let edges_updated: Vec<(usize, usize, f64)> = edges
            .iter()
            .map(|&(u, v, w)| {
                if u == 1 && v == 3 {
                    (u, v, w + delta)
                } else {
                    (u, v, w)
                }
            })
            .collect();
        let lap_updated = CsrMatrixView::build_laplacian(4, &edges_updated);
        tracker.update_edge(&lap_updated, 1, 3, delta);
        let score_incremental = tracker.score();

        // Full recompute for comparison.
        let mut tracker_full = SpectralTracker::new(SpectralConfig::default());
        let score_full = tracker_full.compute(&lap_updated).composite;

        // Incremental should be within 50% of full recompute for small
        // perturbations. The approximation only updates Fiedler and gap
        // components; resistance is not re-estimated incrementally.
        let diff = (score_incremental - score_full).abs();
        let tolerance = 0.5 * score_full.max(0.01);
        assert!(
            diff < tolerance,
            "Incremental score {} differs from full {} by {} (tolerance {})",
            score_incremental,
            score_full,
            diff,
            tolerance
        );

        // After refresh threshold, tracker forces a full recompute.
        let mut tracker_refresh = SpectralTracker::new(SpectralConfig {
            refresh_threshold: 1,
            ..SpectralConfig::default()
        });
        tracker_refresh.compute(&lap);
        // Mark as needing refresh.
        tracker_refresh.updates_since_refresh = 1;
        assert!(tracker_refresh.needs_refresh());
        tracker_refresh.update_edge(&lap_updated, 1, 3, delta);
        // After forced recompute, should match full closely.
        let score_refreshed = tracker_refresh.score();
        let diff_refreshed = (score_refreshed - score_full).abs();
        assert!(
            diff_refreshed < 0.05,
            "Refreshed score {} should closely match full {} (diff {})",
            score_refreshed,
            score_full,
            diff_refreshed
        );
    }

    #[test]
    fn test_health_alerts() {
        // Build a barely connected graph (path) that should trigger alerts.
        let edges = vec![(0, 1, 0.01), (1, 2, 0.01)];
        let lap = CsrMatrixView::build_laplacian(3, &edges);

        let mut monitor = HnswHealthMonitor::new(SpectralConfig::default());
        monitor.update(&lap, None);

        let alerts = monitor.check_health();
        // With such weak edges, we expect low coherence alerts.
        let has_fragile = alerts
            .iter()
            .any(|a| matches!(a, HealthAlert::FragileIndex { .. }));
        let has_low_coherence = alerts
            .iter()
            .any(|a| matches!(a, HealthAlert::LowCoherence { .. }));

        assert!(
            has_fragile || has_low_coherence,
            "Weak graph should trigger FragileIndex or LowCoherence alert. Got: {:?}",
            alerts
        );

        // A well-connected graph should produce fewer or no alerts.
        let strong_edges = vec![
            (0, 1, 1.0),
            (1, 2, 1.0),
            (0, 2, 1.0),
        ];
        let strong_lap = CsrMatrixView::build_laplacian(3, &strong_edges);
        let mut strong_monitor = HnswHealthMonitor::new(SpectralConfig::default());
        strong_monitor.update(&strong_lap, None);
        let strong_alerts = strong_monitor.check_health();

        assert!(
            strong_alerts.len() <= alerts.len(),
            "Strong graph should have fewer alerts ({}) than weak graph ({})",
            strong_alerts.len(),
            alerts.len()
        );
    }
}
