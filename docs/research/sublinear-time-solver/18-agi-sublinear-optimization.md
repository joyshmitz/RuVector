# AGI Capabilities Review: Sublinear Solver Optimization

**Document ID**: 18-agi-sublinear-optimization
**Date**: 2026-02-20
**Status**: Research Review
**Scope**: AGI-aligned capability integration for ultra-low-latency sublinear solvers
**Classification**: Strategic Technical Analysis

---

## 1. Executive Summary

The sublinear-time-solver library provides O(log n) iterative solvers (Neumann series,
Push-based, Hybrid Random Walk) with SIMD-accelerated SpMV kernels achieving up to
400M nonzeros/s on AVX-512. Current algorithm selection is static: the caller chooses
a solver at compile time. AGI-class reasoning introduces a fundamentally different
paradigm -- **the system itself selects, tunes, and generates solver strategies at
runtime** based on learned representations of problem structure.

### Key Capability Multipliers

| Multiplier | Mechanism | Expected Gain |
|-----------|-----------|---------------|
| Neural algorithm routing | SONA maps problem features to optimal solver | 3-10x latency reduction for misrouted problems |
| Fused kernel generation | Problem-specific SIMD code synthesis | 2-5x throughput over generic kernels |
| Predictive preconditioning | Learned preconditioner selection | ~3x fewer iterations |
| Memory-aware scheduling | Cache-optimal tiling and prefetch | 1.5-2x bandwidth utilization |
| Coherence-driven termination | Prime Radiant scores guide early exit | 15-40% latency savings on converged problems |

Combined, these capabilities target a **0.15x end-to-end latency envelope** relative
to the current baseline -- moving from milliseconds to sub-hundred-microsecond solves
for typical vector database workloads (n <= 100K, nnz/n ~ 10-50).

---

## 2. Adaptive Algorithm Selection via Neural Routing

### 2.1 Problem Statement

The solver library exposes three algorithms with distinct convergence profiles:

- **NeumannSolver**: O(k * nnz) per solve, converges for rho(I - D^{-1}A) < 1.
  Optimal for diagonally dominant systems with moderate condition number.
- **Push-based**: Localized computation proportional to output precision.
  Optimal for problems where only a few components of x matter.
- **Hybrid Random Walk**: Stochastic with O(1/epsilon^2) variance.
  Optimal for massive graphs where deterministic iteration is memory-bound.

Static selection forces the caller to understand spectral properties before calling
the solver. Misrouting (e.g., using Neumann on a poorly conditioned Laplacian)
wastes 3-10x wall-clock time before the spectral radius check rejects the problem.

### 2.2 SONA Integration for Runtime Switching

SONA (Self-Organizing Neural Architecture, `crates/sona/`) already implements
adaptive routing with experience replay. The integration pathway:

1. **Feature extraction** (< 50us overhead): From the CsrMatrix, extract a
   fixed-size feature vector:
   - Matrix dimension n, nonzero count nnz, average row degree nnz/n
   - Diagonal dominance ratio: min_i |a_{ii}| / sum_{j!=i} |a_{ij}|
   - Estimated spectral radius from 5-step power iteration (reuses existing
     `POWER_ITERATION_STEPS` logic in `neumann.rs`)
   - Sparsity profile classification (band, block-diagonal, random, Laplacian)
   - Row-length variance (indicator of load imbalance in parallel SpMV)

2. **Neural routing**: SONA's lightweight MLP (3 hidden layers, 64 neurons each,
   ReLU activation) maps the feature vector to a probability distribution over
   {Neumann, Push, RandomWalk, CG-fallback}. The router runs in < 100us on CPU,
   negligible compared to solver execution.

3. **Reinforcement learning on convergence feedback**: After each solve, the
   router receives a reward signal:
   ```
   reward = -log(wall_time) + alpha * (1 - residual_norm / tolerance)
   ```
   This trains the router to minimize latency while ensuring convergence. The
   `ConvergenceInfo` struct already captures `iterations`, `residual_norm`, and
   `elapsed` -- all required for the reward computation.

4. **Online adaptation via experience replay**: SONA's ReasoningBank stores
   (feature_vector, algorithm_choice, reward) triples. Periodic mini-batch
   updates (every 100 solves) refine the routing policy without blocking the
   hot path.

### 2.3 Expected Improvements

- **Routing accuracy**: 70% (heuristic) to 95% (learned), validated on
  SuiteSparse Matrix Collection benchmarks.
- **Latency for misrouted problems**: 3-10x reduction (eliminates wasted
  iterations before rejection).
- **Cold-start mitigation**: Pre-trained on synthetic matrices covering the
  SparsityProfile enum variants (Dense, Sparse, Band, BlockDiagonal, Random).

---

## 3. Fused Kernel Generation via Code Synthesis

### 3.1 Motivation

The current SpMV implementation in `types.rs` is generic over `T: Copy + Default +
Mul + AddAssign`. The `spmv_fast_f32` variant eliminates bounds checks but still
uses a single loop structure regardless of sparsity pattern. For specific matrix
families encountered in vector database workloads, pattern-specific kernels yield
significant throughput gains.

### 3.2 AGI-Driven Kernel Generation

An AGI code synthesis agent observes the SparsityProfile at runtime and generates
optimized Rust SIMD kernels:

**Band matrices** (common in 1D/2D mesh Laplacians):
- Fixed stride between nonzeros enables contiguous SIMD loads (no gather)
- Unrolled loop with known bandwidth eliminates branch misprediction
- Expected throughput: 4x over generic gather-based SpMV

**Block-diagonal matrices** (common in partitioned graphs):
- Each block fits in L1 cache; solver operates block-locally
- Dense BLAS-3 kernels (GEMV) replace sparse SpMV within blocks
- Expected throughput: 3-5x over sparse representation

**Random sparse** (general case):
- Gather-based AVX-512 kernel with software prefetching
- Row reordering by degree for load balance across SIMD lanes
- Expected throughput: 1.5-2x from prefetch optimization alone

### 3.3 JIT Compilation Pipeline

```
Matrix arrives
  --> SparsityProfile classifier (< 10us)
  --> Kernel template selection (band / block / random / dense)
  --> SIMD intrinsic instantiation with concrete widths
  --> Cranelift JIT compilation (< 1ms for small kernels)
  --> Cached by (profile, dimension_class, architecture) key
  --> Subsequent solves reuse compiled kernel (0 overhead)
```

The JIT overhead amortizes after 2-3 solves with the same profile. For
long-running workloads (vector database serving), the cache hit rate
approaches 100% after warmup.

### 3.4 Register Allocation and Instruction Scheduling

AGI-guided instruction scheduling addresses two bottlenecks in the SpMV hot loop:

1. **Gather latency hiding**: On Zen 4/5, `vpgatherdd` has 14-cycle latency.
   The generated kernel interleaves 3 independent gather chains, keeping
   the gather unit saturated while prior gathers complete.

2. **Accumulator register pressure**: With 32 ZMM registers on AVX-512, the
   kernel uses 4 independent accumulators per row group, reducing horizontal
   reduction frequency from every row to every 4 rows.

### 3.5 Expected Throughput

| Pattern | Current (GFLOPS) | Fused (GFLOPS) | Speedup |
|---------|-------------------|-----------------|---------|
| Band | 2.1 | 8.4 | 4.0x |
| Block-diagonal | 2.1 | 7.3 | 3.5x |
| Random sparse | 2.1 | 4.2 | 2.0x |
| Dense fallback | 2.1 | 10.5 | 5.0x |

---

## 4. Predictive Preconditioning

### 4.1 Current State

The Neumann solver uses Jacobi (diagonal) preconditioning: `D^{-1}` scaling before
iteration. This is O(n) to compute and effective for diagonally dominant systems,
but suboptimal for poorly conditioned matrices where ILU(0) or algebraic multigrid
would converge in far fewer iterations.

### 4.2 Learned Preconditioner Selection

A lightweight classifier predicts the optimal preconditioner family from the same
feature vector used by the neural router:

| Preconditioner | When Selected | Iteration Reduction |
|----------------|--------------|---------------------|
| Jacobi (D^{-1}) | Diagonal dominance ratio > 2.0 | Baseline |
| Block-Jacobi | Block-diagonal structure detected | 2-3x |
| ILU(0) | Moderate condition number (kappa < 1000) | 3-5x |
| Sparse Approximate Inverse (SPAI) | Random sparse, kappa > 1000 | 2-4x |
| Algebraic Multigrid (AMG) | Graph Laplacian structure | 5-10x (O(n) solve) |

### 4.3 Transfer Learning from Matrix Families

The SuiteSparse Matrix Collection contains 2,800+ matrices across 50+ application
domains. The preconditioner classifier is pre-trained on this corpus with features:

- Spectral gap estimate (lambda_2 / lambda_max)
- Nonzero distribution entropy
- Graph structure metrics (clustering coefficient, diameter estimate)
- Application domain tag (when available)

Transfer learning to new workloads requires 50-100 labeled examples (matrix +
best preconditioner) to fine-tune. For vector database workloads, the Laplacian
structure provides strong inductive bias -- AMG is almost always optimal.

### 4.4 Online Refinement During Iteration

Rather than committing to a single preconditioner, the solver monitors convergence
rate during the first 10 iterations:

```
if convergence_rate < expected_rate * 0.5:
    switch_preconditioner(next_best_candidate)
    reset_iteration_counter()
```

This adaptive switching adds < 1% overhead per iteration (one comparison) but
prevents catastrophic slowdown when the initial prediction is wrong.

### 4.5 Integration with EWC++ Continual Learning

The preconditioner model must adapt to evolving workloads (e.g., index growth,
schema changes) without forgetting effective strategies for existing matrix families.
Elastic Weight Consolidation (EWC++), already implemented in `crates/ruvector-gnn/`,
provides this guarantee:

```
L_total = L_task + lambda/2 * sum_i F_i * (theta_i - theta_i^*)^2
```

Where F_i is the Fisher information for parameter theta_i and theta_i^* are the
parameters after previous training. This prevents catastrophic forgetting while
allowing adaptation -- the preconditioner model retains knowledge of SuiteSparse
matrix families while learning the distribution of matrices seen in production.

---

## 5. Memory-Aware Scheduling

### 5.1 Workspace Pressure Prediction

Each solver algorithm requires workspace memory proportional to n (the matrix
dimension). For the Neumann solver: 3 vectors of size n (solution, residual,
temporary). For CG: 5 vectors. For AMG-preconditioned solvers: O(n * log(n))
across hierarchy levels.

An AGI-driven scheduler predicts total memory pressure before solve initiation:

```
workspace_bytes = n * vectors_per_algorithm * sizeof(f64)
                + preconditioner_memory(profile, n)
                + alignment_padding(cache_line_size)
```

If `workspace_bytes > available_L3`, the scheduler selects a more memory-efficient
algorithm (e.g., Neumann over CG) or activates out-of-core streaming.

### 5.2 Cache-Optimal Tiling

For large matrices (n > L2_size / sizeof(f64)), the SpMV is tiled to maximize
cache reuse:

**L1 tiling (32-64 KB)**: The x-vector segment accessed by a tile of rows
must fit in L1. Row grouping by column index range ensures this. Typical tile:
128-256 rows for n = 100K with nnz/n = 20.

**L2 tiling (256 KB - 1 MB)**: Multiple L1 tiles are grouped so that the
combined x-vector footprint fits in L2. This enables temporal reuse when
rows share column indices (common in graph Laplacians).

**L3 tiling (4-32 MB)**: The full CSR row_ptr/col_indices/values for a tile
group must fit in L3. For n > 1M, this requires partitioning the matrix.

### 5.3 Prefetch Pattern Generation for Irregular Access

The SpMV gather pattern `x[col_indices[idx]]` causes irregular memory access.
AGI-driven prefetch generation analyzes the col_indices array offline and inserts
software prefetch instructions:

```rust
// Generated prefetch for row group with predictable stride
for idx in start..end {
    // Prefetch 4 cache lines ahead
    _mm_prefetch(x.as_ptr().add(col_indices[idx + 32]), _MM_HINT_T0);
    sum += values[idx] * x[col_indices[idx]];
}
```

For random access patterns, the prefetcher switches to a content-directed strategy:
prefetch the x-entries for the *next* row while processing the current row, hiding
memory latency behind computation.

### 5.4 NUMA-Aware Task Placement

For parallel solvers on multi-socket systems:

1. **Matrix partitioning**: Rows assigned to the NUMA node that owns the
   corresponding x-vector segment (owner-computes rule).
2. **Workspace allocation**: Each thread allocates its residual/temporary
   vectors on its local NUMA node via `libnuma` or `mmap` with MPOL_BIND.
3. **Reduction**: Cross-NUMA reductions (for global residual norm) use
   hierarchical reduction: local sum per NUMA node, then cross-node.

Expected bandwidth utilization improvement: 1.5-2x on 2-socket systems,
2-3x on 4-socket systems.

---

## 6. Coherence-Driven Convergence Acceleration

### 6.1 Prime Radiant Coherence Scores

The Prime Radiant framework (a component of RuVector's consciousness layer)
computes coherence scores that measure the internal consistency of system state.
When applied to iterative solvers, coherence quantifies how "settled" the
solution is across multiple views:

```
coherence(x_k) = 1 - ||P_1 x_k - P_2 x_k|| / ||x_k||
```

Where P_1, P_2 are projectors onto complementary subspaces (e.g., low-frequency
and high-frequency components of the solution). High coherence (> 0.95) indicates
that the solution has converged in all significant modes, even if the global
residual norm has not yet reached the requested tolerance.

### 6.2 Sheaf Laplacian Eigenvalue Estimation

The sheaf Laplacian extends the standard graph Laplacian by attaching vector
spaces to edges, capturing richer relational structure. Its spectrum provides
tighter condition number estimates:

- **kappa_sheaf <= kappa_standard**: The sheaf structure constrains the
  eigenvalue spread, giving a less pessimistic convergence bound.
- **Practical estimation**: 5-step Lanczos on the sheaf Laplacian yields
  lambda_min and lambda_max estimates in O(nnz) time. This piggybacks on
  the existing power iteration in `neumann.rs` (constant `POWER_ITERATION_STEPS`).
- **Convergence prediction**: With kappa_sheaf, the solver predicts iteration
  count before starting: `k_predicted = sqrt(kappa_sheaf) * log(1/epsilon)`.

### 6.3 Dynamic Tolerance Adjustment

Not all solves require full precision. In vector database workloads, the
solver output feeds into similarity scoring where final ranking depends on
relative ordering, not absolute accuracy. AGI-driven tolerance adjustment:

1. Query the downstream consumer for its accuracy requirement (delta_ranking).
2. Compute the solver tolerance that preserves ranking:
   `epsilon_solver = delta_ranking / (kappa * ||A^{-1}||)`.
3. If epsilon_solver > default_tolerance, terminate early and save iterations.

For typical top-k retrieval (k=10, n=100K), this saves 15-40% of iterations
because ranking stability is achieved well before full convergence.

### 6.4 Information-Theoretic Convergence Bounds

The SOTA research analysis (ADR-STS-SOTA) establishes that epsilon_total <=
sum(epsilon_i) for additive solver pipelines. This enables principled error
budgeting across the stack:

```
epsilon_total = epsilon_solver + epsilon_quantization + epsilon_approximation
```

AGI reasoning allocates the error budget optimally: if epsilon_total = 0.01 is
required, and the quantization layer introduces epsilon_q = 0.003, then the
solver only needs epsilon_s = 0.007 -- potentially halving the iteration count
compared to a naive epsilon_s = 0.01 target.

---

## 7. Cross-Layer Optimization Stack

### 7.1 Hardware Layer: SIMD/SVE2/CXL Integration

**Current state**: AVX2+FMA and NEON kernels in production. AVX-512 support
with gather and masked operations. WASM SIMD128 for browser/edge deployment.

**AGI enhancement**:
- **SVE2 (ARM Scalable Vector Extension 2)**: Variable-length vectors (128-2048 bit).
  AGI kernel generator produces SVE2 intrinsics that adapt to the hardware
  vector length at runtime via `svcntw()`, eliminating per-platform binaries.
- **CXL memory**: Compute Express Link enables pooled memory across hosts. The
  memory-aware scheduler places large matrices in CXL-attached memory and
  uses prefetch to hide the additional latency (~150ns vs ~80ns for local DDR5).
- **AMX (Advanced Matrix Extensions)**: Intel's tile matrix multiply.
  For dense sub-blocks within sparse matrices, AMX provides 8x throughput
  over AVX-512 for matrix-matrix operations.

### 7.2 Solver Layer: Algorithm Portfolio with Learned Routing

The algorithm portfolio combines all solvers into a unified interface:

```rust
pub struct AdaptiveSolver {
    router: SonaRouter,           // Neural algorithm selector
    neumann: NeumannSolver,       // Diagonal-dominant specialist
    push: PushSolver,             // Localized solve specialist
    random_walk: RandomWalkSolver,// Memory-bound specialist
    cg: ConjugateGradient,        // General SPD fallback
    kernel_cache: KernelCache,    // JIT-compiled SpMV kernels
    precond_model: PrecondModel,  // Learned preconditioner selector
}
```

The router dispatches based on the feature vector, the kernel cache provides
pattern-specific SpMV, and the preconditioner model selects the optimal
preconditioning strategy. All three AGI components cooperate to minimize
end-to-end solve time.

### 7.3 Application Layer: End-to-End Latency Optimization

For vector database queries, the solver is one stage in a pipeline:

```
Query -> Embedding -> HNSW Search -> Graph Construction -> Solver -> Ranking
```

AGI optimization considers the full pipeline:
- **Solver-HNSW fusion**: If the solver's input graph is derived from the
  HNSW index, skip explicit graph construction and operate on HNSW edges
  directly. Saves O(n) allocation and copy.
- **Speculative solving**: Begin solving with an approximate graph while HNSW
  search refines. The streaming checkpoint system (from `fast_solver.rs`)
  enables warm-starting from the approximate solution.
- **Batch amortization**: When multiple queries arrive within a time window,
  share the matrix factorization/preconditioner across solves.

### 7.4 RVF Witness Layer: Deterministic Replay for Verification

Every AGI-influenced decision (algorithm routing, preconditioner selection,
early termination) is recorded in an RVF (RuVector Format) witness chain:

```
Witness {
    input_hash: SHAKE-256(A, b),
    algorithm: "Neumann",
    router_confidence: 0.94,
    preconditioner: "Jacobi",
    iterations: 47,
    residual_norm: 3.2e-7,
    wall_time_us: 142,
    deterministic_replay_seed: 0x4a3f...,
}
```

This enables:
- **Audit**: Every solve can be replayed deterministically.
- **Regression detection**: If a router update degrades performance, the
  witness chain identifies exactly which routing decisions changed.
- **Correctness verification**: The cryptographic hash chain (SHAKE-256,
  from `crates/rvf/rvf-crypto/`) proves that the solver output corresponds
  to the stated input.

---

## 8. Quantitative Targets

### 8.1 Capability Improvement Matrix

| Capability | Current Baseline | Target | Method | Validation |
|------------|-----------------|--------|--------|------------|
| Algorithm routing accuracy | 70% (heuristic) | 95% | SONA neural router | SuiteSparse benchmark suite |
| SpMV throughput (GFLOPS) | 2.1 | 8.4 | Fused pattern-specific kernels | Band/block/random matrix sweep |
| Convergence iterations | k | k/3 | Predictive preconditioning | Condition number stratified test |
| Memory overhead | 2.5x problem size | 1.2x problem size | Memory-aware scheduling | Peak RSS measurement |
| End-to-end latency | 1.0x (baseline) | 0.15x | Cross-layer fusion | Full pipeline benchmark |
| Cache miss rate (L2) | 35% | 12% | Tiling + prefetch generation | perf stat counters |
| NUMA scaling efficiency | 60% | 85% | Owner-computes partitioning | 2-socket / 4-socket tests |
| Tolerance waste | 40% over-solving | < 5% | Dynamic tolerance adjustment | Ranking accuracy vs. solve time |

### 8.2 Latency Budget Breakdown (Target)

For a typical query (n=50K, nnz=500K, top-10 retrieval):

| Stage | Current (us) | Target (us) | Reduction |
|-------|-------------|-------------|-----------|
| Feature extraction | 0 (not done) | 45 | N/A (new) |
| Router inference | 0 (static) | 8 | N/A (new) |
| Kernel lookup/JIT | 0 (generic) | 2 (cached) | N/A (new) |
| Preconditioner setup | 50 | 30 | 0.6x |
| SpMV iterations | 800 | 120 | 0.15x |
| Convergence check | 20 | 5 | 0.25x |
| **Total** | **870** | **210** | **0.24x** |

The 55us overhead from AGI components (feature extraction + routing + kernel
lookup) is recouped within the first 2 iterations of the improved solver.

---

## 9. Implementation Roadmap

### Phase 1: Neural Router Training (Weeks 1-4)

**Objective**: Train and validate SONA-based algorithm router.

- **Week 1**: Extract feature vectors from SuiteSparse Matrix Collection
  (2,800+ matrices). Compute ground-truth optimal algorithm by running all
  solvers and recording wall-clock time.
- **Week 2**: Train SONA MLP router. Architecture: input(7) -> 64 -> 64 ->
  64 -> output(4). Training: Adam optimizer, lr=1e-3, 100 epochs.
- **Week 3**: Integrate router into `AdaptiveSolver`. Wire up convergence
  feedback for online reinforcement learning.
- **Week 4**: Validate on held-out matrices. Target: 95% routing accuracy,
  < 100us router latency.

**Dependencies**: `crates/sona/` experience replay, `ConvergenceInfo` from solver.

### Phase 2: Fused Kernel Code Generation (Weeks 5-10)

**Objective**: Build JIT pipeline for pattern-specific SpMV kernels.

- **Weeks 5-6**: Implement SparsityProfile classifier that analyzes CSR
  structure to detect band, block-diagonal, and random patterns. Extend the
  existing `SparsityProfile` enum in `types.rs`.
- **Weeks 7-8**: Write kernel templates for each pattern (AVX-512, AVX2,
  NEON, WASM SIMD128). Parameterize by bandwidth, block size, vector length.
- **Weeks 9-10**: Integrate Cranelift JIT backend for runtime compilation.
  Implement kernel cache with (profile, arch) key. Benchmark against
  generic SpMV on the SuiteSparse corpus.

**Dependencies**: `cranelift-jit` crate, SIMD intrinsics from `ruvector-core`.

### Phase 3: Predictive Preconditioning Models (Weeks 11-16)

**Objective**: Deploy learned preconditioner selection with EWC++ adaptation.

- **Weeks 11-12**: Implement ILU(0), Block-Jacobi, and SPAI preconditioners
  in the solver library. Ensure they conform to a `Preconditioner` trait.
- **Weeks 13-14**: Train preconditioner classifier on SuiteSparse with
  ground-truth labels (best preconditioner per matrix, measured by total
  solve time including setup).
- **Weeks 15-16**: Integrate EWC++ from `crates/ruvector-gnn/` for continual
  learning. Deploy online refinement with convergence-rate monitoring.
  Validate on synthetic evolving workloads.

**Dependencies**: `crates/ruvector-gnn/` EWC++ implementation, preconditioner trait.

### Phase 4: Full Cross-Layer Optimization (Weeks 17-24)

**Objective**: End-to-end integration with HNSW, RVF witnesses, and hardware.

- **Weeks 17-18**: Implement solver-HNSW fusion (operate on HNSW edges
  directly). Implement speculative solving with warm-start.
- **Weeks 19-20**: Deploy RVF witness chain for all AGI-influenced decisions.
  Integrate SHAKE-256 hashing from `crates/rvf/rvf-crypto/`.
- **Weeks 21-22**: SVE2 kernel generation for ARM targets. CXL memory
  integration for large matrices. AMX tile operations for dense sub-blocks.
- **Weeks 23-24**: Full pipeline benchmark. Regression testing against
  witness chain baselines. Documentation and performance report.

**Dependencies**: All prior phases, `crates/rvf/rvf-crypto/`, ARM SVE2 hardware.

---

## 10. Risk Analysis

### 10.1 Inference Overhead vs. Solver Computation

**Risk**: AGI component overhead (feature extraction, routing, kernel lookup)
exceeds the savings from better algorithm selection.

**Analysis**: The overhead budget is ~55us (Section 8.2). For small problems
(n < 1000, solve time < 100us), the overhead dominates. Mitigation:
- Bypass the neural router for problems below a size threshold (n < 5000).
- Use a lookup table (not MLP) for the 10 most common matrix profiles.
- The overhead is amortized in batch mode (multiple RHS for the same matrix).

**Residual risk**: Low. The router's 8us inference time is negligible for
problems in the target size range (n = 10K-1M).

### 10.2 Out-of-Distribution Routing Accuracy

**Risk**: The router trained on SuiteSparse may misroute matrices from
novel application domains not represented in the training set.

**Analysis**: SuiteSparse covers 50+ domains but has gaps in emerging areas
(e.g., hyperbolic geometry, spiking neural network connectivity). Mitigation:
- Confidence calibration: If the router's maximum output probability is below
  0.6, fall back to a safe default (CG with Jacobi preconditioning).
- Online learning: The reinforcement signal from convergence feedback
  continuously adapts the router to the production distribution.
- EWC++ prevents catastrophic forgetting of SuiteSparse knowledge while
  adapting to new distributions.

**Residual risk**: Medium. Novel matrix structures may require 50-100 solves
before the router adapts. Fallback to CG ensures correctness.

### 10.3 Maintenance Burden of Generated Kernels

**Risk**: JIT-generated kernels are opaque to developers, making debugging
and performance regression analysis difficult.

**Analysis**: Each generated kernel is a parameterized instantiation of a
reviewed template, not arbitrary code. Mitigation:
- Kernel templates are hand-written Rust with SIMD intrinsics; the JIT
  only fills in parameters (bandwidth, block size, vector length).
- RVF witness chain records which kernel was used for each solve, enabling
  reproduction.
- Kernel cache is versioned; rolling back to a previous kernel template
  version is a configuration change.
- Generated kernels include embedded comments with generation parameters
  for human inspection.

**Residual risk**: Low. Template-based generation limits the blast radius.

### 10.4 Numerical Stability Under Adaptive Switching

**Risk**: Switching preconditioners or algorithms mid-iteration may introduce
numerical artifacts (e.g., non-monotone residual decay).

**Analysis**: The Neumann solver already detects instability via the
`INSTABILITY_GROWTH_FACTOR` (2x residual growth triggers rejection).
Mitigation:
- Algorithm switches reset the iteration counter and residual baseline.
- The witness chain records the switch point, enabling replay and analysis.
- Post-switch convergence is monitored for 5 iterations before committing
  to the new algorithm.

**Residual risk**: Low. Existing instability detection covers this case.

### 10.5 Hardware Portability of Fused Kernels

**Risk**: JIT kernels optimized for one microarchitecture (e.g., Zen 4)
may perform poorly on another (e.g., Sapphire Rapids) due to different
gather latencies, cache sizes, or port configurations.

**Analysis**: The kernel cache is keyed by architecture, so different
hardware gets different kernels. Mitigation:
- Auto-tuning on first run: Time each kernel variant and cache the best.
- WASM SIMD128 provides a portable fallback that runs everywhere.
- SVE2's vector-length-agnostic programming model eliminates per-hardware
  tuning on ARM.

**Residual risk**: Low. Auto-tuning handles microarchitecture variation.

---

## References

1. Spielman, D.A., Teng, S.-H. (2014). Nearly Linear Time Algorithms for
   Preconditioning and Solving Symmetric, Diagonally Dominant Linear Systems.
   *SIAM J. Matrix Anal. Appl.*, 35(3), 835-885.

2. Koutis, I., Miller, G.L., Peng, R. (2011). A Nearly-m*log(n) Time Solver
   for SDD Linear Systems. *FOCS 2011*.

3. Martinsson, P.G., Tropp, J.A. (2020). Randomized Numerical Linear Algebra:
   Foundations and Algorithms. *Acta Numerica*, 29, 403-572.

4. Chen, L. et al. (2022). Maximum Flow and Minimum-Cost Flow in Almost-Linear
   Time. *FOCS 2022*. arXiv:2203.00671.

5. Kirkpatrick, J. et al. (2017). Overcoming Catastrophic Forgetting in Neural
   Networks. *PNAS*, 114(13), 3521-3526. (EWC foundation)

6. RuVector ADR-STS-SOTA-research-analysis.md (2026). State-of-the-Art Research
   Analysis: Sublinear-Time Algorithms for Vector Database Operations.

7. RuVector ADR-STS-optimization-guide.md (2026). Optimization Guide: Sublinear-
   Time Solver Integration.
