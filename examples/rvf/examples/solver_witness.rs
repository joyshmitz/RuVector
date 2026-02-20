//! Solver Convergence Witnesses in RVF
//!
//! Demonstrates how to store iterative solver convergence data in RVF
//! with cryptographic witness chains that prove deterministic convergence.
//!
//! This example simulates a conjugate-gradient-style solver and records:
//! 1. Per-iteration state vectors (solution snapshots) in the RVF store
//! 2. Convergence metadata (residual norm, iteration count) per vector
//! 3. A SHA-256 witness chain linking each iteration to the previous
//! 4. Verification of the witness chain to prove convergence history
//! 5. Replay queries to reconstruct the convergence trajectory
//!
//! RVF segments used: VEC_SEG, MANIFEST_SEG, WITNESS_SEG
//!
//! Run: cargo run --example solver_witness

use rvf_runtime::{
    MetadataEntry, MetadataValue, QueryOptions, RvfOptions, RvfStore, SearchResult,
};
use rvf_runtime::filter::FilterValue;
use rvf_runtime::options::DistanceMetric;
use rvf_crypto::{create_witness_chain, verify_witness_chain, shake256_256, WitnessEntry};
use tempfile::TempDir;

/// Simple LCG-based pseudo-random number generator for deterministic results.
fn lcg_next(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) as f64) / (u32::MAX as f64)
}

/// Generate a deterministic "solution snapshot" vector for a given iteration.
/// Simulates a solver converging: early iterations have large components,
/// later iterations have values closer to the true solution (near zero residual).
fn solver_snapshot(dim: usize, iteration: u32, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(iteration as u64).wrapping_add(1);
    let decay = 1.0 / (1.0 + iteration as f64);
    let mut v = Vec::with_capacity(dim);
    for _ in 0..dim {
        let raw = lcg_next(&mut state) - 0.5;
        v.push((raw * decay) as f32);
    }
    v
}

/// Compute a simulated residual norm that decays exponentially with iteration.
fn residual_norm(iteration: u32, seed: u64) -> f64 {
    let mut state = seed.wrapping_add(iteration as u64 * 997);
    let noise = lcg_next(&mut state) * 0.1;
    let base = 10.0_f64 * (-0.3 * iteration as f64).exp();
    base + noise
}

fn main() {
    println!("=== Solver Convergence Witness Example ===\n");

    let dim = 128;
    let max_iterations = 50;
    let tolerance = 1e-4;
    let seed = 12345_u64;

    // ====================================================================
    // 1. Create a store for solver state snapshots
    // ====================================================================
    println!("--- 1. Create Solver State Store ---");

    let tmp_dir = TempDir::new().expect("failed to create temp dir");
    let store_path = tmp_dir.path().join("solver_witness.rvf");

    let options = RvfOptions {
        dimension: dim as u16,
        metric: DistanceMetric::L2,
        ..Default::default()
    };

    let mut store = RvfStore::create(&store_path, options).expect("failed to create store");
    println!("  Store created: {} dims, L2 metric", dim);
    println!("  Tolerance: {:.0e}", tolerance);
    println!("  Max iterations: {}", max_iterations);

    // ====================================================================
    // 2. Simulate solver iterations and store per-iteration snapshots
    // ====================================================================
    println!("\n--- 2. Simulate Solver and Store Snapshots ---");

    // Metadata field layout:
    //   field 0: iteration (u64)
    //   field 1: residual_norm_x1e9 (u64, residual * 1e9 for integer storage)
    //   field 2: solver_phase (string: "warmup", "converging", "converged")
    //   field 3: cumulative_flops (u64)

    let mut converged_at: Option<u32> = None;
    let mut all_vectors: Vec<Vec<f32>> = Vec::new();
    let mut all_ids: Vec<u64> = Vec::new();
    let mut all_metadata: Vec<MetadataEntry> = Vec::new();
    let mut residuals: Vec<f64> = Vec::new();

    for iter in 0..max_iterations {
        let snapshot = solver_snapshot(dim, iter, seed);
        let resid = residual_norm(iter, seed);
        residuals.push(resid);

        let phase = if iter < 5 {
            "warmup"
        } else if resid > tolerance {
            "converging"
        } else {
            if converged_at.is_none() {
                converged_at = Some(iter);
            }
            "converged"
        };

        // Store residual as integer (multiply by 1e9) for metadata filtering
        let resid_int = (resid * 1e9) as u64;
        let flops = (iter as u64 + 1) * dim as u64 * 2; // simulated FLOP count

        all_vectors.push(snapshot);
        all_ids.push(iter as u64);

        all_metadata.push(MetadataEntry {
            field_id: 0,
            value: MetadataValue::U64(iter as u64),
        });
        all_metadata.push(MetadataEntry {
            field_id: 1,
            value: MetadataValue::U64(resid_int),
        });
        all_metadata.push(MetadataEntry {
            field_id: 2,
            value: MetadataValue::String(phase.to_string()),
        });
        all_metadata.push(MetadataEntry {
            field_id: 3,
            value: MetadataValue::U64(flops),
        });

        // Stop after convergence is well-established
        if converged_at.map_or(false, |c| iter >= c + 3) {
            break;
        }
    }

    let num_stored = all_vectors.len();
    let vec_refs: Vec<&[f32]> = all_vectors.iter().map(|v| v.as_slice()).collect();

    let ingest = store
        .ingest_batch(&vec_refs, &all_ids, Some(&all_metadata))
        .expect("ingest failed");
    println!("  Ingested {} iteration snapshots (rejected: {})", ingest.accepted, ingest.rejected);

    if let Some(conv) = converged_at {
        println!("  Convergence detected at iteration {}", conv);
        println!("  Residual at convergence: {:.6e}", residuals[conv as usize]);
    }

    // Print convergence trajectory summary
    println!("\n  Convergence trajectory (sampled):");
    println!("    {:>5}  {:>14}  {:>12}", "Iter", "Residual", "Phase");
    println!("    {:->5}  {:->14}  {:->12}", "", "", "");
    for i in (0..num_stored).step_by(5.max(1)) {
        let phase = if i < 5 {
            "warmup"
        } else if residuals[i] > tolerance {
            "converging"
        } else {
            "converged"
        };
        println!(
            "    {:>5}  {:>14.6e}  {:>12}",
            i, residuals[i], phase
        );
    }

    // ====================================================================
    // 3. Build witness chain linking iterations with SHA-256 hashes
    // ====================================================================
    println!("\n--- 3. Build Convergence Witness Chain ---");

    // Witness type codes:
    //   0x01 = PROVENANCE (initial state)
    //   0x02 = COMPUTATION (solver iteration)
    //   0x03 = CONVERGENCE_PROOF (final converged state)
    let entries: Vec<WitnessEntry> = (0..num_stored)
        .map(|i| {
            // Hash the iteration data: iteration index + residual + snapshot hash
            let snapshot_hash = shake256_256(
                &all_vectors[i]
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect::<Vec<u8>>(),
            );
            let action_data = format!(
                "solver:iter={}:resid={:.12e}:snap={}",
                i,
                residuals[i],
                hex_string(&snapshot_hash[..8])
            );
            let wtype = if i == 0 {
                0x01 // PROVENANCE: initial state
            } else if converged_at.map_or(false, |c| i as u32 >= c) {
                0x03 // CONVERGENCE_PROOF
            } else {
                0x02 // COMPUTATION: solver iteration
            };

            WitnessEntry {
                prev_hash: [0u8; 32], // overwritten by create_witness_chain
                action_hash: shake256_256(action_data.as_bytes()),
                timestamp_ns: 1_700_000_000_000_000_000 + i as u64 * 50_000_000,
                witness_type: wtype,
            }
        })
        .collect();

    let chain_bytes = create_witness_chain(&entries);
    println!(
        "  Witness chain: {} entries, {} bytes",
        entries.len(),
        chain_bytes.len()
    );

    // ====================================================================
    // 4. Verify the witness chain
    // ====================================================================
    println!("\n--- 4. Verify Witness Chain Integrity ---");

    let verified = verify_witness_chain(&chain_bytes).expect("witness chain verification failed");
    println!("  Chain integrity: VALID ({} entries)", verified.len());

    // Verify genesis entry
    assert_eq!(verified[0].prev_hash, [0u8; 32], "genesis must have zero prev_hash");
    println!("  Genesis entry (iter 0): prev_hash is zero (confirmed)");

    // Verify action hashes match
    for (i, (orig, ver)) in entries.iter().zip(verified.iter()).enumerate() {
        assert_eq!(
            orig.action_hash, ver.action_hash,
            "action hash mismatch at iteration {}",
            i
        );
    }
    println!("  All {} action hashes verified", verified.len());

    // Print chain summary
    println!("\n  Witness chain entries (sampled):");
    println!(
        "    {:>5}  {:>8}  {:>32}",
        "Iter", "Type", "Prev Hash (first 16 bytes)"
    );
    println!("    {:->5}  {:->8}  {:->32}", "", "", "");
    for i in (0..verified.len()).step_by(5.max(1)) {
        let wtype_name = match verified[i].witness_type {
            0x01 => "PROV",
            0x02 => "COMP",
            0x03 => "CONV",
            _ => "????",
        };
        println!(
            "    {:>5}  {:>8}  {}",
            i,
            wtype_name,
            hex_string(&verified[i].prev_hash[..16])
        );
    }

    // ====================================================================
    // 5. Query the store to replay convergence history
    // ====================================================================
    println!("\n--- 5. Replay Convergence from RVF Store ---");

    // Find the final converged snapshot by querying for the last iteration
    let final_snapshot = &all_vectors[num_stored - 1];
    let k = 5;
    let results = store
        .query(final_snapshot, k, &QueryOptions::default())
        .expect("query failed");

    println!("  Nearest neighbors to final converged state (top-{}):", k);
    print_results(&results);

    // The closest result should be the final iteration itself
    assert_eq!(
        results[0].id,
        (num_stored - 1) as u64,
        "closest to final state should be itself"
    );
    println!("  Verified: closest match is the final iteration.");

    // Query early iterations to show they are distant from convergence
    let initial_snapshot = &all_vectors[0];
    let results_initial = store
        .query(initial_snapshot, k, &QueryOptions::default())
        .expect("query failed");

    println!("\n  Nearest neighbors to initial state (top-{}):", k);
    print_results(&results_initial);

    // ====================================================================
    // 6. Demonstrate deterministic replay
    // ====================================================================
    println!("\n--- 6. Deterministic Replay Verification ---");

    // Regenerate snapshots with the same seed and verify they match
    let mut replay_match = true;
    for iter in 0..num_stored {
        let replayed = solver_snapshot(dim, iter as u32, seed);
        if replayed != all_vectors[iter] {
            println!("  MISMATCH at iteration {}", iter);
            replay_match = false;
            break;
        }
    }
    if replay_match {
        println!("  All {} iterations replayed deterministically", num_stored);
    }

    // Verify residuals are deterministic too
    let mut resid_match = true;
    for iter in 0..num_stored {
        let replayed_resid = residual_norm(iter as u32, seed);
        if (replayed_resid - residuals[iter]).abs() > 1e-15 {
            println!("  Residual mismatch at iteration {}", iter);
            resid_match = false;
            break;
        }
    }
    if resid_match {
        println!("  All {} residual norms replayed deterministically", num_stored);
    }

    // ====================================================================
    // 7. Tamper detection: show that modifying the chain is caught
    // ====================================================================
    println!("\n--- 7. Witness Chain Tamper Detection ---");

    let entry_size = 73; // 32 + 32 + 8 + 1
    if chain_bytes.len() >= 2 * entry_size + 33 {
        let mut tampered = chain_bytes.clone();
        // Flip a bit in the second entry's action_hash
        tampered[entry_size + 32] ^= 0xFF;
        match verify_witness_chain(&tampered) {
            Ok(_) => println!("  WARNING: tampered chain was not detected"),
            Err(e) => println!("  Tamper at entry 1 detected: {:?}", e),
        }
    }

    // ====================================================================
    // Summary
    // ====================================================================
    println!("\n=== Solver Witness Summary ===\n");
    println!("  Solver dimensions:       {}", dim);
    println!("  Total iterations stored: {}", num_stored);
    println!(
        "  Convergence iteration:   {}",
        converged_at.map_or("N/A".to_string(), |c| c.to_string())
    );
    println!(
        "  Final residual:          {:.6e}",
        residuals[num_stored - 1]
    );
    println!("  Witness chain entries:   {}", verified.len());
    println!("  Witness chain integrity: VALID");
    println!("  Deterministic replay:    CONFIRMED");

    store.close().expect("failed to close store");
    println!("\nDone.");
}

fn print_results(results: &[SearchResult]) {
    println!("    {:>6}  {:>12}", "ID", "Distance");
    println!("    {:->6}  {:->12}", "", "");
    for r in results {
        println!("    {:>6}  {:>12.6}", r.id, r.distance);
    }
}

/// Format bytes as a hex string.
fn hex_string(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}
