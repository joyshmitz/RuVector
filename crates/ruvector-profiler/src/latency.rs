/// A single latency measurement for one forward pass / kernel invocation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LatencyRecord {
    pub sample_id: usize,
    pub wall_time_us: u64,
    pub kernel_time_us: u64,
    pub seq_len: usize,
}

/// Descriptive statistics over a collection of latency records.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LatencyStats {
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub mean_us: f64,
    pub std_us: f64,
    pub n: usize,
}

/// Compute percentile and summary statistics from [`LatencyRecord`]s.
///
/// Uses `wall_time_us` for all calculations.  Returns zeroed stats when
/// the input slice is empty.
pub fn compute_latency_stats(records: &[LatencyRecord]) -> LatencyStats {
    let n = records.len();
    if n == 0 {
        return LatencyStats {
            p50_us: 0,
            p95_us: 0,
            p99_us: 0,
            mean_us: 0.0,
            std_us: 0.0,
            n: 0,
        };
    }

    let mut times: Vec<u64> = records.iter().map(|r| r.wall_time_us).collect();
    times.sort_unstable();

    let mean = times.iter().copied().sum::<u64>() as f64 / n as f64;
    let variance = times.iter().map(|&t| (t as f64 - mean).powi(2)).sum::<f64>() / n as f64;
    let std = variance.sqrt();

    LatencyStats {
        p50_us: percentile(&times, 50.0),
        p95_us: percentile(&times, 95.0),
        p99_us: percentile(&times, 99.0),
        mean_us: mean,
        std_us: std,
        n,
    }
}

/// Nearest-rank percentile on a **sorted** slice.
fn percentile(sorted: &[u64], pct: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (pct / 100.0 * sorted.len() as f64).ceil() as usize;
    let idx = rank.min(sorted.len()).saturating_sub(1);
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_records(times: &[u64]) -> Vec<LatencyRecord> {
        times
            .iter()
            .enumerate()
            .map(|(i, &t)| LatencyRecord {
                sample_id: i,
                wall_time_us: t,
                kernel_time_us: t,
                seq_len: 128,
            })
            .collect()
    }

    #[test]
    fn stats_empty() {
        let s = compute_latency_stats(&[]);
        assert_eq!(s.n, 0);
        assert_eq!(s.p50_us, 0);
    }

    #[test]
    fn stats_single() {
        let recs = make_records(&[42]);
        let s = compute_latency_stats(&recs);
        assert_eq!(s.n, 1);
        assert_eq!(s.p50_us, 42);
        assert_eq!(s.p99_us, 42);
        assert!((s.mean_us - 42.0).abs() < 1e-9);
        assert!((s.std_us).abs() < 1e-9);
    }

    #[test]
    fn stats_multiple() {
        let recs = make_records(&[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
        let s = compute_latency_stats(&recs);
        assert_eq!(s.n, 10);
        assert_eq!(s.p50_us, 50);
        assert!((s.mean_us - 55.0).abs() < 1e-9);
    }

    #[test]
    fn stats_unsorted_input() {
        let recs = make_records(&[100, 10, 50, 90, 20]);
        let s = compute_latency_stats(&recs);
        assert_eq!(s.p50_us, 50);
    }
}
