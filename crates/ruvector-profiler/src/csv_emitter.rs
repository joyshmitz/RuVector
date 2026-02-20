use crate::latency::LatencyRecord;
use crate::memory::MemorySnapshot;
use std::io::Write;

/// One row of the aggregated benchmark results CSV.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResultRow {
    pub setting: String,
    pub coherence_delta: f64,
    pub kv_cache_reduction: f64,
    pub peak_mem_reduction: f64,
    pub energy_reduction: f64,
    pub p95_latency_us: u64,
    pub accuracy: f64,
}

/// Write aggregated benchmark results to a CSV file.
pub fn write_results_csv(path: &str, rows: &[ResultRow]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(
        f,
        "setting,coherence_delta,kv_cache_reduction,peak_mem_reduction,energy_reduction,p95_latency_us,accuracy"
    )?;
    for r in rows {
        writeln!(
            f,
            "{},{},{},{},{},{},{}",
            escape_csv(&r.setting),
            r.coherence_delta,
            r.kv_cache_reduction,
            r.peak_mem_reduction,
            r.energy_reduction,
            r.p95_latency_us,
            r.accuracy,
        )?;
    }
    Ok(())
}

/// Write raw latency records to a CSV file.
pub fn write_latency_csv(path: &str, records: &[LatencyRecord]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "sample_id,wall_time_us,kernel_time_us,seq_len")?;
    for r in records {
        writeln!(
            f,
            "{},{},{},{}",
            r.sample_id, r.wall_time_us, r.kernel_time_us, r.seq_len,
        )?;
    }
    Ok(())
}

/// Write memory snapshots to a CSV file.
pub fn write_memory_csv(path: &str, snapshots: &[MemorySnapshot]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(
        f,
        "timestamp_us,peak_rss_bytes,kv_cache_bytes,activation_bytes,temp_buffer_bytes"
    )?;
    for s in snapshots {
        writeln!(
            f,
            "{},{},{},{},{}",
            s.timestamp_us,
            s.peak_rss_bytes,
            s.kv_cache_bytes,
            s.activation_bytes,
            s.temp_buffer_bytes,
        )?;
    }
    Ok(())
}

/// Minimal CSV escaping: wrap in quotes if the value contains a comma or quote.
fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_plain() {
        assert_eq!(escape_csv("hello"), "hello");
    }

    #[test]
    fn escape_comma() {
        assert_eq!(escape_csv("a,b"), "\"a,b\"");
    }

    #[test]
    fn escape_quote() {
        assert_eq!(escape_csv("say \"hi\""), "\"say \"\"hi\"\"\"");
    }

    #[test]
    fn write_results_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.csv");
        let path_str = path.to_str().unwrap();

        let rows = vec![
            ResultRow {
                setting: "baseline".into(),
                coherence_delta: 0.01,
                kv_cache_reduction: 0.0,
                peak_mem_reduction: 0.0,
                energy_reduction: 0.0,
                p95_latency_us: 1200,
                accuracy: 0.95,
            },
            ResultRow {
                setting: "lambda=0.1".into(),
                coherence_delta: -0.03,
                kv_cache_reduction: 0.45,
                peak_mem_reduction: 0.30,
                energy_reduction: 0.25,
                p95_latency_us: 950,
                accuracy: 0.93,
            },
        ];
        write_results_csv(path_str, &rows).unwrap();
        let content = std::fs::read_to_string(path_str).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3); // header + 2 data rows
        assert!(lines[0].starts_with("setting,"));
        assert!(lines[1].starts_with("baseline,"));
    }

    #[test]
    fn write_latency_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("latency.csv");
        let path_str = path.to_str().unwrap();

        let records = vec![
            LatencyRecord { sample_id: 0, wall_time_us: 100, kernel_time_us: 80, seq_len: 64 },
            LatencyRecord { sample_id: 1, wall_time_us: 120, kernel_time_us: 90, seq_len: 128 },
        ];
        write_latency_csv(path_str, &records).unwrap();
        let content = std::fs::read_to_string(path_str).unwrap();
        assert_eq!(content.lines().count(), 3);
    }

    #[test]
    fn write_memory_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memory.csv");
        let path_str = path.to_str().unwrap();

        let snaps = vec![MemorySnapshot {
            peak_rss_bytes: 1024,
            kv_cache_bytes: 256,
            activation_bytes: 512,
            temp_buffer_bytes: 128,
            timestamp_us: 999,
        }];
        write_memory_csv(path_str, &snaps).unwrap();
        let content = std::fs::read_to_string(path_str).unwrap();
        assert_eq!(content.lines().count(), 2);
        assert!(content.contains("999,1024,256,512,128"));
    }
}
