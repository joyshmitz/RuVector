use std::time::{SystemTime, UNIX_EPOCH};

/// A point-in-time snapshot of memory usage.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemorySnapshot {
    pub peak_rss_bytes: u64,
    pub kv_cache_bytes: u64,
    pub activation_bytes: u64,
    pub temp_buffer_bytes: u64,
    pub timestamp_us: u64,
}

/// Aggregated memory statistics produced by [`MemoryTracker::report`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryReport {
    pub label: String,
    pub peak_rss: u64,
    pub mean_rss: u64,
    pub kv_cache_total: u64,
    pub activation_total: u64,
}

/// Captures a [`MemorySnapshot`] using OS-specific facilities.
///
/// On Linux this reads `VmRSS` from `/proc/self/status`.  On other
/// platforms a zero-valued fallback is returned so the crate still
/// compiles everywhere.
pub fn capture_memory() -> MemorySnapshot {
    let rss = read_vm_rss();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;

    MemorySnapshot {
        peak_rss_bytes: rss,
        kv_cache_bytes: 0,
        activation_bytes: 0,
        temp_buffer_bytes: 0,
        timestamp_us: ts,
    }
}

#[cfg(target_os = "linux")]
fn read_vm_rss() -> u64 {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let trimmed = rest.trim().trim_end_matches("kB").trim();
                if let Ok(kb) = trimmed.parse::<u64>() {
                    return kb * 1024;
                }
            }
        }
    }
    0
}

#[cfg(not(target_os = "linux"))]
fn read_vm_rss() -> u64 {
    0
}

/// Collects a series of [`MemorySnapshot`]s under a named label.
pub struct MemoryTracker {
    pub snapshots: Vec<MemorySnapshot>,
    pub label: String,
}

impl MemoryTracker {
    pub fn new(label: &str) -> Self {
        Self {
            snapshots: Vec::new(),
            label: label.to_string(),
        }
    }

    /// Take a snapshot and append it to the internal buffer.
    pub fn snapshot(&mut self) {
        self.snapshots.push(capture_memory());
    }

    /// Return the peak RSS across all recorded snapshots.
    pub fn peak(&self) -> u64 {
        self.snapshots.iter().map(|s| s.peak_rss_bytes).max().unwrap_or(0)
    }

    /// Produce an aggregated [`MemoryReport`].
    pub fn report(&self) -> MemoryReport {
        let n = self.snapshots.len() as u64;
        let peak_rss = self.peak();
        let mean_rss = if n > 0 {
            self.snapshots.iter().map(|s| s.peak_rss_bytes).sum::<u64>() / n
        } else {
            0
        };
        let kv_cache_total = self.snapshots.iter().map(|s| s.kv_cache_bytes).sum();
        let activation_total = self.snapshots.iter().map(|s| s.activation_bytes).sum();

        MemoryReport {
            label: self.label.clone(),
            peak_rss,
            mean_rss,
            kv_cache_total,
            activation_total,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_returns_nonzero_timestamp() {
        let snap = capture_memory();
        assert!(snap.timestamp_us > 0);
    }

    #[test]
    fn tracker_peak_empty() {
        let t = MemoryTracker::new("empty");
        assert_eq!(t.peak(), 0);
    }

    #[test]
    fn tracker_report_aggregates() {
        let mut t = MemoryTracker::new("test");
        t.snapshots.push(MemorySnapshot {
            peak_rss_bytes: 100,
            kv_cache_bytes: 10,
            activation_bytes: 20,
            temp_buffer_bytes: 5,
            timestamp_us: 1,
        });
        t.snapshots.push(MemorySnapshot {
            peak_rss_bytes: 200,
            kv_cache_bytes: 30,
            activation_bytes: 40,
            temp_buffer_bytes: 15,
            timestamp_us: 2,
        });
        let r = t.report();
        assert_eq!(r.peak_rss, 200);
        assert_eq!(r.mean_rss, 150);
        assert_eq!(r.kv_cache_total, 40);
        assert_eq!(r.activation_total, 60);
        assert_eq!(r.label, "test");
    }
}
