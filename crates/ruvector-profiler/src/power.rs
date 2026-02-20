/// A single power measurement sample.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PowerSample {
    pub watts: f64,
    pub timestamp_us: u64,
}

/// Result of integrating power samples over time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnergyResult {
    pub total_joules: f64,
    pub mean_watts: f64,
    pub peak_watts: f64,
    pub duration_s: f64,
    pub samples: usize,
}

/// Trait for reading instantaneous power from a hardware source.
///
/// Real implementations would wrap NVML (GPU) or RAPL (CPU).  Use
/// [`MockPowerSource`] for deterministic tests.
pub trait PowerSource {
    fn read_watts(&self) -> f64;
}

/// A mock power source that returns a fixed wattage.
pub struct MockPowerSource {
    pub watts: f64,
}

impl PowerSource for MockPowerSource {
    fn read_watts(&self) -> f64 {
        self.watts
    }
}

/// Estimate total energy consumption via trapezoidal integration.
///
/// Samples must be sorted by `timestamp_us`.  Returns a zeroed result
/// when fewer than two samples are provided.
pub fn estimate_energy(samples: &[PowerSample]) -> EnergyResult {
    let n = samples.len();
    if n < 2 {
        return EnergyResult {
            total_joules: 0.0,
            mean_watts: samples.first().map_or(0.0, |s| s.watts),
            peak_watts: samples.first().map_or(0.0, |s| s.watts),
            duration_s: 0.0,
            samples: n,
        };
    }

    let mut total_joules = 0.0;
    let mut peak_watts: f64 = f64::NEG_INFINITY;
    let mut sum_watts = 0.0;

    for i in 0..n {
        let w = samples[i].watts;
        sum_watts += w;
        if w > peak_watts {
            peak_watts = w;
        }
        if i > 0 {
            let dt_us = samples[i].timestamp_us.saturating_sub(samples[i - 1].timestamp_us);
            let dt_s = dt_us as f64 / 1_000_000.0;
            let avg_w = (samples[i - 1].watts + samples[i].watts) / 2.0;
            total_joules += avg_w * dt_s;
        }
    }

    let duration_us =
        samples.last().unwrap().timestamp_us.saturating_sub(samples.first().unwrap().timestamp_us);
    let duration_s = duration_us as f64 / 1_000_000.0;
    let mean_watts = sum_watts / n as f64;

    EnergyResult {
        total_joules,
        mean_watts,
        peak_watts,
        duration_s,
        samples: n,
    }
}

/// Collects [`PowerSample`]s under a named label.
pub struct PowerTracker {
    pub samples: Vec<PowerSample>,
    pub label: String,
}

impl PowerTracker {
    pub fn new(label: &str) -> Self {
        Self {
            samples: Vec::new(),
            label: label.to_string(),
        }
    }

    /// Record a sample from a [`PowerSource`].
    pub fn sample(&mut self, source: &dyn PowerSource) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.samples.push(PowerSample {
            watts: source.read_watts(),
            timestamp_us: ts,
        });
    }

    /// Integrate all collected samples.
    pub fn energy(&self) -> EnergyResult {
        estimate_energy(&self.samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_empty() {
        let r = estimate_energy(&[]);
        assert_eq!(r.total_joules, 0.0);
        assert_eq!(r.samples, 0);
    }

    #[test]
    fn energy_single_sample() {
        let r = estimate_energy(&[PowerSample { watts: 42.0, timestamp_us: 100 }]);
        assert_eq!(r.total_joules, 0.0);
        assert_eq!(r.mean_watts, 42.0);
        assert_eq!(r.samples, 1);
    }

    #[test]
    fn energy_trapezoidal() {
        // 100W constant for 1 second = 100 J
        let samples = vec![
            PowerSample { watts: 100.0, timestamp_us: 0 },
            PowerSample { watts: 100.0, timestamp_us: 1_000_000 },
        ];
        let r = estimate_energy(&samples);
        assert!((r.total_joules - 100.0).abs() < 1e-9);
        assert!((r.duration_s - 1.0).abs() < 1e-9);
        assert_eq!(r.peak_watts, 100.0);
    }

    #[test]
    fn energy_trapezoid_varying() {
        // Ramp from 0W to 200W over 1 second -> average 100W -> 100 J
        let samples = vec![
            PowerSample { watts: 0.0, timestamp_us: 0 },
            PowerSample { watts: 200.0, timestamp_us: 1_000_000 },
        ];
        let r = estimate_energy(&samples);
        assert!((r.total_joules - 100.0).abs() < 1e-9);
    }

    #[test]
    fn mock_power_source() {
        let src = MockPowerSource { watts: 75.0 };
        assert_eq!(src.read_watts(), 75.0);
    }

    #[test]
    fn power_tracker_collects() {
        let src = MockPowerSource { watts: 50.0 };
        let mut tracker = PowerTracker::new("gpu");
        tracker.sample(&src);
        tracker.sample(&src);
        assert_eq!(tracker.samples.len(), 2);
        assert_eq!(tracker.label, "gpu");
    }
}
