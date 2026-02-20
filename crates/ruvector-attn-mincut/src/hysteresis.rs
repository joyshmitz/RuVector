/// Temporal hysteresis tracker for stable gating decisions.
///
/// An edge's gating state only flips after the new decision has been
/// consistent for `tau` consecutive steps, preventing oscillation.
#[derive(Debug, Clone)]
pub struct HysteresisTracker {
    /// Previous stabilised mask (None on first step).
    prev_mask: Option<Vec<bool>>,
    /// Number of consecutive steps each edge has had a *different* decision
    /// from `prev_mask`. When `counts[i] >= tau` the edge flips.
    counts: Vec<usize>,
    /// Hysteresis window size.
    tau: usize,
    /// Current time step.
    step: usize,
}

impl HysteresisTracker {
    /// Create a new tracker with the given hysteresis window.
    pub fn new(tau: usize) -> Self {
        Self {
            prev_mask: None,
            counts: Vec::new(),
            tau,
            step: 0,
        }
    }

    /// Apply hysteresis to a raw gating mask and return the stabilised mask.
    ///
    /// On the first call the raw mask is accepted as-is. On subsequent calls
    /// an edge only flips if the raw decision has disagreed with the current
    /// stable state for `tau` consecutive steps.
    pub fn apply(&mut self, raw_mask: &[bool]) -> Vec<bool> {
        self.step += 1;

        let stable = match &self.prev_mask {
            None => {
                // First step -- accept raw mask directly
                self.counts = vec![0; raw_mask.len()];
                self.prev_mask = Some(raw_mask.to_vec());
                return raw_mask.to_vec();
            }
            Some(prev) => prev.clone(),
        };

        // Resize counts if mask length changed (sequence length change)
        if self.counts.len() != raw_mask.len() {
            self.counts = vec![0; raw_mask.len()];
            self.prev_mask = Some(raw_mask.to_vec());
            return raw_mask.to_vec();
        }

        let mut result = stable.clone();

        for i in 0..raw_mask.len() {
            if raw_mask[i] != stable[i] {
                self.counts[i] += 1;
                if self.counts[i] >= self.tau {
                    // Flip the edge
                    result[i] = raw_mask[i];
                    self.counts[i] = 0;
                }
            } else {
                // Decision agrees with stable state -- reset counter
                self.counts[i] = 0;
            }
        }

        self.prev_mask = Some(result.clone());
        result
    }

    /// Current time step.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Read-only access to the current stable mask (None before first call).
    pub fn current_mask(&self) -> Option<&[bool]> {
        self.prev_mask.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_step_passthrough() {
        let mut tracker = HysteresisTracker::new(3);
        let mask = vec![true, false, true];
        let out = tracker.apply(&mask);
        assert_eq!(out, mask);
        assert_eq!(tracker.step(), 1);
    }

    #[test]
    fn test_no_flip_before_tau() {
        let mut tracker = HysteresisTracker::new(3);
        let initial = vec![true, true, false];
        tracker.apply(&initial);

        // Present a different mask for only 2 steps (< tau=3)
        let changed = vec![false, true, true];
        let out1 = tracker.apply(&changed);
        assert_eq!(out1, initial, "should not flip after 1 disagreement");

        let out2 = tracker.apply(&changed);
        assert_eq!(out2, initial, "should not flip after 2 disagreements");
    }

    #[test]
    fn test_flip_at_tau() {
        let mut tracker = HysteresisTracker::new(2);
        let initial = vec![true, false];
        tracker.apply(&initial);

        let changed = vec![false, true];
        tracker.apply(&changed); // count = 1
        let out = tracker.apply(&changed); // count = 2 >= tau -> flip
        assert_eq!(out, changed);
    }

    #[test]
    fn test_counter_reset_on_agreement() {
        let mut tracker = HysteresisTracker::new(3);
        let initial = vec![true];
        tracker.apply(&initial);

        // Disagree once
        tracker.apply(&vec![false]);
        // Then agree again -- counter resets
        tracker.apply(&vec![true]);
        // Disagree twice more -- should not flip (total non-consecutive = 3, but reset in between)
        tracker.apply(&vec![false]);
        let out = tracker.apply(&vec![false]);
        // Only 2 consecutive disagreements, need 3
        assert_eq!(out, vec![true]);
    }

    #[test]
    fn test_resize_on_length_change() {
        let mut tracker = HysteresisTracker::new(2);
        tracker.apply(&vec![true, false]);
        // Different length -- resets
        let out = tracker.apply(&vec![true, false, true]);
        assert_eq!(out.len(), 3);
    }
}
