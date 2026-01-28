//! ReasoningBank - Adaptive Learning for Temporal Reasoning
//!
//! Implements trajectory tracking, verdict judgment, and strategy optimization
//! based on the lean-agentic design pattern.
//!
//! Key components:
//! - Trajectory tracking for solution attempts
//! - Verdict judgment (success/failure/suboptimal)
//! - Pattern learning from successful solutions
//! - Strategy optimization based on historical performance
//! - Confidence calibration from feedback

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Verdict for a solution trajectory
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Verdict {
    /// Solution was correct
    Success,
    /// Solution was acceptable but not optimal
    Acceptable,
    /// Solution was wrong but close
    Suboptimal { reason: String, delta: f64 },
    /// Low confidence in solution
    LowConfidence,
    /// Complete failure
    Failed,
}

impl Verdict {
    pub fn is_success(&self) -> bool {
        matches!(self, Verdict::Success | Verdict::Acceptable)
    }

    pub fn score(&self) -> f64 {
        match self {
            Verdict::Success => 1.0,
            Verdict::Acceptable => 0.8,
            Verdict::Suboptimal { delta, .. } => 0.5 - delta.min(0.3),
            Verdict::LowConfidence => 0.3,
            Verdict::Failed => 0.0,
        }
    }
}

/// A single solution attempt
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolutionAttempt {
    /// The proposed solution
    pub solution: String,
    /// Confidence in this solution
    pub confidence: f64,
    /// Steps taken to reach this solution
    pub steps: usize,
    /// Tool calls made
    pub tool_calls: usize,
    /// Strategy used
    pub strategy: String,
}

/// Trajectory tracking for a single puzzle
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trajectory {
    /// Puzzle identifier
    pub puzzle_id: String,
    /// Puzzle difficulty
    pub difficulty: u8,
    /// Constraint types encountered
    pub constraint_types: Vec<String>,
    /// Solution attempts made
    pub attempts: Vec<SolutionAttempt>,
    /// Final verdict
    pub verdict: Option<Verdict>,
    /// Correct solution (if known)
    pub correct_solution: Option<String>,
    /// Time taken in ms
    pub latency_ms: u64,
}

impl Trajectory {
    pub fn new(puzzle_id: &str, difficulty: u8) -> Self {
        Self {
            puzzle_id: puzzle_id.to_string(),
            difficulty,
            constraint_types: Vec::new(),
            attempts: Vec::new(),
            verdict: None,
            correct_solution: None,
            latency_ms: 0,
        }
    }

    pub fn record_attempt(
        &mut self,
        solution: String,
        confidence: f64,
        steps: usize,
        tool_calls: usize,
        strategy: &str,
    ) {
        self.attempts.push(SolutionAttempt {
            solution,
            confidence,
            steps,
            tool_calls,
            strategy: strategy.to_string(),
        });
    }

    pub fn set_verdict(&mut self, verdict: Verdict, correct: Option<String>) {
        self.verdict = Some(verdict);
        self.correct_solution = correct;
    }
}

/// Learned pattern from successful solutions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearnedPattern {
    /// Constraint type this pattern applies to
    pub constraint_type: String,
    /// Difficulty range
    pub difficulty_range: (u8, u8),
    /// Best strategy for this pattern
    pub best_strategy: String,
    /// Success rate with this pattern
    pub success_rate: f64,
    /// Average steps needed
    pub avg_steps: f64,
    /// Number of observations
    pub observations: usize,
}

/// Strategy configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Strategy {
    /// Strategy name
    pub name: String,
    /// Whether to use calendar rewriting
    pub use_rewriting: bool,
    /// Maximum search steps
    pub max_steps: usize,
    /// Beam width for search
    pub beam_width: usize,
    /// Confidence threshold
    pub confidence_threshold: f64,
}

impl Default for Strategy {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            use_rewriting: true,
            max_steps: 50,
            beam_width: 3,
            confidence_threshold: 0.7,
        }
    }
}

/// Confidence calibration data
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CalibrationData {
    /// Reported confidence -> actual accuracy mapping
    pub calibration_points: Vec<(f64, bool)>,
    /// Calibrated thresholds by difficulty
    pub thresholds: HashMap<u8, f64>,
}

impl CalibrationData {
    /// Record a calibration point
    pub fn record(&mut self, confidence: f64, correct: bool) {
        self.calibration_points.push((confidence, correct));
    }

    /// Get calibrated confidence threshold for a difficulty
    pub fn get_threshold(&self, difficulty: u8) -> f64 {
        self.thresholds.get(&difficulty).copied().unwrap_or(0.7)
    }

    /// Recalibrate thresholds based on observed data
    pub fn recalibrate(&mut self) {
        if self.calibration_points.len() < 20 {
            return;
        }

        // Group by confidence buckets
        let mut buckets: HashMap<u8, (usize, usize)> = HashMap::new();
        for &(conf, correct) in &self.calibration_points {
            let bucket = (conf * 10.0) as u8;
            let entry = buckets.entry(bucket).or_insert((0, 0));
            entry.0 += 1;
            if correct {
                entry.1 += 1;
            }
        }

        // Find threshold where precision >= 0.9
        for bucket in (5..=10).rev() {
            if let Some(&(total, correct)) = buckets.get(&bucket) {
                if total >= 5 {
                    let precision = correct as f64 / total as f64;
                    if precision >= 0.9 {
                        // Use this bucket's lower bound as default threshold
                        let threshold = bucket as f64 / 10.0;
                        for diff in 1..=10 {
                            self.thresholds.insert(diff, threshold);
                        }
                        break;
                    }
                }
            }
        }
    }
}

/// ReasoningBank - Central learning and adaptation system
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReasoningBank {
    /// All recorded trajectories
    pub trajectories: Vec<Trajectory>,
    /// Learned patterns by constraint type
    pub patterns: HashMap<String, Vec<LearnedPattern>>,
    /// Strategy performance by name
    pub strategy_stats: HashMap<String, StrategyStats>,
    /// Confidence calibration data
    pub calibration: CalibrationData,
    /// Current best strategy by difficulty
    pub best_strategies: HashMap<u8, String>,
    /// Pattern index for O(1) lookups: (constraint_type, difficulty) -> pattern_idx
    #[serde(skip)]
    pattern_index: HashMap<(String, u8), usize>,
    /// Constraint type frequency for prioritization
    #[serde(skip)]
    constraint_frequency: HashMap<String, usize>,
}

/// Statistics for a strategy
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StrategyStats {
    pub attempts: usize,
    pub successes: usize,
    pub total_steps: usize,
    pub total_latency_ms: u64,
}

impl StrategyStats {
    pub fn success_rate(&self) -> f64 {
        if self.attempts == 0 {
            return 0.5; // Prior
        }
        self.successes as f64 / self.attempts as f64
    }

    pub fn avg_steps(&self) -> f64 {
        if self.attempts == 0 {
            return 50.0;
        }
        self.total_steps as f64 / self.attempts as f64
    }
}

impl ReasoningBank {
    /// Create a new ReasoningBank
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed trajectory
    pub fn record_trajectory(&mut self, trajectory: Trajectory) {
        // Update strategy stats
        if let Some(attempt) = trajectory.attempts.first() {
            let stats = self
                .strategy_stats
                .entry(attempt.strategy.clone())
                .or_default();
            stats.attempts += 1;
            stats.total_steps += attempt.steps;
            stats.total_latency_ms += trajectory.latency_ms;

            if trajectory
                .verdict
                .as_ref()
                .map(|v| v.is_success())
                .unwrap_or(false)
            {
                stats.successes += 1;
            }
        }

        // Update calibration
        if let Some(attempt) = trajectory.attempts.first() {
            let correct = trajectory
                .verdict
                .as_ref()
                .map(|v| v.is_success())
                .unwrap_or(false);
            self.calibration.record(attempt.confidence, correct);
        }

        // Learn patterns from successful trajectories
        if trajectory
            .verdict
            .as_ref()
            .map(|v| v.is_success())
            .unwrap_or(false)
        {
            self.learn_from_success(&trajectory);
        }

        // Update best strategies
        self.update_best_strategies();

        // Store trajectory
        self.trajectories.push(trajectory);

        // Recalibrate periodically
        if self.trajectories.len() % 50 == 0 {
            self.calibration.recalibrate();
        }
    }

    /// Learn patterns from a successful trajectory
    fn learn_from_success(&mut self, trajectory: &Trajectory) {
        let attempt = match trajectory.attempts.first() {
            Some(a) => a,
            None => return,
        };

        for constraint_type in &trajectory.constraint_types {
            // Update constraint frequency
            *self
                .constraint_frequency
                .entry(constraint_type.clone())
                .or_insert(0) += 1;

            let patterns = self.patterns.entry(constraint_type.clone()).or_default();

            // Find or create pattern
            let pattern_idx = patterns.iter().position(|p| {
                p.best_strategy == attempt.strategy
                    && trajectory.difficulty >= p.difficulty_range.0
                    && trajectory.difficulty <= p.difficulty_range.1
            });

            if let Some(idx) = pattern_idx {
                // Update existing pattern
                let p = &mut patterns[idx];
                let n = p.observations as f64;
                p.success_rate = (p.success_rate * n + 1.0) / (n + 1.0);
                p.avg_steps = (p.avg_steps * n + attempt.steps as f64) / (n + 1.0);
                p.observations += 1;

                // Update pattern index for fast lookup
                self.pattern_index
                    .insert((constraint_type.clone(), trajectory.difficulty), idx);
            } else {
                // Create new pattern
                let new_idx = patterns.len();
                patterns.push(LearnedPattern {
                    constraint_type: constraint_type.clone(),
                    difficulty_range: (
                        trajectory.difficulty.saturating_sub(2),
                        trajectory.difficulty.saturating_add(2),
                    ),
                    best_strategy: attempt.strategy.clone(),
                    success_rate: 1.0,
                    avg_steps: attempt.steps as f64,
                    observations: 1,
                });

                // Index the new pattern
                for d in trajectory.difficulty.saturating_sub(2)
                    ..=trajectory.difficulty.saturating_add(2)
                {
                    self.pattern_index
                        .insert((constraint_type.clone(), d), new_idx);
                }
            }
        }
    }

    /// Record multiple trajectories in batch (for parallel processing)
    pub fn record_trajectories_batch(&mut self, trajectories: Vec<Trajectory>) {
        for trajectory in trajectories {
            self.record_trajectory(trajectory);
        }
    }

    /// Update best strategies by difficulty
    fn update_best_strategies(&mut self) {
        for difficulty in 1..=10 {
            let mut best_strategy = "default".to_string();
            let mut best_score = 0.0;

            for (strategy, stats) in &self.strategy_stats {
                // Score = success_rate - penalty for steps
                let score = stats.success_rate() - (stats.avg_steps() / 100.0);
                if score > best_score {
                    best_score = score;
                    best_strategy = strategy.clone();
                }
            }

            self.best_strategies.insert(difficulty, best_strategy);
        }
    }

    /// Get recommended strategy for a puzzle (optimized with index)
    pub fn get_strategy(&self, difficulty: u8, constraint_types: &[String]) -> Strategy {
        // Fast path: check pattern index first for O(1) lookup
        for ct in constraint_types {
            if let Some(&idx) = self.pattern_index.get(&(ct.clone(), difficulty)) {
                if let Some(patterns) = self.patterns.get(ct) {
                    if let Some(pattern) = patterns.get(idx) {
                        if pattern.success_rate > 0.7 && pattern.observations >= 3 {
                            return self.strategy_from_name(&pattern.best_strategy, difficulty);
                        }
                    }
                }
            }
        }

        // Slow path: linear search for patterns
        for ct in constraint_types {
            if let Some(patterns) = self.patterns.get(ct) {
                // Find best pattern for this difficulty
                let best = patterns
                    .iter()
                    .filter(|p| {
                        difficulty >= p.difficulty_range.0 && difficulty <= p.difficulty_range.1
                    })
                    .max_by(|a, b| a.success_rate.partial_cmp(&b.success_rate).unwrap());

                if let Some(pattern) = best {
                    if pattern.success_rate > 0.7 && pattern.observations >= 3 {
                        return self.strategy_from_name(&pattern.best_strategy, difficulty);
                    }
                }
            }
        }

        // Fall back to best strategy for difficulty
        let strategy_name = self
            .best_strategies
            .get(&difficulty)
            .cloned()
            .unwrap_or_else(|| "default".to_string());

        self.strategy_from_name(&strategy_name, difficulty)
    }

    fn strategy_from_name(&self, name: &str, difficulty: u8) -> Strategy {
        match name {
            "aggressive" => Strategy {
                name: "aggressive".to_string(),
                use_rewriting: true,
                max_steps: 30,
                beam_width: 5,
                confidence_threshold: 0.6,
            },
            "conservative" => Strategy {
                name: "conservative".to_string(),
                use_rewriting: true,
                max_steps: 100,
                beam_width: 2,
                confidence_threshold: 0.85,
            },
            "adaptive" => Strategy {
                name: "adaptive".to_string(),
                use_rewriting: true,
                max_steps: 50 + (difficulty as usize * 5),
                beam_width: 3,
                confidence_threshold: self.calibration.get_threshold(difficulty),
            },
            _ => Strategy::default(),
        }
    }

    /// Get hints for a puzzle based on learned patterns
    pub fn get_hints(&self, constraint_types: &[String]) -> Vec<String> {
        let mut hints = Vec::new();

        for ct in constraint_types {
            if let Some(patterns) = self.patterns.get(ct) {
                for pattern in patterns.iter().filter(|p| p.observations >= 5) {
                    hints.push(format!(
                        "For {} constraints, {} strategy has {:.0}% success",
                        ct,
                        pattern.best_strategy,
                        pattern.success_rate * 100.0
                    ));
                }
            }
        }

        hints
    }

    /// Calculate learning progress metrics
    pub fn learning_progress(&self) -> LearningProgress {
        let total = self.trajectories.len();
        if total == 0 {
            return LearningProgress::default();
        }

        let successes = self
            .trajectories
            .iter()
            .filter(|t| t.verdict.as_ref().map(|v| v.is_success()).unwrap_or(false))
            .count();

        // Calculate improvement over time (compare first half vs second half)
        let half = total / 2;
        let first_half_success = self.trajectories[..half]
            .iter()
            .filter(|t| t.verdict.as_ref().map(|v| v.is_success()).unwrap_or(false))
            .count() as f64
            / half as f64;

        let second_half_success = self.trajectories[half..]
            .iter()
            .filter(|t| t.verdict.as_ref().map(|v| v.is_success()).unwrap_or(false))
            .count() as f64
            / (total - half) as f64;

        let improvement = second_half_success - first_half_success;

        // Calculate pattern coverage
        let unique_patterns: usize = self.patterns.values().map(|ps| ps.len()).sum();

        LearningProgress {
            total_trajectories: total,
            success_rate: successes as f64 / total as f64,
            improvement_rate: improvement,
            patterns_learned: unique_patterns,
            strategies_tried: self.strategy_stats.len(),
            is_improving: improvement > 0.0,
        }
    }
}

/// Learning progress summary
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LearningProgress {
    pub total_trajectories: usize,
    pub success_rate: f64,
    pub improvement_rate: f64,
    pub patterns_learned: usize,
    pub strategies_tried: usize,
    pub is_improving: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoning_bank_learning() {
        let mut bank = ReasoningBank::new();

        // Record some successful trajectories
        for i in 0..10 {
            let mut traj = Trajectory::new(&format!("puzzle_{}", i), 5);
            traj.constraint_types.push("RelativeDate".to_string());
            traj.record_attempt("2024-01-15".to_string(), 0.8, 20, 5, "adaptive");
            traj.set_verdict(Verdict::Success, Some("2024-01-15".to_string()));
            traj.latency_ms = 100;
            bank.record_trajectory(traj);
        }

        // Should have learned patterns
        assert!(bank.patterns.contains_key("RelativeDate"));
        assert!(bank.strategy_stats.contains_key("adaptive"));

        let stats = &bank.strategy_stats["adaptive"];
        assert_eq!(stats.successes, 10);
        assert!(stats.success_rate() > 0.9);
    }

    #[test]
    fn test_strategy_selection() {
        let mut bank = ReasoningBank::new();

        // Train on aggressive strategy for easy puzzles
        for i in 0..20 {
            let mut traj = Trajectory::new(&format!("easy_{}", i), 3);
            traj.constraint_types.push("Before".to_string());
            traj.record_attempt("2024-01-10".to_string(), 0.9, 10, 2, "aggressive");
            traj.set_verdict(Verdict::Success, Some("2024-01-10".to_string()));
            bank.record_trajectory(traj);
        }

        // Should recommend aggressive for easy puzzles
        let strategy = bank.get_strategy(3, &["Before".to_string()]);
        assert_eq!(strategy.name, "aggressive");
    }
}
