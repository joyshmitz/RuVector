//! Temporal Reasoning Benchmark Framework
//!
//! Implements temporal constraint solving and benchmarking based on:
//! - TimePuzzles benchmark methodology
//! - Tool-augmented iterative temporal reasoning
//! - Calendar math and cross-cultural date systems

use anyhow::{anyhow, Result};
use chrono::{Datelike, NaiveDate, Weekday};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Temporal constraint types
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TemporalConstraint {
    /// Date is exactly this value
    Exact(NaiveDate),
    /// Date is after this date
    After(NaiveDate),
    /// Date is before this date
    Before(NaiveDate),
    /// Date is between two dates (inclusive)
    Between(NaiveDate, NaiveDate),
    /// Date is on a specific day of week
    DayOfWeek(Weekday),
    /// Date is N days after reference
    DaysAfter(String, i64),
    /// Date is N days before reference
    DaysBefore(String, i64),
    /// Date is in a specific month
    InMonth(u32),
    /// Date is in a specific year
    InYear(i32),
    /// Date is a specific day of month
    DayOfMonth(u32),
    /// Relative to a named event (e.g., "Easter", "Chinese New Year")
    RelativeToEvent(String, i64),
}

/// A temporal puzzle with constraints
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalPuzzle {
    /// Unique puzzle ID
    pub id: String,
    /// Human-readable description
    pub description: String,
    /// Constraints that define the puzzle
    pub constraints: Vec<TemporalConstraint>,
    /// Named reference dates
    pub references: HashMap<String, NaiveDate>,
    /// Valid solution dates (for evaluation)
    pub solutions: Vec<NaiveDate>,
    /// Difficulty level (1-10)
    pub difficulty: u8,
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl TemporalPuzzle {
    /// Create a new puzzle
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            constraints: Vec::new(),
            references: HashMap::new(),
            solutions: Vec::new(),
            difficulty: 5,
            tags: Vec::new(),
        }
    }

    /// Add a constraint
    pub fn with_constraint(mut self, constraint: TemporalConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Add a reference date
    pub fn with_reference(mut self, name: impl Into<String>, date: NaiveDate) -> Self {
        self.references.insert(name.into(), date);
        self
    }

    /// Set solution dates
    pub fn with_solutions(mut self, solutions: Vec<NaiveDate>) -> Self {
        self.solutions = solutions;
        self
    }

    /// Set difficulty
    pub fn with_difficulty(mut self, difficulty: u8) -> Self {
        self.difficulty = difficulty.min(10).max(1);
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Check if a date satisfies all constraints
    pub fn check_date(&self, date: NaiveDate) -> Result<bool> {
        for constraint in &self.constraints {
            if !self.check_constraint(date, constraint)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Check a single constraint
    fn check_constraint(&self, date: NaiveDate, constraint: &TemporalConstraint) -> Result<bool> {
        match constraint {
            TemporalConstraint::Exact(d) => Ok(date == *d),
            TemporalConstraint::After(d) => Ok(date > *d),
            TemporalConstraint::Before(d) => Ok(date < *d),
            TemporalConstraint::Between(start, end) => Ok(date >= *start && date <= *end),
            TemporalConstraint::DayOfWeek(dow) => Ok(date.weekday() == *dow),
            TemporalConstraint::DaysAfter(ref_name, days) => {
                let ref_date = self
                    .references
                    .get(ref_name)
                    .ok_or_else(|| anyhow!("Unknown reference: {}", ref_name))?;
                let target = *ref_date + chrono::Duration::days(*days);
                Ok(date == target)
            }
            TemporalConstraint::DaysBefore(ref_name, days) => {
                let ref_date = self
                    .references
                    .get(ref_name)
                    .ok_or_else(|| anyhow!("Unknown reference: {}", ref_name))?;
                let target = *ref_date - chrono::Duration::days(*days);
                Ok(date == target)
            }
            TemporalConstraint::InMonth(month) => Ok(date.month() == *month),
            TemporalConstraint::InYear(year) => Ok(date.year() == *year),
            TemporalConstraint::DayOfMonth(day) => Ok(date.day() == *day),
            TemporalConstraint::RelativeToEvent(event_name, days) => {
                // Look up event in references
                let event_date = self
                    .references
                    .get(event_name)
                    .ok_or_else(|| anyhow!("Unknown event: {}", event_name))?;
                let target = *event_date + chrono::Duration::days(*days);
                Ok(date == target)
            }
        }
    }

    /// Solve the puzzle by searching date space
    pub fn solve(&self, search_range: (NaiveDate, NaiveDate)) -> Result<Vec<NaiveDate>> {
        let mut solutions = Vec::new();
        let mut current = search_range.0;
        while current <= search_range.1 {
            if self.check_date(current)? {
                solutions.push(current);
            }
            current = current.succ_opt().unwrap_or(current);
        }
        Ok(solutions)
    }
}

/// Puzzle solver with tool augmentation
#[derive(Clone, Debug)]
pub struct TemporalSolver {
    /// Enable calendar math tool
    pub calendar_tool: bool,
    /// Enable web search tool
    pub web_search_tool: bool,
    /// Maximum steps allowed
    pub max_steps: usize,
    /// Current step count
    pub steps: usize,
    /// Tool call count
    pub tool_calls: usize,
}

impl Default for TemporalSolver {
    fn default() -> Self {
        Self {
            calendar_tool: true,
            web_search_tool: false,
            max_steps: 100,
            steps: 0,
            tool_calls: 0,
        }
    }
}

impl TemporalSolver {
    /// Create solver with tools
    pub fn with_tools(calendar: bool, web_search: bool) -> Self {
        Self {
            calendar_tool: calendar,
            web_search_tool: web_search,
            ..Default::default()
        }
    }

    /// Solve a puzzle with step tracking
    pub fn solve(&mut self, puzzle: &TemporalPuzzle) -> Result<SolverResult> {
        self.steps = 0;
        self.tool_calls = 0;

        let start_time = std::time::Instant::now();

        // Rewrite constraints to explicit dates if calendar tool enabled
        let effective_puzzle = if self.calendar_tool {
            self.tool_calls += 1;
            self.rewrite_constraints(puzzle)?
        } else {
            puzzle.clone()
        };

        // Determine search range from effective (rewritten) constraints
        let range = self.determine_search_range(&effective_puzzle)?;

        // Search for solutions
        let mut found_solutions = Vec::new();
        let mut current = range.0;

        while current <= range.1 && self.steps < self.max_steps {
            self.steps += 1;
            if effective_puzzle.check_date(current)? {
                found_solutions.push(current);
            }
            current = match current.succ_opt() {
                Some(d) => d,
                None => break,
            };
        }

        let latency = start_time.elapsed();

        // Check correctness
        let correct = if puzzle.solutions.is_empty() {
            true // No ground truth
        } else {
            found_solutions.iter().all(|s| puzzle.solutions.contains(s))
                && puzzle
                    .solutions
                    .iter()
                    .all(|s| found_solutions.contains(s) || *s < range.0 || *s > range.1)
        };

        Ok(SolverResult {
            puzzle_id: puzzle.id.clone(),
            solved: !found_solutions.is_empty(),
            correct,
            solutions: found_solutions,
            steps: self.steps,
            tool_calls: self.tool_calls,
            latency_ms: latency.as_millis() as u64,
        })
    }

    /// Determine search range from constraints
    fn determine_search_range(&self, puzzle: &TemporalPuzzle) -> Result<(NaiveDate, NaiveDate)> {
        let mut min_date = NaiveDate::from_ymd_opt(1900, 1, 1).unwrap();
        let mut max_date = NaiveDate::from_ymd_opt(2100, 12, 31).unwrap();

        for constraint in &puzzle.constraints {
            match constraint {
                TemporalConstraint::Exact(d) => {
                    min_date = *d;
                    max_date = *d;
                }
                TemporalConstraint::After(d) => {
                    if *d >= min_date {
                        min_date = d.succ_opt().unwrap_or(*d);
                    }
                }
                TemporalConstraint::Before(d) => {
                    if *d <= max_date {
                        max_date = d.pred_opt().unwrap_or(*d);
                    }
                }
                TemporalConstraint::Between(start, end) => {
                    if *start > min_date {
                        min_date = *start;
                    }
                    if *end < max_date {
                        max_date = *end;
                    }
                }
                TemporalConstraint::InYear(year) => {
                    let year_start = NaiveDate::from_ymd_opt(*year, 1, 1).unwrap_or(min_date);
                    let year_end = NaiveDate::from_ymd_opt(*year, 12, 31).unwrap_or(max_date);
                    if year_start > min_date {
                        min_date = year_start;
                    }
                    if year_end < max_date {
                        max_date = year_end;
                    }
                }
                _ => {}
            }
        }

        Ok((min_date, max_date))
    }

    /// Rewrite relative constraints to explicit dates
    fn rewrite_constraints(&self, puzzle: &TemporalPuzzle) -> Result<TemporalPuzzle> {
        let mut new_puzzle = puzzle.clone();
        let mut new_constraints = Vec::new();

        for constraint in &puzzle.constraints {
            match constraint {
                TemporalConstraint::DaysAfter(ref_name, days) => {
                    if let Some(ref_date) = puzzle.references.get(ref_name) {
                        let target = *ref_date + chrono::Duration::days(*days);
                        new_constraints.push(TemporalConstraint::Exact(target));
                    } else {
                        new_constraints.push(constraint.clone());
                    }
                }
                TemporalConstraint::DaysBefore(ref_name, days) => {
                    if let Some(ref_date) = puzzle.references.get(ref_name) {
                        let target = *ref_date - chrono::Duration::days(*days);
                        new_constraints.push(TemporalConstraint::Exact(target));
                    } else {
                        new_constraints.push(constraint.clone());
                    }
                }
                TemporalConstraint::RelativeToEvent(event_name, days) => {
                    if let Some(event_date) = puzzle.references.get(event_name) {
                        let target = *event_date + chrono::Duration::days(*days);
                        new_constraints.push(TemporalConstraint::Exact(target));
                    } else {
                        new_constraints.push(constraint.clone());
                    }
                }
                _ => new_constraints.push(constraint.clone()),
            }
        }

        new_puzzle.constraints = new_constraints;
        Ok(new_puzzle)
    }
}

/// Result from solving a puzzle
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolverResult {
    pub puzzle_id: String,
    pub solved: bool,
    pub correct: bool,
    pub solutions: Vec<NaiveDate>,
    pub steps: usize,
    pub tool_calls: usize,
    pub latency_ms: u64,
}

/// Benchmark configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of puzzles to run
    pub num_puzzles: usize,
    /// Difficulty range
    pub difficulty_range: (u8, u8),
    /// Enable calendar tool
    pub calendar_tool: bool,
    /// Enable web search tool
    pub web_search_tool: bool,
    /// Maximum steps per puzzle
    pub max_steps: usize,
    /// Constraint density (1-5)
    pub constraint_density: u8,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_puzzles: 50,
            difficulty_range: (1, 10),
            calendar_tool: true,
            web_search_tool: false,
            max_steps: 100,
            constraint_density: 3,
        }
    }
}

/// Benchmark results
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub config: BenchmarkConfig,
    pub total_puzzles: usize,
    pub solved_count: usize,
    pub correct_count: usize,
    pub accuracy: f64,
    pub avg_steps: f64,
    pub avg_tool_calls: f64,
    pub avg_latency_ms: f64,
    pub results: Vec<SolverResult>,
}

impl BenchmarkResults {
    /// Create from individual results
    pub fn from_results(config: BenchmarkConfig, results: Vec<SolverResult>) -> Self {
        let total = results.len();
        let solved = results.iter().filter(|r| r.solved).count();
        let correct = results.iter().filter(|r| r.correct).count();
        let avg_steps = results.iter().map(|r| r.steps as f64).sum::<f64>() / total as f64;
        let avg_tools = results.iter().map(|r| r.tool_calls as f64).sum::<f64>() / total as f64;
        let avg_latency = results.iter().map(|r| r.latency_ms as f64).sum::<f64>() / total as f64;

        Self {
            config,
            total_puzzles: total,
            solved_count: solved,
            correct_count: correct,
            accuracy: correct as f64 / total as f64,
            avg_steps,
            avg_tool_calls: avg_tools,
            avg_latency_ms: avg_latency,
            results,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_puzzle() {
        let puzzle = TemporalPuzzle::new("test-1", "Find a date in January 2024")
            .with_constraint(TemporalConstraint::InYear(2024))
            .with_constraint(TemporalConstraint::InMonth(1))
            .with_constraint(TemporalConstraint::DayOfMonth(15));

        let expected = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        assert!(puzzle.check_date(expected).unwrap());
        assert!(!puzzle
            .check_date(NaiveDate::from_ymd_opt(2024, 2, 15).unwrap())
            .unwrap());
    }

    #[test]
    fn test_relative_constraint() {
        let base = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let puzzle = TemporalPuzzle::new("test-2", "Find a date 10 days after New Year")
            .with_reference("new_year", base)
            .with_constraint(TemporalConstraint::DaysAfter("new_year".to_string(), 10));

        let expected = NaiveDate::from_ymd_opt(2024, 1, 11).unwrap();
        assert!(puzzle.check_date(expected).unwrap());
    }

    #[test]
    fn test_solver_with_rewriting() {
        let base = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap();
        let puzzle = TemporalPuzzle::new("test-3", "Find date relative to event")
            .with_reference("event", base)
            .with_constraint(TemporalConstraint::DaysAfter("event".to_string(), 5))
            .with_solutions(vec![NaiveDate::from_ymd_opt(2024, 6, 20).unwrap()]);

        let mut solver = TemporalSolver::with_tools(true, false);
        let result = solver.solve(&puzzle).unwrap();

        assert!(result.solved);
        assert!(result.correct);
        assert_eq!(result.solutions.len(), 1);
    }
}

// ============================================================================
// Adaptive Solver with ReasoningBank Learning
// ============================================================================

use crate::reasoning_bank::{ReasoningBank, Strategy, Trajectory, Verdict};

/// Adaptive temporal solver with learning capabilities
///
/// Uses ReasoningBank to:
/// - Track solution trajectories
/// - Learn from successes and failures
/// - Adapt strategy based on puzzle characteristics
/// - Achieve sublinear regret through experience
#[derive(Clone, Debug)]
pub struct AdaptiveSolver {
    /// Internal solver
    solver: TemporalSolver,
    /// ReasoningBank for learning
    pub reasoning_bank: ReasoningBank,
    /// Current strategy
    current_strategy: Strategy,
    /// Total episodes completed
    pub episodes: usize,
}

impl Default for AdaptiveSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveSolver {
    /// Create a new adaptive solver
    pub fn new() -> Self {
        Self {
            solver: TemporalSolver::default(),
            reasoning_bank: ReasoningBank::new(),
            current_strategy: Strategy::default(),
            episodes: 0,
        }
    }

    /// Create with pre-trained ReasoningBank
    pub fn with_reasoning_bank(reasoning_bank: ReasoningBank) -> Self {
        Self {
            solver: TemporalSolver::default(),
            reasoning_bank,
            current_strategy: Strategy::default(),
            episodes: 0,
        }
    }

    /// Solve a puzzle with adaptive learning
    pub fn solve(&mut self, puzzle: &TemporalPuzzle) -> Result<SolverResult> {
        // Get constraint types for pattern matching
        let constraint_types: Vec<String> = puzzle
            .constraints
            .iter()
            .map(|c| constraint_type_name(c))
            .collect();

        // Get recommended strategy from ReasoningBank
        self.current_strategy = self
            .reasoning_bank
            .get_strategy(puzzle.difficulty, &constraint_types);

        // Configure solver based on strategy
        self.solver.calendar_tool = self.current_strategy.use_rewriting;
        self.solver.max_steps = self.current_strategy.max_steps;

        // Create trajectory for this puzzle
        let mut trajectory = Trajectory::new(&puzzle.id, puzzle.difficulty);
        trajectory.constraint_types = constraint_types;

        // Solve the puzzle
        let start = std::time::Instant::now();
        let result = self.solver.solve(puzzle)?;
        trajectory.latency_ms = start.elapsed().as_millis() as u64;

        // Record attempt
        let solution_str = result
            .solutions
            .first()
            .map(|d| d.to_string())
            .unwrap_or_else(|| "none".to_string());

        let confidence = self.calculate_confidence(&result, puzzle);

        trajectory.record_attempt(
            solution_str,
            confidence,
            result.steps,
            result.tool_calls,
            &self.current_strategy.name,
        );

        // Determine verdict
        let verdict = if result.correct {
            if confidence >= 0.9 {
                Verdict::Success
            } else {
                Verdict::Acceptable
            }
        } else if result.solved {
            Verdict::Suboptimal {
                reason: "Solution found but incorrect".to_string(),
                delta: 1.0 - confidence,
            }
        } else if confidence < self.current_strategy.confidence_threshold {
            Verdict::LowConfidence
        } else {
            Verdict::Failed
        };

        trajectory.set_verdict(verdict, puzzle.solutions.first().map(|d| d.to_string()));

        // Record trajectory for learning
        self.reasoning_bank.record_trajectory(trajectory);
        self.episodes += 1;

        Ok(result)
    }

    /// Calculate confidence in a result
    fn calculate_confidence(&self, result: &SolverResult, puzzle: &TemporalPuzzle) -> f64 {
        let mut confidence = 0.5;

        // Higher confidence if solved quickly
        if result.solved {
            confidence += 0.2;
            if result.steps < self.solver.max_steps / 2 {
                confidence += 0.1;
            }
        }

        // Higher confidence with tool use on complex puzzles
        if result.tool_calls > 0 && puzzle.difficulty > 5 {
            confidence += 0.1;
        }

        // Lower confidence if took many steps
        if result.steps > self.solver.max_steps * 3 / 4 {
            confidence -= 0.1;
        }

        // Adjust based on learned calibration
        let calibrated_threshold = self
            .reasoning_bank
            .calibration
            .get_threshold(puzzle.difficulty);
        if confidence >= calibrated_threshold {
            confidence += 0.05;
        }

        confidence.min(1.0).max(0.0)
    }

    /// Get learning progress
    pub fn learning_progress(&self) -> crate::reasoning_bank::LearningProgress {
        self.reasoning_bank.learning_progress()
    }

    /// Get hints for a puzzle
    pub fn get_hints(&self, constraint_types: &[String]) -> Vec<String> {
        self.reasoning_bank.get_hints(constraint_types)
    }
}

/// Get the type name of a constraint for pattern matching
fn constraint_type_name(constraint: &TemporalConstraint) -> String {
    match constraint {
        TemporalConstraint::Exact(_) => "Exact".to_string(),
        TemporalConstraint::After(_) => "After".to_string(),
        TemporalConstraint::Before(_) => "Before".to_string(),
        TemporalConstraint::Between(_, _) => "Between".to_string(),
        TemporalConstraint::DayOfWeek(_) => "DayOfWeek".to_string(),
        TemporalConstraint::DaysAfter(_, _) => "DaysAfter".to_string(),
        TemporalConstraint::DaysBefore(_, _) => "DaysBefore".to_string(),
        TemporalConstraint::InMonth(_) => "InMonth".to_string(),
        TemporalConstraint::InYear(_) => "InYear".to_string(),
        TemporalConstraint::DayOfMonth(_) => "DayOfMonth".to_string(),
        TemporalConstraint::RelativeToEvent(_, _) => "RelativeToEvent".to_string(),
    }
}
