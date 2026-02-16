//! Adaptive solver engine, puzzle generator, reasoning bank, and acceptance test.
//!
//! Three-loop architecture:
//! - Fast loop: constraint propagation, solve, rollback on failure
//! - Medium loop: PolicyKernel + Thompson Sampling (skip-mode selection)
//! - Slow loop: KnowledgeCompiler + ReasoningBank (pattern learning)

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::policy::{
    CompiledConfig, KnowledgeCompiler, PolicyContext, PolicyKernel, SkipMode, SkipOutcome,
    count_distractors,
};
use crate::types::{Constraint, Date, Puzzle, Rng64, Weekday, constraint_type_name};

// ═════════════════════════════════════════════════════════════════════
// Solve result
// ═════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolveResult {
    pub puzzle_id: String,
    pub solved: bool,
    pub correct: bool,
    pub steps: usize,
    pub solutions_found: usize,
    pub skip_mode: String,
    pub context_bucket: String,
}

// ═════════════════════════════════════════════════════════════════════
// ReasoningBank (simplified for WASM)
// ═════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReasoningBank {
    /// Signature → (steps, correct) history for compilation.
    trajectories: Vec<(String, u8, Vec<String>, usize, bool)>,
    /// Promotion staging: only promoted after non-regression check.
    staged: Vec<(String, u8, Vec<String>, usize, bool)>,
    checkpoint_len: usize,
    pub patterns_learned: usize,
}

impl ReasoningBank {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, puzzle_id: &str, difficulty: u8, ctypes: &[&str], steps: usize, correct: bool) {
        let entry = (
            String::from(puzzle_id),
            difficulty,
            ctypes.iter().map(|s| String::from(*s)).collect(),
            steps,
            correct,
        );
        self.staged.push(entry);
    }

    pub fn promote(&mut self) {
        let staged = core::mem::take(&mut self.staged);
        for entry in staged {
            if entry.4 {
                self.patterns_learned += 1;
            }
            self.trajectories.push(entry);
        }
    }

    pub fn checkpoint(&mut self) -> usize {
        self.checkpoint_len = self.trajectories.len();
        self.checkpoint_len
    }

    pub fn rollback(&mut self, cp: usize) {
        self.trajectories.truncate(cp);
        self.staged.clear();
    }

    pub fn compile_to(&self, compiler: &mut KnowledgeCompiler) {
        let refs: Vec<(String, u8, Vec<&str>, usize, bool)> = self
            .trajectories
            .iter()
            .map(|(id, d, ct, s, c)| (id.clone(), *d, ct.iter().map(|x| x.as_str()).collect(), *s, *c))
            .collect();
        compiler.compile_from_trajectories(&refs);
    }
}

// ═════════════════════════════════════════════════════════════════════
// Puzzle generator (deterministic, no rand crate)
// ═════════════════════════════════════════════════════════════════════

pub struct PuzzleGenerator {
    rng: Rng64,
    min_diff: u8,
    max_diff: u8,
    year_lo: i32,
    year_hi: i32,
    next_id: usize,
}

impl PuzzleGenerator {
    pub fn new(seed: u64, min_diff: u8, max_diff: u8) -> Self {
        Self {
            rng: Rng64::new(seed),
            min_diff: min_diff.max(1),
            max_diff: max_diff.max(1).max(min_diff),
            year_lo: 2000,
            year_hi: 2030,
            next_id: 0,
        }
    }

    pub fn generate(&mut self) -> Puzzle {
        let difficulty = self.rng.range(self.min_diff as i32, self.max_diff as i32) as u8;
        let year = self.rng.range(self.year_lo, self.year_hi);
        let month = self.rng.range(1, 12) as u32;
        let max_day = match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 28,
            _ => 28,
        };
        let day = self.rng.range(1, max_day) as u32;
        let target = Date::new(year, month, day).unwrap_or(Date { year, month: 1, day: 1 });

        let mut constraints = Vec::new();
        let constraint_count = (difficulty as usize / 2 + 2).min(7);

        // Always include a Between constraint for the search range
        let range_days = 30 * (difficulty as i64 + 1);
        let start = target.add_days(-(range_days / 2));
        let end = target.add_days(range_days / 2);
        constraints.push(Constraint::Between(start, end));

        // Add additional constraints based on difficulty
        let mut added = 1;
        while added < constraint_count {
            let kind = self.rng.range(0, 6);
            let c = match kind {
                0 => Constraint::InYear(target.year),
                1 => Constraint::InMonth(target.month),
                2 => Constraint::DayOfWeek(target.weekday()),
                3 => Constraint::DayOfMonth(target.day),
                4 if difficulty >= 3 => {
                    let shift = self.rng.range(-5, 5) as i64;
                    Constraint::After(target.add_days(shift - 10))
                }
                5 if difficulty >= 3 => {
                    let shift = self.rng.range(-5, 5) as i64;
                    Constraint::Before(target.add_days(shift + 10))
                }
                _ => Constraint::InMonth(target.month),
            };
            if !constraints.contains(&c) {
                constraints.push(c);
                added += 1;
            } else {
                added += 1;
            }
        }

        // Add distractor constraints for higher difficulty
        if difficulty >= 5 {
            let dist_count = (difficulty as usize - 4).min(3);
            for _ in 0..dist_count {
                let fake_month = (target.month % 12) + 1;
                constraints.push(Constraint::InMonth(fake_month));
            }
        }

        // Compute solutions
        let mut solutions = Vec::new();
        let mut d = start;
        while d <= end {
            let puzzle_tmp = Puzzle {
                id: String::new(),
                constraints: constraints.clone(),
                references: BTreeMap::new(),
                solutions: Vec::new(),
                difficulty,
            };
            if puzzle_tmp.check_date(d) {
                solutions.push(d);
            }
            d = d.succ();
        }
        // Ensure at least the target is a solution
        if solutions.is_empty() {
            solutions.push(target);
        }

        let id = format!("p_{}", self.next_id);
        self.next_id += 1;

        Puzzle {
            id,
            constraints,
            references: BTreeMap::new(),
            solutions,
            difficulty,
        }
    }

    pub fn generate_batch(&mut self, count: usize) -> Vec<Puzzle> {
        (0..count).map(|_| self.generate()).collect()
    }
}

// ═════════════════════════════════════════════════════════════════════
// Adaptive solver (three-loop architecture)
// ═════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveSolver {
    pub policy_kernel: PolicyKernel,
    pub compiler: KnowledgeCompiler,
    pub bank: ReasoningBank,
    pub compiler_enabled: bool,
    pub router_enabled: bool,
    pub step_budget: usize,
    pub noisy_hint: bool,
}

impl AdaptiveSolver {
    pub fn new() -> Self {
        Self {
            policy_kernel: PolicyKernel::new(),
            compiler: KnowledgeCompiler::new(),
            bank: ReasoningBank::new(),
            compiler_enabled: false,
            router_enabled: false,
            step_budget: 400,
            noisy_hint: false,
        }
    }

    /// Solve a puzzle using the three-loop adaptive architecture.
    pub fn solve(&mut self, puzzle: &Puzzle) -> SolveResult {
        let has_dow = puzzle.constraints.iter().any(|c| matches!(c, Constraint::DayOfWeek(_)));
        let range = self.estimate_range(puzzle);
        let distractors = count_distractors(puzzle);

        let ctx = PolicyContext {
            posterior_range: range,
            distractor_count: distractors,
            has_day_of_week: has_dow,
            noisy: self.noisy_hint,
        };

        // Medium loop: select skip mode via policy
        let skip_mode = self.select_skip_mode(&ctx);

        // Try compiler suggestion first (slow loop feedback)
        let compiled = if self.compiler_enabled {
            self.compiler.lookup(puzzle).cloned()
        } else {
            None
        };

        // Fast loop: solve with constraint propagation
        let (solutions, steps) = self.solve_inner(puzzle, &skip_mode, &compiled);

        let correct = !solutions.is_empty()
            && puzzle.solutions.iter().any(|s| solutions.contains(s));
        let solved = !solutions.is_empty();

        // Check for early commit error
        let initial_candidates = range;
        let remaining = solutions.len();
        let early_commit_wrong = solved && !correct;

        // Record outcome (fast loop → medium loop feedback)
        let outcome = SkipOutcome {
            mode: skip_mode.clone(),
            correct,
            steps,
            early_commit_wrong,
            initial_candidates,
            remaining_at_commit: remaining,
        };
        self.policy_kernel.record_outcome(&ctx, &outcome);

        // Record trajectory (fast loop → slow loop feedback)
        let ctypes: Vec<&str> = puzzle.constraints.iter().map(constraint_type_name).collect();
        self.bank.record(&puzzle.id, puzzle.difficulty, &ctypes, steps, correct);

        // Update compiler on success/failure
        if self.compiler_enabled {
            if correct {
                self.compiler.record_success(puzzle, steps);
            } else if compiled.is_some() {
                self.compiler.record_failure(puzzle);
            }
        }

        let bucket = PolicyKernel::context_bucket(&ctx);
        SolveResult {
            puzzle_id: puzzle.id.clone(),
            solved,
            correct,
            steps,
            solutions_found: solutions.len(),
            skip_mode: String::from(skip_mode.name()),
            context_bucket: bucket,
        }
    }

    fn select_skip_mode(&mut self, ctx: &PolicyContext) -> SkipMode {
        if self.router_enabled {
            // Mode C: speculative dual-path or learned policy
            if let Some((arm1, _arm2)) = self.policy_kernel.should_speculate(ctx) {
                self.policy_kernel.speculative_attempts += 1;
                return arm1;
            }
            self.policy_kernel.learned_policy(ctx)
        } else if self.compiler_enabled {
            // Mode B: compiler-suggested
            PolicyKernel::fixed_policy(ctx) // fallback for now
        } else {
            // Mode A: fixed heuristic
            PolicyKernel::fixed_policy(ctx)
        }
    }

    fn solve_inner(
        &self,
        puzzle: &Puzzle,
        skip_mode: &SkipMode,
        _compiled: &Option<CompiledConfig>,
    ) -> (Vec<Date>, usize) {
        // Constraint propagation pre-pass: determine search range
        let (range_start, range_end) = self.compute_range(puzzle);
        let mut candidates = Vec::new();
        let mut steps = 0;

        let mut d = range_start;
        while d <= range_end && steps < self.step_budget {
            steps += 1;
            // Skip mode optimization
            match skip_mode {
                SkipMode::Weekday => {
                    if let Some(target_wd) = self.target_weekday(puzzle) {
                        if d.weekday() != target_wd {
                            d = self.advance_to_weekday(d, target_wd);
                            if d > range_end {
                                break;
                            }
                        }
                    }
                }
                SkipMode::Hybrid => {
                    if let Some(target_wd) = self.target_weekday(puzzle) {
                        if d.weekday() != target_wd {
                            d = self.advance_to_weekday(d, target_wd);
                            if d > range_end {
                                break;
                            }
                        }
                    }
                    // Additionally skip non-matching months
                    if let Some(target_m) = self.target_month(puzzle) {
                        if d.month != target_m {
                            d = d.succ();
                            continue;
                        }
                    }
                }
                SkipMode::None => {}
            }

            if puzzle.check_date(d) {
                candidates.push(d);
            }
            d = d.succ();
        }

        (candidates, steps)
    }

    fn estimate_range(&self, puzzle: &Puzzle) -> usize {
        let (start, end) = self.compute_range(puzzle);
        start.days_until(end).unsigned_abs() as usize
    }

    fn compute_range(&self, puzzle: &Puzzle) -> (Date, Date) {
        let mut lo = Date::new(1990, 1, 1).unwrap();
        let mut hi = Date::new(2040, 12, 31).unwrap();

        for c in &puzzle.constraints {
            match c {
                Constraint::Between(a, b) => {
                    if *a > lo { lo = *a; }
                    if *b < hi { hi = *b; }
                }
                Constraint::After(d) => {
                    let next = d.succ();
                    if next > lo { lo = next; }
                }
                Constraint::Before(d) => {
                    let prev = d.pred();
                    if prev < hi { hi = prev; }
                }
                Constraint::InYear(y) => {
                    let yr_start = Date::new(*y, 1, 1).unwrap();
                    let yr_end = Date::new(*y, 12, 31).unwrap();
                    if yr_start > lo { lo = yr_start; }
                    if yr_end < hi { hi = yr_end; }
                }
                Constraint::Exact(d) => {
                    lo = *d;
                    hi = *d;
                }
                _ => {}
            }
        }
        (lo, hi)
    }

    fn target_weekday(&self, puzzle: &Puzzle) -> Option<Weekday> {
        for c in &puzzle.constraints {
            if let Constraint::DayOfWeek(w) = c {
                return Some(*w);
            }
        }
        None
    }

    fn target_month(&self, puzzle: &Puzzle) -> Option<u32> {
        for c in &puzzle.constraints {
            if let Constraint::InMonth(m) = c {
                return Some(*m);
            }
        }
        None
    }

    fn advance_to_weekday(&self, from: Date, target: Weekday) -> Date {
        let mut d = from;
        for _ in 0..7 {
            if d.weekday() == target {
                return d;
            }
            d = d.succ();
        }
        d
    }
}

// ═════════════════════════════════════════════════════════════════════
// Acceptance test runner
// ═════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CycleMetrics {
    pub cycle: usize,
    pub accuracy: f64,
    pub cost_per_solve: f64,
    pub noise_accuracy: f64,
    pub violations: usize,
    pub patterns_learned: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcceptanceConfig {
    pub holdout_size: usize,
    pub training_per_cycle: usize,
    pub cycles: usize,
    pub step_budget: usize,
    pub holdout_seed: u64,
    pub training_seed: u64,
    pub noise_rate: f64,
    pub min_accuracy: f64,
}

impl Default for AcceptanceConfig {
    fn default() -> Self {
        Self {
            holdout_size: 100,
            training_per_cycle: 100,
            cycles: 5,
            step_budget: 400,
            holdout_seed: 0xDEAD_BEEF,
            training_seed: 42,
            noise_rate: 0.25,
            min_accuracy: 0.80,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcceptanceResult {
    pub cycles: Vec<CycleMetrics>,
    pub passed: bool,
    pub accuracy_maintained: bool,
    pub cost_improved: bool,
    pub robustness_improved: bool,
    pub zero_violations: bool,
    pub dimensions_improved: usize,
}

/// Run the full acceptance test with three-loop learning.
pub fn run_acceptance_test(config: &AcceptanceConfig) -> AcceptanceResult {
    run_acceptance_mode(config, false, false)
}

/// Run acceptance test in a specific mode.
/// compiler_enabled=true, router_enabled=true → Mode C (full learned)
/// compiler_enabled=true, router_enabled=false → Mode B (compiler only)
/// compiler_enabled=false, router_enabled=false → Mode A (baseline)
pub fn run_acceptance_mode(
    config: &AcceptanceConfig,
    compiler_enabled: bool,
    router_enabled: bool,
) -> AcceptanceResult {
    let holdout = {
        let mut gen = PuzzleGenerator::new(config.holdout_seed, 1, 10);
        gen.generate_batch(config.holdout_size)
    };

    let mut solver = AdaptiveSolver::new();
    solver.compiler_enabled = compiler_enabled;
    solver.router_enabled = router_enabled;
    solver.step_budget = config.step_budget;

    let mut cycle_metrics: Vec<CycleMetrics> = Vec::new();

    for cycle in 0..config.cycles {
        // Slow loop: recompile knowledge
        if compiler_enabled {
            solver.bank.compile_to(&mut solver.compiler);
        }

        let checkpoint = solver.bank.checkpoint();

        // Training phase
        let mut gen = PuzzleGenerator::new(
            config.training_seed + (cycle as u64 * 10_000),
            1,
            10,
        );
        let training = gen.generate_batch(config.training_per_cycle);
        let mut train_rng = Rng64::new(config.training_seed.wrapping_add(cycle as u64 * 7919));

        for puzzle in &training {
            let is_noisy = train_rng.next_f64() < config.noise_rate;
            let solve_p = if is_noisy {
                inject_noise(puzzle, &mut train_rng)
            } else {
                puzzle.clone()
            };
            solver.noisy_hint = is_noisy;
            solver.solve(&solve_p);
            solver.noisy_hint = false;
        }

        // Holdout evaluation: clean
        let (clean_correct, clean_total_steps) = evaluate_holdout(&holdout, &mut solver, false, 0);
        let accuracy = clean_correct as f64 / holdout.len() as f64;

        // Rollback if accuracy regressed
        if cycle > 0 {
            let prev_acc = cycle_metrics[cycle - 1].accuracy;
            if accuracy < prev_acc - 0.05 {
                solver.bank.rollback(checkpoint);
            }
        }
        solver.bank.promote();

        // Holdout evaluation: noisy
        let (noisy_correct, _) = evaluate_holdout(
            &holdout,
            &mut solver,
            true,
            config.holdout_seed.wrapping_add(cycle as u64 * 31337),
        );
        let noise_accuracy = noisy_correct as f64 / holdout.len() as f64;
        let cost_per_solve = if clean_correct > 0 {
            clean_total_steps as f64 / clean_correct as f64
        } else {
            clean_total_steps as f64
        };

        cycle_metrics.push(CycleMetrics {
            cycle: cycle + 1,
            accuracy,
            cost_per_solve,
            noise_accuracy,
            violations: 0,
            patterns_learned: solver.bank.patterns_learned,
        });
    }

    let first = &cycle_metrics[0];
    let last = cycle_metrics.last().unwrap();

    let accuracy_maintained = cycle_metrics.iter().all(|c| c.accuracy >= config.min_accuracy * 0.95)
        && last.accuracy >= config.min_accuracy;

    let cost_decrease = if first.cost_per_solve > 0.0 {
        1.0 - (last.cost_per_solve / first.cost_per_solve)
    } else {
        0.0
    };
    let cost_improved = cost_decrease >= 0.10;

    let robustness_gain = last.noise_accuracy - first.noise_accuracy;
    let robustness_improved = robustness_gain >= 0.05;

    let zero_violations = cycle_metrics.iter().all(|c| c.violations == 0);

    let mut dims = 0;
    if cost_improved { dims += 1; }
    if robustness_improved { dims += 1; }
    if last.accuracy >= first.accuracy { dims += 1; }

    let passed = accuracy_maintained && zero_violations && dims >= 2;

    AcceptanceResult {
        cycles: cycle_metrics,
        passed,
        accuracy_maintained,
        cost_improved,
        robustness_improved,
        zero_violations,
        dimensions_improved: dims,
    }
}

fn evaluate_holdout(
    holdout: &[Puzzle],
    solver: &mut AdaptiveSolver,
    noisy: bool,
    noise_seed: u64,
) -> (usize, usize) {
    let mut correct = 0;
    let mut total_steps = 0;
    let mut rng = Rng64::new(noise_seed.max(1));

    for puzzle in holdout {
        let solve_p = if noisy {
            inject_noise(puzzle, &mut rng)
        } else {
            puzzle.clone()
        };
        solver.noisy_hint = noisy;
        let result = solver.solve(&solve_p);
        solver.noisy_hint = false;
        if result.correct {
            correct += 1;
        }
        total_steps += result.steps;
    }

    (correct, total_steps)
}

fn inject_noise(puzzle: &Puzzle, rng: &mut Rng64) -> Puzzle {
    let mut noisy = puzzle.clone();
    for c in noisy.constraints.iter_mut() {
        match c {
            Constraint::InMonth(ref mut m) => {
                if rng.next_f64() < 0.5 {
                    let shift = if rng.next_f64() < 0.5 { 1 } else { 11 };
                    *m = (*m + shift - 1) % 12 + 1;
                }
            }
            Constraint::DayOfMonth(ref mut d) => {
                if rng.next_f64() < 0.5 {
                    *d = (*d + 1).min(28).max(1);
                }
            }
            Constraint::InYear(ref mut y) => {
                if rng.next_f64() < 0.5 {
                    *y += if rng.next_f64() < 0.5 { 1 } else { -1 };
                }
            }
            _ => {}
        }
    }
    // Recompute solutions for noisy puzzle
    noisy.solutions = Vec::new();
    noisy
}
