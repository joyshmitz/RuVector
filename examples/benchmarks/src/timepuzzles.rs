//! TimePuzzles Generator
//!
//! Generates constraint-based temporal reasoning puzzles
//! based on the TimePuzzles benchmark methodology (arXiv:2601.07148)
//!
//! Key features:
//! - Factual temporal anchors with calendar relations
//! - Cross-cultural date systems
//! - Controlled difficulty levels
//! - Dynamic puzzle generation

use crate::temporal::{TemporalConstraint, TemporalPuzzle};
use anyhow::Result;
use chrono::{Datelike, NaiveDate, Weekday};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Puzzle generator configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PuzzleGeneratorConfig {
    /// Minimum difficulty (1-10)
    pub min_difficulty: u8,
    /// Maximum difficulty (1-10)
    pub max_difficulty: u8,
    /// Constraint density (1-5)
    pub constraint_density: u8,
    /// Include cross-cultural references
    pub cross_cultural: bool,
    /// Include relative constraints
    pub relative_constraints: bool,
    /// Year range for puzzles
    pub year_range: (i32, i32),
    /// Random seed (optional)
    pub seed: Option<u64>,
}

impl Default for PuzzleGeneratorConfig {
    fn default() -> Self {
        Self {
            min_difficulty: 1,
            max_difficulty: 10,
            constraint_density: 3,
            cross_cultural: true,
            relative_constraints: true,
            year_range: (2000, 2030),
            seed: None,
        }
    }
}

/// Known events for temporal anchoring
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalAnchor {
    pub name: String,
    pub date: NaiveDate,
    pub category: String,
    pub culture: String,
}

impl TemporalAnchor {
    pub fn new(
        name: impl Into<String>,
        year: i32,
        month: u32,
        day: u32,
        category: impl Into<String>,
        culture: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            date: NaiveDate::from_ymd_opt(year, month, day).unwrap(),
            category: category.into(),
            culture: culture.into(),
        }
    }
}

/// TimePuzzles generator
pub struct PuzzleGenerator {
    config: PuzzleGeneratorConfig,
    anchors: Vec<TemporalAnchor>,
    rng: StdRng,
}

impl PuzzleGenerator {
    /// Create a new generator with config
    pub fn new(config: PuzzleGeneratorConfig) -> Self {
        let rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut gen = Self {
            config,
            anchors: Vec::new(),
            rng,
        };
        gen.init_anchors();
        gen
    }

    /// Initialize standard temporal anchors
    fn init_anchors(&mut self) {
        // Western holidays
        self.anchors.push(TemporalAnchor::new(
            "Christmas",
            2024,
            12,
            25,
            "holiday",
            "western",
        ));
        self.anchors.push(TemporalAnchor::new(
            "New Year", 2024, 1, 1, "holiday", "western",
        ));
        self.anchors.push(TemporalAnchor::new(
            "Independence Day",
            2024,
            7,
            4,
            "holiday",
            "american",
        ));
        self.anchors.push(TemporalAnchor::new(
            "Halloween",
            2024,
            10,
            31,
            "holiday",
            "western",
        ));
        self.anchors.push(TemporalAnchor::new(
            "Valentine's Day",
            2024,
            2,
            14,
            "holiday",
            "western",
        ));

        // Cross-cultural events
        if self.config.cross_cultural {
            // Chinese New Year 2024 (Year of the Dragon)
            self.anchors.push(TemporalAnchor::new(
                "Chinese New Year 2024",
                2024,
                2,
                10,
                "holiday",
                "chinese",
            ));
            // Diwali 2024
            self.anchors.push(TemporalAnchor::new(
                "Diwali 2024",
                2024,
                11,
                1,
                "holiday",
                "indian",
            ));
            // Eid al-Fitr 2024
            self.anchors.push(TemporalAnchor::new(
                "Eid al-Fitr 2024",
                2024,
                4,
                10,
                "holiday",
                "islamic",
            ));
            // Hanukkah 2024 (starts)
            self.anchors.push(TemporalAnchor::new(
                "Hanukkah 2024",
                2024,
                12,
                25,
                "holiday",
                "jewish",
            ));
        }

        // Historical events
        self.anchors.push(TemporalAnchor::new(
            "Moon Landing",
            1969,
            7,
            20,
            "historical",
            "global",
        ));
        self.anchors.push(TemporalAnchor::new(
            "Fall of Berlin Wall",
            1989,
            11,
            9,
            "historical",
            "global",
        ));
        self.anchors.push(TemporalAnchor::new(
            "Y2K",
            2000,
            1,
            1,
            "historical",
            "global",
        ));
    }

    /// Generate a single puzzle
    pub fn generate_puzzle(&mut self, id: impl Into<String>) -> Result<TemporalPuzzle> {
        let id = id.into();
        let difficulty = self
            .rng
            .gen_range(self.config.min_difficulty..=self.config.max_difficulty);
        let num_constraints = self.config.constraint_density as usize + difficulty as usize / 2;

        // Select a target date
        let year = self
            .rng
            .gen_range(self.config.year_range.0..=self.config.year_range.1);
        let month = self.rng.gen_range(1..=12);
        let max_day = match month {
            4 | 6 | 9 | 11 => 30,
            2 => {
                if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                    29
                } else {
                    28
                }
            }
            _ => 31,
        };
        let day = self.rng.gen_range(1..=max_day);
        let target = NaiveDate::from_ymd_opt(year, month, day).unwrap();

        let mut puzzle = TemporalPuzzle::new(id.clone(), format!("Find the date (puzzle {})", id))
            .with_difficulty(difficulty)
            .with_solutions(vec![target]);

        // Add constraints and collect used anchors
        let constraint_types = self.select_constraint_types(num_constraints, difficulty);
        let mut used_anchors: Vec<TemporalAnchor> = Vec::new();

        for ct in constraint_types {
            let (constraint, anchor_opt) = self.generate_constraint_with_anchor(&ct, target)?;
            puzzle.constraints.push(constraint);
            if let Some(anchor) = anchor_opt {
                used_anchors.push(anchor);
            }
        }

        // Add all used anchors to references
        for anchor in used_anchors {
            puzzle.references.insert(anchor.name.clone(), anchor.date);
        }

        // Add tags
        puzzle.tags = vec![
            format!("difficulty:{}", difficulty),
            format!("year:{}", year),
        ];

        Ok(puzzle)
    }

    /// Select constraint types based on difficulty
    fn select_constraint_types(&mut self, count: usize, difficulty: u8) -> Vec<ConstraintType> {
        let mut types = Vec::new();

        // Base constraints (always include)
        types.push(ConstraintType::Year);

        // Add more constraints based on difficulty
        if difficulty >= 2 {
            types.push(ConstraintType::Month);
        }
        if difficulty >= 3 {
            types.push(ConstraintType::DayOfWeek);
        }
        if difficulty >= 4 && self.config.relative_constraints {
            types.push(ConstraintType::RelativeToAnchor);
        }
        if difficulty >= 5 {
            types.push(ConstraintType::DayRange);
        }
        if difficulty >= 6 {
            types.push(ConstraintType::DateRange);
        }
        if difficulty >= 7 {
            types.push(ConstraintType::MultipleConditions);
        }

        // Trim to count
        while types.len() < count {
            let base = vec![
                ConstraintType::Year,
                ConstraintType::Month,
                ConstraintType::DayOfWeek,
                ConstraintType::DayRange,
            ];
            types.push(base.choose(&mut self.rng).unwrap().clone());
        }
        types.truncate(count);
        types
    }

    /// Generate a specific constraint, returning the constraint and any anchor used
    fn generate_constraint_with_anchor(
        &mut self,
        ct: &ConstraintType,
        target: NaiveDate,
    ) -> Result<(TemporalConstraint, Option<TemporalAnchor>)> {
        match ct {
            ConstraintType::Year => Ok((TemporalConstraint::InYear(target.year()), None)),
            ConstraintType::Month => Ok((TemporalConstraint::InMonth(target.month()), None)),
            ConstraintType::DayOfMonth => Ok((TemporalConstraint::DayOfMonth(target.day()), None)),
            ConstraintType::DayOfWeek => {
                Ok((TemporalConstraint::DayOfWeek(target.weekday()), None))
            }
            ConstraintType::DayRange => {
                let start = target.day().saturating_sub(self.rng.gen_range(0..5));
                let end = (target.day() + self.rng.gen_range(0..5)).min(28);
                let start_date =
                    NaiveDate::from_ymd_opt(target.year(), target.month(), start.max(1))
                        .unwrap_or(target);
                let end_date =
                    NaiveDate::from_ymd_opt(target.year(), target.month(), end).unwrap_or(target);
                Ok((TemporalConstraint::Between(start_date, end_date), None))
            }
            ConstraintType::DateRange => {
                let days_before = self.rng.gen_range(0..10);
                let days_after = self.rng.gen_range(0..10);
                let start = target - chrono::Duration::days(days_before);
                let end = target + chrono::Duration::days(days_after);
                Ok((TemporalConstraint::Between(start, end), None))
            }
            ConstraintType::RelativeToAnchor => {
                if let Some(anchor) = self.anchors.choose(&mut self.rng).cloned() {
                    let diff = (target - anchor.date).num_days();
                    let constraint = if diff >= 0 {
                        TemporalConstraint::DaysAfter(anchor.name.clone(), diff)
                    } else {
                        TemporalConstraint::DaysBefore(anchor.name.clone(), diff.abs())
                    };
                    Ok((constraint, Some(anchor)))
                } else {
                    Ok((TemporalConstraint::InYear(target.year()), None))
                }
            }
            ConstraintType::MultipleConditions => {
                // This is a meta-type, just return year constraint
                Ok((TemporalConstraint::InYear(target.year()), None))
            }
        }
    }

    /// Generate a batch of puzzles
    pub fn generate_batch(&mut self, count: usize) -> Result<Vec<TemporalPuzzle>> {
        let mut puzzles = Vec::with_capacity(count);
        for i in 0..count {
            let puzzle = self.generate_puzzle(format!("puzzle-{:04}", i + 1))?;
            puzzles.push(puzzle);
        }
        Ok(puzzles)
    }

    /// Generate puzzles at specific difficulty
    pub fn generate_at_difficulty(
        &mut self,
        count: usize,
        difficulty: u8,
    ) -> Result<Vec<TemporalPuzzle>> {
        let orig_min = self.config.min_difficulty;
        let orig_max = self.config.max_difficulty;

        self.config.min_difficulty = difficulty;
        self.config.max_difficulty = difficulty;

        let puzzles = self.generate_batch(count);

        self.config.min_difficulty = orig_min;
        self.config.max_difficulty = orig_max;

        puzzles
    }
}

/// Constraint type enumeration
#[derive(Clone, Debug)]
enum ConstraintType {
    Year,
    Month,
    DayOfMonth,
    DayOfWeek,
    DayRange,
    DateRange,
    RelativeToAnchor,
    MultipleConditions,
}

/// Sample puzzle sets
pub struct SamplePuzzles;

impl SamplePuzzles {
    /// Get easy puzzles (difficulty 1-3)
    pub fn easy() -> Vec<TemporalPuzzle> {
        let mut gen = PuzzleGenerator::new(PuzzleGeneratorConfig {
            min_difficulty: 1,
            max_difficulty: 3,
            seed: Some(42),
            ..Default::default()
        });
        gen.generate_batch(10).unwrap()
    }

    /// Get medium puzzles (difficulty 4-6)
    pub fn medium() -> Vec<TemporalPuzzle> {
        let mut gen = PuzzleGenerator::new(PuzzleGeneratorConfig {
            min_difficulty: 4,
            max_difficulty: 6,
            seed: Some(42),
            ..Default::default()
        });
        gen.generate_batch(10).unwrap()
    }

    /// Get hard puzzles (difficulty 7-10)
    pub fn hard() -> Vec<TemporalPuzzle> {
        let mut gen = PuzzleGenerator::new(PuzzleGeneratorConfig {
            min_difficulty: 7,
            max_difficulty: 10,
            seed: Some(42),
            ..Default::default()
        });
        gen.generate_batch(10).unwrap()
    }

    /// Get cross-cultural puzzles
    pub fn cross_cultural() -> Vec<TemporalPuzzle> {
        let mut gen = PuzzleGenerator::new(PuzzleGeneratorConfig {
            cross_cultural: true,
            relative_constraints: true,
            min_difficulty: 5,
            max_difficulty: 8,
            seed: Some(42),
            ..Default::default()
        });
        gen.generate_batch(10).unwrap()
    }

    /// Get a mixed sample set (50 puzzles across all difficulties)
    pub fn mixed_sample() -> Vec<TemporalPuzzle> {
        let mut all = Vec::new();
        all.extend(Self::easy());
        all.extend(Self::medium());
        all.extend(Self::hard());
        all.extend(Self::cross_cultural());

        // Add more easy/medium to match TimePuzzles distribution
        let mut gen = PuzzleGenerator::new(PuzzleGeneratorConfig {
            min_difficulty: 2,
            max_difficulty: 5,
            seed: Some(123),
            ..Default::default()
        });
        all.extend(gen.generate_batch(10).unwrap());

        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_puzzle_generation() {
        let mut gen = PuzzleGenerator::new(PuzzleGeneratorConfig {
            seed: Some(42),
            ..Default::default()
        });

        let puzzle = gen.generate_puzzle("test-1").unwrap();
        assert!(!puzzle.constraints.is_empty());
        assert!(!puzzle.solutions.is_empty());
    }

    #[test]
    fn test_batch_generation() {
        let mut gen = PuzzleGenerator::new(PuzzleGeneratorConfig {
            seed: Some(42),
            ..Default::default()
        });

        let puzzles = gen.generate_batch(20).unwrap();
        assert_eq!(puzzles.len(), 20);
    }

    #[test]
    fn test_sample_puzzles() {
        let easy = SamplePuzzles::easy();
        assert_eq!(easy.len(), 10);
        assert!(easy.iter().all(|p| p.difficulty <= 3));

        let hard = SamplePuzzles::hard();
        assert!(hard.iter().all(|p| p.difficulty >= 7));
    }
}
