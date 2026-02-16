//! WASM bindings for the Domain Expansion Engine.
//!
//! Provides JavaScript-accessible interfaces for cross-domain transfer learning,
//! Meta Thompson Sampling, PolicyKernel population search, and the acceleration
//! scoreboard. All domain operations run at native speed in the browser/edge.

use ruvector_domain_expansion::{
    AccelerationScoreboard, ArmId, ContextBucket, CostCurve,
    DomainExpansionEngine, DomainId, Evaluation, MetaThompsonEngine,
    PopulationSearch, Solution, Task,
};
use wasm_bindgen::prelude::*;

// ─── Engine ──────────────────────────────────────────────────────────────────

/// WASM-exported domain expansion engine.
#[wasm_bindgen]
pub struct WasmDomainExpansionEngine {
    inner: DomainExpansionEngine,
}

#[wasm_bindgen]
impl WasmDomainExpansionEngine {
    /// Create a new domain expansion engine with 3 core domains.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: DomainExpansionEngine::new(),
        }
    }

    /// Get registered domain IDs as JSON array.
    #[wasm_bindgen(js_name = domainIds)]
    pub fn domain_ids(&self) -> JsValue {
        let ids: Vec<String> = self.inner.domain_ids().into_iter().map(|d| d.0).collect();
        serde_wasm_bindgen::to_value(&ids).unwrap_or(JsValue::NULL)
    }

    /// Generate tasks for a domain. Returns JSON array of tasks.
    #[wasm_bindgen(js_name = generateTasks)]
    pub fn generate_tasks(&self, domain_id: &str, count: usize, difficulty: f32) -> JsValue {
        let id = DomainId(domain_id.to_string());
        let tasks = self.inner.generate_tasks(&id, count, difficulty);
        serde_wasm_bindgen::to_value(&tasks).unwrap_or(JsValue::NULL)
    }

    /// Generate holdout tasks for all domains.
    #[wasm_bindgen(js_name = generateHoldouts)]
    pub fn generate_holdouts(&mut self, tasks_per_domain: usize, difficulty: f32) {
        self.inner.generate_holdouts(tasks_per_domain, difficulty);
    }

    /// Evaluate a solution (JSON). Returns evaluation JSON.
    #[wasm_bindgen(js_name = evaluateAndRecord)]
    pub fn evaluate_and_record(
        &mut self,
        domain_id: &str,
        task_json: &str,
        solution_json: &str,
        difficulty_tier: &str,
        category: &str,
        arm: &str,
    ) -> JsValue {
        let task: Task = match serde_json::from_str(task_json) {
            Ok(t) => t,
            Err(_) => return JsValue::NULL,
        };
        let solution: Solution = match serde_json::from_str(solution_json) {
            Ok(s) => s,
            Err(_) => return JsValue::NULL,
        };

        let bucket = ContextBucket {
            difficulty_tier: difficulty_tier.to_string(),
            category: category.to_string(),
        };

        let eval = self.inner.evaluate_and_record(
            &DomainId(domain_id.to_string()),
            &task,
            &solution,
            bucket,
            ArmId(arm.to_string()),
        );

        serde_wasm_bindgen::to_value(&eval).unwrap_or(JsValue::NULL)
    }

    /// Select the best arm for a context using Thompson Sampling.
    #[wasm_bindgen(js_name = selectArm)]
    pub fn select_arm(
        &self,
        domain_id: &str,
        difficulty_tier: &str,
        category: &str,
    ) -> Option<String> {
        let bucket = ContextBucket {
            difficulty_tier: difficulty_tier.to_string(),
            category: category.to_string(),
        };
        self.inner
            .select_arm(&DomainId(domain_id.to_string()), &bucket)
            .map(|a| a.0)
    }

    /// Check if speculation should be triggered.
    #[wasm_bindgen(js_name = shouldSpeculate)]
    pub fn should_speculate(
        &self,
        domain_id: &str,
        difficulty_tier: &str,
        category: &str,
    ) -> bool {
        let bucket = ContextBucket {
            difficulty_tier: difficulty_tier.to_string(),
            category: category.to_string(),
        };
        self.inner
            .should_speculate(&DomainId(domain_id.to_string()), &bucket)
    }

    /// Initiate transfer from source to target domain.
    #[wasm_bindgen(js_name = initiateTransfer)]
    pub fn initiate_transfer(&mut self, source: &str, target: &str) {
        self.inner.initiate_transfer(
            &DomainId(source.to_string()),
            &DomainId(target.to_string()),
        );
    }

    /// Verify a transfer delta. Returns verification JSON.
    #[wasm_bindgen(js_name = verifyTransfer)]
    pub fn verify_transfer(
        &self,
        source: &str,
        target: &str,
        source_before: f32,
        source_after: f32,
        target_before: f32,
        target_after: f32,
        baseline_cycles: u64,
        transfer_cycles: u64,
    ) -> JsValue {
        let v = self.inner.verify_transfer(
            &DomainId(source.to_string()),
            &DomainId(target.to_string()),
            source_before,
            source_after,
            target_before,
            target_after,
            baseline_cycles,
            transfer_cycles,
        );
        serde_wasm_bindgen::to_value(&v).unwrap_or(JsValue::NULL)
    }

    /// Evaluate all policy kernels on holdout tasks.
    #[wasm_bindgen(js_name = evaluatePopulation)]
    pub fn evaluate_population(&mut self) {
        self.inner.evaluate_population();
    }

    /// Evolve the policy kernel population.
    #[wasm_bindgen(js_name = evolvePopulation)]
    pub fn evolve_population(&mut self) {
        self.inner.evolve_population();
    }

    /// Get population statistics as JSON.
    #[wasm_bindgen(js_name = populationStats)]
    pub fn population_stats(&self) -> JsValue {
        let stats = self.inner.population_stats();
        serde_wasm_bindgen::to_value(&stats).unwrap_or(JsValue::NULL)
    }

    /// Get the scoreboard summary as JSON.
    #[wasm_bindgen(js_name = scoreboardSummary)]
    pub fn scoreboard_summary(&self) -> JsValue {
        let summary = self.inner.scoreboard_summary();
        serde_wasm_bindgen::to_value(&summary).unwrap_or(JsValue::NULL)
    }

    /// Get the best policy kernel as JSON.
    #[wasm_bindgen(js_name = bestKernel)]
    pub fn best_kernel(&self) -> JsValue {
        match self.inner.best_kernel() {
            Some(k) => serde_wasm_bindgen::to_value(k).unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }

    /// Get counterexamples for a domain as JSON.
    #[wasm_bindgen(js_name = counterexamples)]
    pub fn counterexamples(&self, domain_id: &str) -> JsValue {
        let examples = self
            .inner
            .counterexamples(&DomainId(domain_id.to_string()));
        let serializable: Vec<(&Task, &Solution, &Evaluation)> = examples
            .iter()
            .map(|(t, s, e)| (t, s, e))
            .collect();
        serde_wasm_bindgen::to_value(&serializable).unwrap_or(JsValue::NULL)
    }
}

// ─── Standalone Thompson Sampling ────────────────────────────────────────────

/// WASM-exported standalone Thompson Sampling engine.
#[wasm_bindgen]
pub struct WasmThompsonEngine {
    inner: MetaThompsonEngine,
}

#[wasm_bindgen]
impl WasmThompsonEngine {
    /// Create a Thompson engine with the given arms.
    #[wasm_bindgen(constructor)]
    pub fn new(arms_json: &str) -> Self {
        let arms: Vec<String> = serde_json::from_str(arms_json).unwrap_or_default();
        Self {
            inner: MetaThompsonEngine::new(arms),
        }
    }

    /// Initialize a domain with uniform priors.
    #[wasm_bindgen(js_name = initDomain)]
    pub fn init_domain(&mut self, domain_id: &str) {
        self.inner
            .init_domain_uniform(DomainId(domain_id.to_string()));
    }

    /// Record an outcome.
    #[wasm_bindgen(js_name = recordOutcome)]
    pub fn record_outcome(
        &mut self,
        domain_id: &str,
        difficulty_tier: &str,
        category: &str,
        arm: &str,
        reward: f32,
        cost: f32,
    ) {
        let bucket = ContextBucket {
            difficulty_tier: difficulty_tier.to_string(),
            category: category.to_string(),
        };
        self.inner.record_outcome(
            &DomainId(domain_id.to_string()),
            bucket,
            ArmId(arm.to_string()),
            reward,
            cost,
        );
    }

    /// Select the best arm.
    #[wasm_bindgen(js_name = selectArm)]
    pub fn select_arm(
        &self,
        domain_id: &str,
        difficulty_tier: &str,
        category: &str,
    ) -> Option<String> {
        let bucket = ContextBucket {
            difficulty_tier: difficulty_tier.to_string(),
            category: category.to_string(),
        };
        let mut rng = rand::thread_rng();
        self.inner
            .select_arm(&DomainId(domain_id.to_string()), &bucket, &mut rng)
            .map(|a| a.0)
    }

    /// Extract transfer prior as JSON.
    #[wasm_bindgen(js_name = extractPrior)]
    pub fn extract_prior(&self, domain_id: &str) -> JsValue {
        match self.inner.extract_prior(&DomainId(domain_id.to_string())) {
            Some(prior) => serde_wasm_bindgen::to_value(&prior).unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }
}

// ─── Population Search ───────────────────────────────────────────────────────

/// WASM-exported population-based policy search.
#[wasm_bindgen]
pub struct WasmPopulationSearch {
    inner: PopulationSearch,
}

#[wasm_bindgen]
impl WasmPopulationSearch {
    /// Create a new population search with given size.
    #[wasm_bindgen(constructor)]
    pub fn new(pop_size: usize) -> Self {
        Self {
            inner: PopulationSearch::new(pop_size),
        }
    }

    /// Get current population size.
    #[wasm_bindgen(js_name = populationSize)]
    pub fn population_size(&self) -> usize {
        self.inner.population().len()
    }

    /// Get current generation.
    pub fn generation(&self) -> u32 {
        self.inner.generation()
    }

    /// Evolve to next generation.
    pub fn evolve(&mut self) {
        self.inner.evolve();
    }

    /// Get stats as JSON.
    pub fn stats(&self) -> JsValue {
        let stats = self.inner.stats();
        serde_wasm_bindgen::to_value(&stats).unwrap_or(JsValue::NULL)
    }
}

// ─── Acceleration Scoreboard ─────────────────────────────────────────────────

/// WASM-exported acceleration scoreboard.
#[wasm_bindgen]
pub struct WasmScoreboard {
    inner: AccelerationScoreboard,
}

#[wasm_bindgen]
impl WasmScoreboard {
    /// Create a new scoreboard.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: AccelerationScoreboard::new(),
        }
    }

    /// Add a cost curve from JSON.
    #[wasm_bindgen(js_name = addCurve)]
    pub fn add_curve(&mut self, curve_json: &str) {
        if let Ok(curve) = serde_json::from_str::<CostCurve>(curve_json) {
            self.inner.add_curve(curve);
        }
    }

    /// Compute acceleration between two domains.
    #[wasm_bindgen(js_name = computeAcceleration)]
    pub fn compute_acceleration(&mut self, baseline: &str, transfer: &str) -> JsValue {
        match self.inner.compute_acceleration(
            &DomainId(baseline.to_string()),
            &DomainId(transfer.to_string()),
        ) {
            Some(entry) => serde_wasm_bindgen::to_value(&entry).unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }

    /// Check if progressive acceleration holds.
    #[wasm_bindgen(js_name = progressiveAcceleration)]
    pub fn progressive_acceleration(&self) -> bool {
        self.inner.progressive_acceleration()
    }

    /// Get full summary as JSON.
    pub fn summary(&self) -> JsValue {
        let s = self.inner.summary();
        serde_wasm_bindgen::to_value(&s).unwrap_or(JsValue::NULL)
    }
}
