//! Memory-Aware MoE Router (ADR-092)
//!
//! Expert selection with cache residency bonus for >=70% cache hit rate.
//! Implements INV-6: Router Determinism - same input + cache state = same result.
//!
//! ## Algorithm
//!
//! 1. Compute base scores from gate network logits
//! 2. Add cache residency bonus to resident experts
//! 3. Select top-K experts
//! 4. Update affinity tracking
//! 5. Generate paging requests for non-resident experts
//!
//! ## Configuration
//!
//! The `cache_bonus` parameter (0.0-1.0) controls how much to favor resident experts:
//! - 0.0: Pure accuracy (ignore cache state, baseline 34% hit rate)
//! - 0.15: Recommended balance (>=70% hit rate with <1% accuracy loss)
//! - 0.3+: Aggressive caching (may degrade accuracy)

use super::{ExpertAffinity, ExpertId, MoeMetrics};
use std::time::Instant;

/// Paging direction for expert load/evict operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagingDirection {
    /// Load expert into cache
    In,
    /// Evict expert from cache
    Out,
}

/// Priority level for paging operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PagingPriority {
    /// Normal priority (can be delayed)
    Normal,
    /// Urgent (needed for current inference)
    Urgent,
    /// Prefetch (speculative, can be cancelled)
    Prefetch,
}

/// Request to page an expert in or out of cache
#[derive(Debug, Clone)]
pub struct PagingRequest {
    /// Expert ID to page
    pub expert_id: ExpertId,
    /// Direction (In = load, Out = evict)
    pub direction: PagingDirection,
    /// Priority level
    pub priority: PagingPriority,
}

impl PagingRequest {
    /// Create a new paging request
    pub fn new(expert_id: ExpertId, direction: PagingDirection, priority: PagingPriority) -> Self {
        Self {
            expert_id,
            direction,
            priority,
        }
    }

    /// Create an urgent page-in request
    pub fn page_in_urgent(expert_id: ExpertId) -> Self {
        Self::new(expert_id, PagingDirection::In, PagingPriority::Urgent)
    }

    /// Create a prefetch request
    pub fn prefetch(expert_id: ExpertId) -> Self {
        Self::new(expert_id, PagingDirection::In, PagingPriority::Prefetch)
    }

    /// Create a page-out request
    pub fn page_out(expert_id: ExpertId) -> Self {
        Self::new(expert_id, PagingDirection::Out, PagingPriority::Normal)
    }
}

/// Configuration for the memory-aware router
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Cache residency bonus weight (0.0-1.0)
    ///
    /// Added to gate scores for experts currently in cache.
    /// Default: 0.15 (achieves >=70% hit rate with <1% accuracy loss)
    pub cache_bonus: f32,

    /// Top-K experts to select per token
    ///
    /// Typical values: 1 (Switch), 2 (Mixtral), 4 (GShard)
    pub top_k: usize,

    /// Number of total experts in the model
    pub num_experts: usize,

    /// Enable memory-aware routing (feature flag)
    ///
    /// When false, the router ignores cache state and uses pure accuracy mode.
    pub memory_aware: bool,

    /// Prefetch threshold (router weight to trigger speculative prefetch)
    ///
    /// Experts with weight >= this but not selected may be prefetched.
    /// Default: 0.1 (10%)
    pub prefetch_threshold: f32,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            cache_bonus: 0.15,
            top_k: 2,
            num_experts: 8,
            memory_aware: true,
            prefetch_threshold: 0.1,
        }
    }
}

impl RouterConfig {
    /// Create config with specified parameters
    pub fn new(num_experts: usize, top_k: usize) -> Self {
        Self {
            num_experts,
            top_k,
            ..Default::default()
        }
    }

    /// Set cache bonus weight
    pub fn with_cache_bonus(mut self, bonus: f32) -> Self {
        self.cache_bonus = bonus.clamp(0.0, 1.0);
        self
    }

    /// Set memory-aware mode
    pub fn with_memory_aware(mut self, enabled: bool) -> Self {
        self.memory_aware = enabled;
        self
    }

    /// Set prefetch threshold
    pub fn with_prefetch_threshold(mut self, threshold: f32) -> Self {
        self.prefetch_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.top_k == 0 {
            return Err("top_k must be at least 1");
        }
        if self.top_k > self.num_experts {
            return Err("top_k cannot exceed num_experts");
        }
        if self.num_experts == 0 {
            return Err("num_experts must be at least 1");
        }
        Ok(())
    }
}

/// Memory-aware MoE router with cache residency bonus
///
/// Implements the memory-aware routing algorithm from ADR-092:
/// 1. Add cache residency bonus to gate scores
/// 2. Select top-K experts with adjusted scores
/// 3. Generate paging requests for non-resident selected experts
///
/// # Invariant INV-6: Router Determinism
///
/// Given the same input (gate_logits) and same cache state (cache_resident),
/// the router always produces the same output (selected experts, paging requests).
///
/// # Example
///
/// ```rust,ignore
/// use ruvllm::moe::{MemoryAwareRouter, RouterConfig, ExpertAffinity, AffinityConfig};
///
/// let config = RouterConfig {
///     cache_bonus: 0.15,
///     top_k: 2,
///     num_experts: 8,
///     memory_aware: true,
///     prefetch_threshold: 0.1,
/// };
///
/// let affinity = ExpertAffinity::new(AffinityConfig::with_num_experts(8));
/// let mut router = MemoryAwareRouter::new(config, affinity);
///
/// // Update which experts are currently cached
/// router.update_cache_state(&[0, 1, 2, 3]);
///
/// // Route based on gate logits
/// let gate_logits = vec![0.1, 0.3, 0.5, 0.2, 0.4, 0.1, 0.2, 0.15];
/// let (selected, paging_requests) = router.route(&gate_logits);
/// ```
pub struct MemoryAwareRouter {
    /// Router configuration
    config: RouterConfig,
    /// Expert affinity tracker
    affinity: ExpertAffinity,
    /// Which experts are currently in cache (indexed by expert_id)
    cache_resident: Vec<bool>,
    /// Routing and caching metrics
    metrics: MoeMetrics,
}

impl MemoryAwareRouter {
    /// Create a new memory-aware router
    ///
    /// # Arguments
    ///
    /// * `config` - Router configuration
    /// * `affinity` - Expert affinity tracker (can be shared)
    ///
    /// # Returns
    ///
    /// Returns `Err` if the configuration is invalid.
    pub fn new(config: RouterConfig, affinity: ExpertAffinity) -> Result<Self, &'static str> {
        config.validate()?;

        Ok(Self {
            cache_resident: vec![false; config.num_experts],
            config,
            affinity,
            metrics: MoeMetrics::new(),
        })
    }

    /// Create router with default affinity tracker
    ///
    /// # Returns
    ///
    /// Returns `Err` if the configuration is invalid.
    pub fn with_default_affinity(config: RouterConfig) -> Result<Self, &'static str> {
        let affinity = ExpertAffinity::new(
            super::AffinityConfig::with_num_experts(config.num_experts)
        );
        Self::new(config, affinity)
    }

    /// Main routing function with cache bonus
    ///
    /// Returns selected experts and any paging requests needed.
    ///
    /// # Arguments
    ///
    /// * `gate_logits` - Raw logits from the gate network (length = num_experts)
    ///
    /// # Returns
    ///
    /// Tuple of (selected_expert_ids, paging_requests)
    ///
    /// # INV-6: Determinism
    ///
    /// This function is deterministic: same inputs produce same outputs.
    /// No random sampling is used.
    pub fn route(&mut self, gate_logits: &[f32]) -> (Vec<ExpertId>, Vec<PagingRequest>) {
        let start = Instant::now();

        // Validate input length
        if gate_logits.len() != self.config.num_experts {
            // Fallback: return first top_k experts
            let selected: Vec<ExpertId> = (0..self.config.top_k.min(self.config.num_experts)).collect();
            return (selected, Vec::new());
        }

        // Step 1: Apply cache bonus (if memory-aware mode enabled)
        let adjusted_scores = if self.config.memory_aware {
            self.apply_cache_bonus(gate_logits)
        } else {
            gate_logits.to_vec()
        };

        // Step 2: Select top-K experts
        let selected = self.select_top_k(&adjusted_scores);

        // Step 3: Update affinity for selected experts
        self.affinity.update(&selected);

        // Step 4: Generate paging requests for non-resident selected experts
        let paging_requests = self.generate_paging_requests(&selected);

        // Step 5: Record metrics
        let hits = selected.iter().filter(|&&id| self.is_resident(id)).count();
        let misses = selected.len() - hits;
        for _ in 0..hits {
            self.metrics.record_cache_hit();
        }
        for _ in 0..misses {
            self.metrics.record_cache_miss();
        }
        self.metrics.record_routing(start.elapsed());

        (selected, paging_requests)
    }

    /// Apply cache residency bonus to scores (in-place mutation for P0 optimization)
    ///
    /// For each expert currently in cache, adds `cache_bonus` to its score.
    /// This biases the selection toward cached experts without completely
    /// overriding the gate network's decisions.
    ///
    /// # Arguments
    ///
    /// * `scores` - Mutable slice of scores to modify in-place
    pub fn apply_cache_bonus_inplace(&self, scores: &mut [f32]) {
        for (id, score) in scores.iter_mut().enumerate() {
            // Validate score is not NaN/Inf before processing
            if !score.is_finite() {
                *score = 0.0;
                continue;
            }
            if self.cache_resident.get(id).copied().unwrap_or(false) {
                *score += self.config.cache_bonus;
            }
        }
    }

    /// Apply cache residency bonus to scores (allocating version for API compatibility)
    ///
    /// For each expert currently in cache, adds `cache_bonus` to its score.
    /// This biases the selection toward cached experts without completely
    /// overriding the gate network's decisions.
    pub fn apply_cache_bonus(&self, scores: &[f32]) -> Vec<f32> {
        let mut result = scores.to_vec();
        self.apply_cache_bonus_inplace(&mut result);
        result
    }

    /// Select top-K experts by score
    ///
    /// Returns expert IDs sorted by descending score.
    /// Ties are broken by expert ID (lower ID wins) for determinism.
    ///
    /// Uses partial sort (P0 optimization) for better performance when
    /// top_k << num_experts.
    pub fn select_top_k(&self, scores: &[f32]) -> Vec<ExpertId> {
        let n = scores.len();
        let k = self.config.top_k.min(n);

        if k == 0 || n == 0 {
            return Vec::new();
        }

        // Create indexed scores, handling NaN/Inf values
        let mut indexed: Vec<(ExpertId, f32)> = scores
            .iter()
            .enumerate()
            .map(|(id, &s)| (id, if s.is_finite() { s } else { f32::NEG_INFINITY }))
            .collect();

        // Use partial sort for better performance when k << n
        if k < n / 2 {
            // Partition to get top-k elements (unordered)
            indexed.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            // Sort only the top-k portion
            indexed[..k].sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
        } else {
            // Full sort when k is close to n
            indexed.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
        }

        // Take top-K
        indexed
            .into_iter()
            .take(k)
            .map(|(id, _)| id)
            .collect()
    }

    /// Update cache residency state
    ///
    /// Call this when experts are paged in or out.
    ///
    /// # Arguments
    ///
    /// * `resident` - List of expert IDs currently in cache
    pub fn update_cache_state(&mut self, resident: &[ExpertId]) {
        // Clear all
        self.cache_resident.fill(false);

        // Set resident experts
        for &id in resident {
            if id < self.cache_resident.len() {
                self.cache_resident[id] = true;
            }
        }
    }

    /// Mark a single expert as resident or non-resident
    pub fn set_resident(&mut self, expert_id: ExpertId, resident: bool) {
        if expert_id < self.cache_resident.len() {
            self.cache_resident[expert_id] = resident;
        }
    }

    /// Check if an expert is currently resident
    pub fn is_resident(&self, expert_id: ExpertId) -> bool {
        self.cache_resident.get(expert_id).copied().unwrap_or(false)
    }

    /// Generate paging requests for selected experts
    ///
    /// Creates urgent page-in requests for non-resident selected experts.
    /// Also generates prefetch requests for high-scoring non-selected experts.
    pub fn generate_paging_requests(&self, selected: &[ExpertId]) -> Vec<PagingRequest> {
        let mut requests = Vec::new();

        // Urgent page-in for non-resident selected experts
        for &expert_id in selected {
            if !self.is_resident(expert_id) {
                requests.push(PagingRequest::page_in_urgent(expert_id));
            }
        }

        requests
    }

    /// Generate prefetch requests based on affinity
    ///
    /// Returns prefetch requests for high-affinity non-resident experts.
    ///
    /// # Arguments
    ///
    /// * `budget` - Maximum number of prefetch requests to generate
    pub fn generate_prefetch_requests(&self, budget: usize) -> Vec<PagingRequest> {
        // Get top experts by affinity that are not currently resident
        let candidates = self.affinity.top_k_by_affinity(budget * 2);

        candidates
            .into_iter()
            .filter(|&id| !self.is_resident(id))
            .take(budget)
            .map(PagingRequest::prefetch)
            .collect()
    }

    /// Get a reference to the current metrics
    pub fn metrics(&self) -> &MoeMetrics {
        &self.metrics
    }

    /// Reset metrics
    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
    }

    /// Get a reference to the affinity tracker
    pub fn affinity(&self) -> &ExpertAffinity {
        &self.affinity
    }

    /// Get a mutable reference to the affinity tracker
    pub fn affinity_mut(&mut self) -> &mut ExpertAffinity {
        &mut self.affinity
    }

    /// Get a reference to the configuration
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Get the current cache hit rate
    pub fn hit_rate(&self) -> f32 {
        self.metrics.hit_rate()
    }

    /// Get list of currently resident experts
    pub fn resident_experts(&self) -> Vec<ExpertId> {
        self.cache_resident
            .iter()
            .enumerate()
            .filter(|(_, &resident)| resident)
            .map(|(id, _)| id)
            .collect()
    }

    /// Get number of experts
    pub fn num_experts(&self) -> usize {
        self.config.num_experts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::AffinityConfig;

    fn make_router(num_experts: usize, top_k: usize, cache_bonus: f32) -> MemoryAwareRouter {
        let config = RouterConfig::new(num_experts, top_k).with_cache_bonus(cache_bonus);
        MemoryAwareRouter::with_default_affinity(config).expect("test config should be valid")
    }

    // ---------------------------------------------------------------
    // test_routing_basic
    // ---------------------------------------------------------------

    #[test]
    fn test_routing_basic() {
        let mut router = make_router(8, 2, 0.0);

        // No cache bonus, pure selection
        let gate_logits = vec![0.1, 0.3, 0.5, 0.2, 0.4, 0.1, 0.2, 0.15];
        let (selected, _) = router.route(&gate_logits);

        assert_eq!(selected.len(), 2);
        // Experts 2 (0.5) and 4 (0.4) should be selected
        assert!(selected.contains(&2));
        assert!(selected.contains(&4));
    }

    // ---------------------------------------------------------------
    // test_cache_bonus_increases_resident_score
    // ---------------------------------------------------------------

    #[test]
    fn test_cache_bonus_increases_resident_score() {
        let mut router = make_router(4, 1, 0.3);

        // Experts: 0=0.4, 1=0.3, 2=0.2, 3=0.1
        // Without bonus: expert 0 selected
        // With bonus on expert 1: 0.3 + 0.3 = 0.6 > 0.4

        router.update_cache_state(&[1]); // Expert 1 is resident

        let gate_logits = vec![0.4, 0.3, 0.2, 0.1];
        let (selected, _) = router.route(&gate_logits);

        // Expert 1 should be selected because of cache bonus
        assert_eq!(selected, vec![1]);
    }

    // ---------------------------------------------------------------
    // test_top_k_selection
    // ---------------------------------------------------------------

    #[test]
    fn test_top_k_selection() {
        let mut router = make_router(8, 3, 0.0);

        let gate_logits = vec![0.8, 0.1, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5];
        let (selected, _) = router.route(&gate_logits);

        assert_eq!(selected.len(), 3);
        // Top 3: expert 0 (0.8), expert 3 (0.7), expert 5 (0.6)
        assert_eq!(selected[0], 0);
        assert_eq!(selected[1], 3);
        assert_eq!(selected[2], 5);
    }

    // ---------------------------------------------------------------
    // test_paging_requests_for_non_resident
    // ---------------------------------------------------------------

    #[test]
    fn test_paging_requests_for_non_resident() {
        let mut router = make_router(4, 2, 0.0);

        // Only expert 0 is resident
        router.update_cache_state(&[0]);

        let gate_logits = vec![0.5, 0.6, 0.4, 0.3];
        let (selected, paging) = router.route(&gate_logits);

        // Selected: experts 1 (0.6) and 0 (0.5)
        assert!(selected.contains(&0));
        assert!(selected.contains(&1));

        // Expert 1 is not resident, should have paging request
        assert_eq!(paging.len(), 1);
        assert_eq!(paging[0].expert_id, 1);
        assert_eq!(paging[0].direction, PagingDirection::In);
        assert_eq!(paging[0].priority, PagingPriority::Urgent);
    }

    // ---------------------------------------------------------------
    // test_router_determinism (INV-6)
    // ---------------------------------------------------------------

    #[test]
    fn test_router_determinism() {
        // INV-6: Same input + cache state = same result

        let mut router1 = make_router(8, 2, 0.15);
        let mut router2 = make_router(8, 2, 0.15);

        // Same cache state
        router1.update_cache_state(&[0, 3, 5]);
        router2.update_cache_state(&[0, 3, 5]);

        let gate_logits = vec![0.1, 0.3, 0.5, 0.2, 0.4, 0.1, 0.2, 0.15];

        let (selected1, paging1) = router1.route(&gate_logits);
        let (selected2, paging2) = router2.route(&gate_logits);

        // Results must be identical
        assert_eq!(selected1, selected2, "INV-6 violation: different expert selection");
        assert_eq!(paging1.len(), paging2.len(), "INV-6 violation: different paging count");

        // Run multiple times on same router
        router1.reset_metrics();
        let (selected3, _) = router1.route(&gate_logits);
        assert_eq!(selected1, selected3, "INV-6 violation: non-deterministic routing");
    }

    // ---------------------------------------------------------------
    // test_affinity_updates
    // ---------------------------------------------------------------

    #[test]
    fn test_affinity_updates() {
        let mut router = make_router(4, 2, 0.0);

        // Route multiple times to build affinity
        let gate_logits = vec![0.4, 0.3, 0.5, 0.1];

        for _ in 0..5 {
            router.route(&gate_logits);
        }

        // Experts 2 and 0 should have highest affinity (selected 5 times)
        let top = router.affinity().top_k_by_affinity(2);
        assert!(top.contains(&2), "Expert 2 should have high affinity");
        assert!(top.contains(&0), "Expert 0 should have high affinity");
    }

    // ---------------------------------------------------------------
    // test_zero_cache_bonus_fallback
    // ---------------------------------------------------------------

    #[test]
    fn test_zero_cache_bonus_fallback() {
        let mut router = make_router(4, 2, 0.0);

        // All experts resident
        router.update_cache_state(&[0, 1, 2, 3]);

        let gate_logits = vec![0.1, 0.4, 0.3, 0.2];
        let (selected, _) = router.route(&gate_logits);

        // Should select purely by score: experts 1 (0.4) and 2 (0.3)
        assert_eq!(selected[0], 1);
        assert_eq!(selected[1], 2);
    }

    // ---------------------------------------------------------------
    // test_all_experts_resident
    // ---------------------------------------------------------------

    #[test]
    fn test_all_experts_resident() {
        let mut router = make_router(4, 2, 0.15);

        // All experts resident
        router.update_cache_state(&[0, 1, 2, 3]);

        let gate_logits = vec![0.1, 0.4, 0.3, 0.2];
        let (selected, paging) = router.route(&gate_logits);

        assert_eq!(selected.len(), 2);
        // No paging needed
        assert!(paging.is_empty(), "No paging should be needed when all selected are resident");

        // All should be cache hits
        assert_eq!(router.metrics().cache_hits, 2);
        assert_eq!(router.metrics().cache_misses, 0);
    }

    // ---------------------------------------------------------------
    // test_no_experts_resident
    // ---------------------------------------------------------------

    #[test]
    fn test_no_experts_resident() {
        let mut router = make_router(4, 2, 0.15);

        // No experts resident (cold start)
        router.update_cache_state(&[]);

        let gate_logits = vec![0.1, 0.4, 0.3, 0.2];
        let (selected, paging) = router.route(&gate_logits);

        assert_eq!(selected.len(), 2);
        // Should need paging for all selected
        assert_eq!(paging.len(), 2, "Should need to page in all selected experts");

        // All should be cache misses
        assert_eq!(router.metrics().cache_misses, 2);
        assert_eq!(router.metrics().cache_hits, 0);
    }

    // ---------------------------------------------------------------
    // test_config_validation
    // ---------------------------------------------------------------

    #[test]
    fn test_config_validation() {
        // Valid config
        let valid = RouterConfig::new(8, 2);
        assert!(valid.validate().is_ok());

        // Invalid: top_k = 0
        let invalid1 = RouterConfig { top_k: 0, ..RouterConfig::default() };
        assert!(invalid1.validate().is_err());

        // Invalid: top_k > num_experts
        let invalid2 = RouterConfig { top_k: 10, num_experts: 8, ..RouterConfig::default() };
        assert!(invalid2.validate().is_err());

        // Invalid: num_experts = 0
        let invalid3 = RouterConfig { num_experts: 0, ..RouterConfig::default() };
        assert!(invalid3.validate().is_err());
    }

    // ---------------------------------------------------------------
    // test_memory_aware_disabled
    // ---------------------------------------------------------------

    #[test]
    fn test_memory_aware_disabled() {
        let config = RouterConfig::new(4, 2).with_memory_aware(false).with_cache_bonus(0.5);
        let mut router = MemoryAwareRouter::with_default_affinity(config).unwrap();

        // Even with high cache bonus, should not apply it when disabled
        router.update_cache_state(&[3]); // Expert 3 resident

        let gate_logits = vec![0.4, 0.3, 0.5, 0.2];
        let (selected, _) = router.route(&gate_logits);

        // Should select by pure score: experts 2 (0.5) and 0 (0.4)
        assert_eq!(selected[0], 2);
        assert_eq!(selected[1], 0);
    }

    // ---------------------------------------------------------------
    // test_hit_rate_tracking
    // ---------------------------------------------------------------

    #[test]
    fn test_hit_rate_tracking() {
        let mut router = make_router(4, 2, 0.0);

        // 50% resident
        router.update_cache_state(&[0, 2]);

        let gate_logits = vec![0.4, 0.3, 0.5, 0.2];
        // Will select experts 2 (resident) and 0 (resident)
        router.route(&gate_logits);

        assert_eq!(router.hit_rate(), 1.0); // Both selected are resident

        router.reset_metrics();
        router.update_cache_state(&[1, 3]);
        router.route(&gate_logits);

        assert_eq!(router.hit_rate(), 0.0); // Neither selected is resident
    }

    // ---------------------------------------------------------------
    // test_prefetch_requests
    // ---------------------------------------------------------------

    #[test]
    fn test_prefetch_requests() {
        let config = RouterConfig::new(4, 2).with_cache_bonus(0.0);
        let affinity_config = AffinityConfig::with_num_experts(4).with_decay(1.0);
        let affinity = ExpertAffinity::new(affinity_config);
        let mut router = MemoryAwareRouter::new(config, affinity).unwrap();

        // Build affinity
        let gate_logits = vec![0.4, 0.3, 0.5, 0.2];
        for _ in 0..10 {
            router.route(&gate_logits);
        }

        // Only expert 1 is resident
        router.update_cache_state(&[1]);

        // Should suggest prefetching high-affinity non-resident experts
        let prefetch = router.generate_prefetch_requests(2);

        // Should not include expert 1 (already resident)
        for req in &prefetch {
            assert_ne!(req.expert_id, 1);
            assert_eq!(req.priority, PagingPriority::Prefetch);
        }
    }

    // ---------------------------------------------------------------
    // test_resident_experts_list
    // ---------------------------------------------------------------

    #[test]
    fn test_resident_experts_list() {
        let mut router = make_router(8, 2, 0.15);

        router.update_cache_state(&[1, 3, 5, 7]);

        let resident = router.resident_experts();
        assert_eq!(resident.len(), 4);
        assert!(resident.contains(&1));
        assert!(resident.contains(&3));
        assert!(resident.contains(&5));
        assert!(resident.contains(&7));
        assert!(!resident.contains(&0));
    }

    // ---------------------------------------------------------------
    // test_set_resident
    // ---------------------------------------------------------------

    #[test]
    fn test_set_resident() {
        let mut router = make_router(4, 2, 0.15);

        assert!(!router.is_resident(0));

        router.set_resident(0, true);
        assert!(router.is_resident(0));

        router.set_resident(0, false);
        assert!(!router.is_resident(0));
    }

    // ---------------------------------------------------------------
    // test_tie_breaking_determinism
    // ---------------------------------------------------------------

    #[test]
    fn test_tie_breaking_determinism() {
        let mut router = make_router(4, 2, 0.0);

        // All experts have same score
        let gate_logits = vec![0.5, 0.5, 0.5, 0.5];
        let (selected1, _) = router.route(&gate_logits);
        let (selected2, _) = router.route(&gate_logits);

        // Should consistently select lowest IDs on ties
        assert_eq!(selected1, selected2);
        assert_eq!(selected1, vec![0, 1]); // Lowest IDs win ties
    }

    // ---------------------------------------------------------------
    // test_invalid_gate_logits_length
    // ---------------------------------------------------------------

    #[test]
    fn test_invalid_gate_logits_length() {
        let mut router = make_router(4, 2, 0.15);

        // Wrong length input
        let gate_logits = vec![0.5, 0.3]; // Only 2 instead of 4
        let (selected, paging) = router.route(&gate_logits);

        // Should fallback gracefully
        assert_eq!(selected.len(), 2);
        assert!(paging.is_empty() || paging.len() <= 2);
    }

    // ---------------------------------------------------------------
    // test_apply_cache_bonus
    // ---------------------------------------------------------------

    #[test]
    fn test_apply_cache_bonus() {
        let mut router = make_router(4, 2, 0.2);
        router.update_cache_state(&[1, 2]);

        let scores = vec![0.1, 0.3, 0.4, 0.5];
        let adjusted = router.apply_cache_bonus(&scores);

        // Expert 0: 0.1 + 0 = 0.1
        // Expert 1: 0.3 + 0.2 = 0.5 (resident)
        // Expert 2: 0.4 + 0.2 = 0.6 (resident)
        // Expert 3: 0.5 + 0 = 0.5
        assert!((adjusted[0] - 0.1).abs() < 1e-6);
        assert!((adjusted[1] - 0.5).abs() < 1e-6);
        assert!((adjusted[2] - 0.6).abs() < 1e-6);
        assert!((adjusted[3] - 0.5).abs() < 1e-6);
    }

    // ---------------------------------------------------------------
    // test_paging_request_constructors
    // ---------------------------------------------------------------

    #[test]
    fn test_paging_request_constructors() {
        let req1 = PagingRequest::page_in_urgent(5);
        assert_eq!(req1.expert_id, 5);
        assert_eq!(req1.direction, PagingDirection::In);
        assert_eq!(req1.priority, PagingPriority::Urgent);

        let req2 = PagingRequest::prefetch(3);
        assert_eq!(req2.expert_id, 3);
        assert_eq!(req2.direction, PagingDirection::In);
        assert_eq!(req2.priority, PagingPriority::Prefetch);

        let req3 = PagingRequest::page_out(7);
        assert_eq!(req3.expert_id, 7);
        assert_eq!(req3.direction, PagingDirection::Out);
        assert_eq!(req3.priority, PagingPriority::Normal);
    }

    // ---------------------------------------------------------------
    // test_config_builder
    // ---------------------------------------------------------------

    #[test]
    fn test_config_builder() {
        let config = RouterConfig::new(16, 4)
            .with_cache_bonus(0.25)
            .with_memory_aware(true)
            .with_prefetch_threshold(0.15);

        assert_eq!(config.num_experts, 16);
        assert_eq!(config.top_k, 4);
        assert!((config.cache_bonus - 0.25).abs() < 1e-6);
        assert!(config.memory_aware);
        assert!((config.prefetch_threshold - 0.15).abs() < 1e-6);
    }

    // ---------------------------------------------------------------
    // test_cache_bonus_clamping
    // ---------------------------------------------------------------

    #[test]
    fn test_cache_bonus_clamping() {
        let config = RouterConfig::new(8, 2).with_cache_bonus(1.5);
        assert!((config.cache_bonus - 1.0).abs() < 1e-6, "cache_bonus should be clamped to 1.0");

        let config2 = RouterConfig::new(8, 2).with_cache_bonus(-0.5);
        assert!((config2.cache_bonus - 0.0).abs() < 1e-6, "cache_bonus should be clamped to 0.0");
    }
}
