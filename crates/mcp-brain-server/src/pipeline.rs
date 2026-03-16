//! Cloud-native data pipeline for real-time injection and optimization.
//!
//! Integrates with Google Cloud Pub/Sub for event-driven ingestion,
//! Cloud Scheduler for periodic optimization, and provides metrics
//! for Cloud Monitoring.
//!
//! Also contains the RVF container construction pipeline (ADR-075 Phase 5).

use chrono::{DateTime, Utc};
use rvf_types::SegmentFlags;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;

// ═══════════════════════════════════════════════════════════════════════
// RVF Container Construction (preserved from ADR-075 Phase 5)
// ═══════════════════════════════════════════════════════════════════════

/// Input data for building an RVF container.
pub struct RvfPipelineInput<'a> {
    pub memory_id: &'a str,
    pub embedding: &'a [f32],
    pub title: &'a str,
    pub content: &'a str,
    pub tags: &'a [String],
    pub category: &'a str,
    pub contributor_id: &'a str,
    pub witness_chain: Option<&'a [u8]>,
    pub dp_proof_json: Option<&'a str>,
    pub redaction_log_json: Option<&'a str>,
}

/// Build an RVF container from pipeline input.
/// Returns the serialized container bytes (concatenated 64-byte-aligned segments).
pub fn build_rvf_container(input: &RvfPipelineInput<'_>) -> Result<Vec<u8>, String> {
    let flags = SegmentFlags::empty();
    let mut container = Vec::new();
    let mut seg_id: u64 = 1;

    // Segment 1: VEC (0x01)
    {
        let mut payload = Vec::with_capacity(input.embedding.len() * 4);
        for &val in input.embedding {
            payload.extend_from_slice(&val.to_le_bytes());
        }
        let seg = rvf_wire::write_segment(0x01, &payload, flags, seg_id);
        container.extend_from_slice(&seg);
        seg_id += 1;
    }

    // Segment 2: META (0x07)
    {
        let meta = serde_json::json!({
            "memory_id": input.memory_id,
            "title": input.title,
            "content": input.content,
            "tags": input.tags,
            "category": input.category,
            "contributor_id": input.contributor_id,
        });
        let payload = serde_json::to_vec(&meta)
            .map_err(|e| format!("Failed to serialize RVF metadata: {e}"))?;
        let seg = rvf_wire::write_segment(0x07, &payload, flags, seg_id);
        container.extend_from_slice(&seg);
        seg_id += 1;
    }

    // Segment 3: WITNESS (0x0A)
    if let Some(chain) = input.witness_chain {
        let seg = rvf_wire::write_segment(0x0A, chain, flags, seg_id);
        container.extend_from_slice(&seg);
        seg_id += 1;
    }

    // Segment 4: DiffPrivacyProof (0x34)
    if let Some(proof) = input.dp_proof_json {
        let seg = rvf_wire::write_segment(0x34, proof.as_bytes(), flags, seg_id);
        container.extend_from_slice(&seg);
        seg_id += 1;
    }

    // Segment 5: RedactionLog (0x35)
    if let Some(log) = input.redaction_log_json {
        let seg = rvf_wire::write_segment(0x35, log.as_bytes(), flags, seg_id);
        container.extend_from_slice(&seg);
        let _ = seg_id;
    }

    Ok(container)
}

/// Count the number of segments in a serialized RVF container.
pub fn count_segments(container: &[u8]) -> usize {
    let mut count = 0;
    let mut offset = 0;
    while offset + 64 <= container.len() {
        let payload_len = u64::from_le_bytes(
            container[offset + 16..offset + 24]
                .try_into()
                .unwrap_or([0u8; 8]),
        ) as usize;
        let padded = rvf_wire::calculate_padded_size(64, payload_len);
        count += 1;
        offset += padded;
    }
    count
}

// ═══════════════════════════════════════════════════════════════════════
// Cloud Pub/Sub Integration
// ═══════════════════════════════════════════════════════════════════════

/// A decoded Pub/Sub message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PubSubMessage {
    /// Raw decoded payload bytes (base64-decoded from the push envelope).
    pub data: Vec<u8>,
    /// Message attributes from Pub/Sub.
    pub attributes: HashMap<String, String>,
    /// Pub/Sub message ID for acknowledgment.
    pub message_id: String,
    /// Publish timestamp.
    pub publish_time: Option<DateTime<Utc>>,
}

/// Push envelope from Cloud Pub/Sub (HTTP POST body).
#[derive(Debug, Deserialize)]
pub struct PubSubPushEnvelope {
    pub message: PubSubPushMessage,
    pub subscription: String,
}

#[derive(Debug, Deserialize)]
pub struct PubSubPushMessage {
    pub data: Option<String>,
    #[serde(default)]
    pub attributes: HashMap<String, String>,
    #[serde(rename = "messageId")]
    pub message_id: String,
    #[serde(rename = "publishTime")]
    pub publish_time: Option<DateTime<Utc>>,
}

/// Client for Google Cloud Pub/Sub pull-based message retrieval.
#[derive(Debug)]
pub struct PubSubClient {
    project_id: String,
    subscription_id: String,
    http: reqwest::Client,
    use_metadata_server: bool,
}

impl PubSubClient {
    pub fn new(project_id: String, subscription_id: String) -> Self {
        let use_metadata_server = std::env::var("PUBSUB_EMULATOR_HOST").is_err();
        Self {
            project_id,
            subscription_id,
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
            use_metadata_server,
        }
    }

    /// Decode a push-envelope into a `PubSubMessage`.
    pub fn decode_push(envelope: PubSubPushEnvelope) -> Result<PubSubMessage, String> {
        use base64::Engine;
        let data = match envelope.message.data {
            Some(b64) => base64::engine::general_purpose::STANDARD
                .decode(&b64)
                .map_err(|e| format!("base64 decode failed: {e}"))?,
            None => Vec::new(),
        };
        Ok(PubSubMessage {
            data,
            attributes: envelope.message.attributes,
            message_id: envelope.message.message_id,
            publish_time: envelope.message.publish_time,
        })
    }

    /// Acknowledge a message by its ack_id (pull mode only).
    pub async fn acknowledge(&self, ack_ids: &[String]) -> Result<(), String> {
        if ack_ids.is_empty() {
            return Ok(());
        }
        let url = format!(
            "https://pubsub.googleapis.com/v1/projects/{}/subscriptions/{}:acknowledge",
            self.project_id, self.subscription_id
        );
        let body = serde_json::json!({ "ackIds": ack_ids });
        let mut req = self.http.post(&url).json(&body);
        if self.use_metadata_server {
            if let Some(token) = get_metadata_token(&self.http).await {
                req = req.bearer_auth(token);
            }
        }
        let resp = req.send().await.map_err(|e| format!("ack failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("ack returned {}", resp.status()));
        }
        Ok(())
    }
}

/// Fetch an access token from the GCE metadata server.
async fn get_metadata_token(http: &reqwest::Client) -> Option<String> {
    let url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";
    let resp = http
        .get(url)
        .header("Metadata-Flavor", "Google")
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    #[derive(Deserialize)]
    struct TokenResp {
        access_token: String,
    }
    let tr: TokenResp = resp.json().await.ok()?;
    Some(tr.access_token)
}

// ═══════════════════════════════════════════════════════════════════════
// Data Injection Pipeline
// ═══════════════════════════════════════════════════════════════════════

/// Source of injected data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum InjectionSource {
    PubSub,
    BatchUpload,
    RssFeed,
    Webhook,
}

/// An item flowing through the injection pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionItem {
    pub source: InjectionSource,
    pub title: String,
    pub content: String,
    pub category: Option<String>,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, String>,
    pub received_at: DateTime<Utc>,
}

/// Result of pipeline processing for a single item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionResult {
    pub item_hash: String,
    pub accepted: bool,
    pub duplicate: bool,
    pub stage_reached: String,
    pub error: Option<String>,
}

/// Processes incoming data from multiple sources through the pipeline:
/// validate -> embed -> dedup -> store -> graph-update -> train-check
#[derive(Debug)]
pub struct DataInjector {
    /// SHA-256 content hashes of previously ingested items (dedup set).
    seen_hashes: dashmap::DashMap<String, DateTime<Utc>>,
    /// Counter of new items since last training cycle.
    new_items_since_train: AtomicU64,
}

impl DataInjector {
    pub fn new() -> Self {
        Self {
            seen_hashes: dashmap::DashMap::new(),
            new_items_since_train: AtomicU64::new(0),
        }
    }

    /// Compute a content hash for deduplication.
    pub fn content_hash(title: &str, content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(title.as_bytes());
        hasher.update(b"|");
        hasher.update(content.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Run the injection pipeline for a single item.
    /// Returns the result indicating which stage was reached.
    pub fn process(&self, item: &InjectionItem) -> InjectionResult {
        // Stage 1: validate
        if item.title.is_empty() || item.content.is_empty() {
            return InjectionResult {
                item_hash: String::new(),
                accepted: false,
                duplicate: false,
                stage_reached: "validate".into(),
                error: Some("title and content must be non-empty".into()),
            };
        }

        // Stage 2: dedup via content hash
        let hash = Self::content_hash(&item.title, &item.content);
        if self.seen_hashes.contains_key(&hash) {
            return InjectionResult {
                item_hash: hash,
                accepted: false,
                duplicate: true,
                stage_reached: "dedup".into(),
                error: None,
            };
        }

        // Mark as seen
        self.seen_hashes.insert(hash.clone(), Utc::now());
        self.new_items_since_train.fetch_add(1, Ordering::Relaxed);

        // Stages 3-5 (embed, store, graph-update) are performed by the caller
        // using the brain's existing FirestoreClient and graph modules.
        InjectionResult {
            item_hash: hash,
            accepted: true,
            duplicate: false,
            stage_reached: "ready_for_embed".into(),
            error: None,
        }
    }

    /// Number of new items since the last training cycle.
    pub fn new_items_count(&self) -> u64 {
        self.new_items_since_train.load(Ordering::Relaxed)
    }

    /// Reset the new-items counter (called after a training cycle).
    pub fn reset_train_counter(&self) {
        self.new_items_since_train.store(0, Ordering::Relaxed);
    }

    /// Total number of unique content hashes seen.
    pub fn dedup_set_size(&self) -> usize {
        self.seen_hashes.len()
    }
}

impl Default for DataInjector {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Optimization Scheduler
// ═══════════════════════════════════════════════════════════════════════

/// Configuration for optimization cycle intervals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Train after this many new memories.
    pub train_item_threshold: u64,
    /// Train after this many seconds of inactivity.
    pub train_interval_secs: u64,
    /// Drift monitoring interval in seconds.
    pub drift_interval_secs: u64,
    /// Cross-domain transfer interval in seconds.
    pub transfer_interval_secs: u64,
    /// Graph rebalancing interval in seconds.
    pub graph_rebalance_secs: u64,
    /// Memory cleanup interval in seconds.
    pub cleanup_interval_secs: u64,
    /// Attractor analysis interval in seconds.
    pub attractor_interval_secs: u64,
    /// Quality threshold below which memories are pruned.
    pub prune_quality_threshold: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            train_item_threshold: 100,
            train_interval_secs: 300,       // 5 minutes
            drift_interval_secs: 900,       // 15 minutes
            transfer_interval_secs: 1800,   // 30 minutes
            graph_rebalance_secs: 3600,     // 1 hour
            cleanup_interval_secs: 86400,   // 24 hours
            attractor_interval_secs: 1200,  // 20 minutes
            prune_quality_threshold: 0.3,
        }
    }
}

/// Tracks timestamps and counters to decide when optimization tasks fire.
#[derive(Debug)]
pub struct OptimizationScheduler {
    pub config: SchedulerConfig,
    last_train: RwLock<DateTime<Utc>>,
    last_drift_check: RwLock<DateTime<Utc>>,
    last_transfer: RwLock<DateTime<Utc>>,
    last_graph_rebalance: RwLock<DateTime<Utc>>,
    last_cleanup: RwLock<DateTime<Utc>>,
    last_attractor: RwLock<DateTime<Utc>>,
    cycles_completed: AtomicU64,
}

impl OptimizationScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        let now = Utc::now();
        Self {
            config,
            last_train: RwLock::new(now),
            last_drift_check: RwLock::new(now),
            last_transfer: RwLock::new(now),
            last_graph_rebalance: RwLock::new(now),
            last_cleanup: RwLock::new(now),
            last_attractor: RwLock::new(now),
            cycles_completed: AtomicU64::new(0),
        }
    }

    /// Check which optimization tasks are due and return their names.
    pub async fn due_tasks(&self, new_item_count: u64) -> Vec<String> {
        let now = Utc::now();
        let mut due = Vec::new();

        let secs_since = |ts: &DateTime<Utc>| (now - *ts).num_seconds().max(0) as u64;

        if new_item_count >= self.config.train_item_threshold
            || secs_since(&*self.last_train.read().await) >= self.config.train_interval_secs
        {
            due.push("training".into());
        }
        if secs_since(&*self.last_drift_check.read().await) >= self.config.drift_interval_secs {
            due.push("drift_monitoring".into());
        }
        if secs_since(&*self.last_transfer.read().await) >= self.config.transfer_interval_secs {
            due.push("cross_domain_transfer".into());
        }
        if secs_since(&*self.last_graph_rebalance.read().await) >= self.config.graph_rebalance_secs
        {
            due.push("graph_rebalancing".into());
        }
        if secs_since(&*self.last_cleanup.read().await) >= self.config.cleanup_interval_secs {
            due.push("memory_cleanup".into());
        }
        if secs_since(&*self.last_attractor.read().await) >= self.config.attractor_interval_secs {
            due.push("attractor_analysis".into());
        }

        due
    }

    /// Mark a task as completed, updating its timestamp.
    pub async fn mark_completed(&self, task: &str) {
        let now = Utc::now();
        match task {
            "training" => *self.last_train.write().await = now,
            "drift_monitoring" => *self.last_drift_check.write().await = now,
            "cross_domain_transfer" => *self.last_transfer.write().await = now,
            "graph_rebalancing" => *self.last_graph_rebalance.write().await = now,
            "memory_cleanup" => *self.last_cleanup.write().await = now,
            "attractor_analysis" => *self.last_attractor.write().await = now,
            _ => {}
        }
        self.cycles_completed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn cycles_completed(&self) -> u64 {
        self.cycles_completed.load(Ordering::Relaxed)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Health & Metrics
// ═══════════════════════════════════════════════════════════════════════

/// Pipeline metrics for Cloud Monitoring.
#[derive(Debug, Serialize)]
pub struct PipelineMetrics {
    pub messages_received: u64,
    pub messages_processed: u64,
    pub messages_failed: u64,
    pub injections_per_minute: f64,
    pub last_training_time: Option<DateTime<Utc>>,
    pub last_drift_check: Option<DateTime<Utc>>,
    pub last_transfer: Option<DateTime<Utc>>,
    pub queue_depth: u64,
    pub optimization_cycles_completed: u64,
}

/// Atomic counters for thread-safe metric collection.
#[derive(Debug)]
pub struct MetricsCollector {
    received: AtomicU64,
    processed: AtomicU64,
    failed: AtomicU64,
    queue_depth: AtomicU64,
    /// Rolling window: (timestamp_secs, count) pairs for injections/min.
    recent_injections: RwLock<Vec<(i64, u64)>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            received: AtomicU64::new(0),
            processed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            queue_depth: AtomicU64::new(0),
            recent_injections: RwLock::new(Vec::new()),
        }
    }

    pub fn record_received(&self) {
        self.received.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_processed(&self) {
        self.processed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_failed(&self) {
        self.failed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_queue_depth(&self, depth: u64) {
        self.queue_depth.store(depth, Ordering::Relaxed);
    }

    /// Record an injection event for the rolling-window rate.
    pub async fn record_injection(&self) {
        let now = Utc::now().timestamp();
        let mut window = self.recent_injections.write().await;
        window.push((now, 1));
        // Keep only the last 5 minutes
        let cutoff = now - 300;
        window.retain(|(ts, _)| *ts >= cutoff);
    }

    /// Compute injections per minute over the last 5 minutes.
    pub async fn injections_per_minute(&self) -> f64 {
        let window = self.recent_injections.read().await;
        if window.is_empty() {
            return 0.0;
        }
        let total: u64 = window.iter().map(|(_, c)| c).sum();
        let now = Utc::now().timestamp();
        let oldest = window.first().map(|(ts, _)| *ts).unwrap_or(now);
        let span_mins = ((now - oldest) as f64 / 60.0).max(1.0 / 60.0);
        total as f64 / span_mins
    }

    /// Snapshot current metrics. Scheduler timestamps come from the caller.
    pub async fn snapshot(
        &self,
        scheduler: &OptimizationScheduler,
    ) -> PipelineMetrics {
        PipelineMetrics {
            messages_received: self.received.load(Ordering::Relaxed),
            messages_processed: self.processed.load(Ordering::Relaxed),
            messages_failed: self.failed.load(Ordering::Relaxed),
            injections_per_minute: self.injections_per_minute().await,
            last_training_time: Some(*scheduler.last_train.read().await),
            last_drift_check: Some(*scheduler.last_drift_check.read().await),
            last_transfer: Some(*scheduler.last_transfer.read().await),
            queue_depth: self.queue_depth.load(Ordering::Relaxed),
            optimization_cycles_completed: scheduler.cycles_completed(),
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Feed Ingestion (RSS/Atom)
// ═══════════════════════════════════════════════════════════════════════

/// Configuration for a single feed source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedSource {
    pub url: String,
    pub poll_interval_secs: u64,
    pub default_category: Option<String>,
    pub default_tags: Vec<String>,
}

/// A parsed feed entry ready for injection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedEntry {
    pub title: String,
    pub content: String,
    pub link: Option<String>,
    pub published: Option<DateTime<Utc>>,
    pub content_hash: String,
    pub source_url: String,
    pub category: Option<String>,
    pub tags: Vec<String>,
}

/// Ingests RSS/Atom feeds and converts entries to `InjectionItem`s.
#[derive(Debug)]
pub struct FeedIngester {
    sources: Vec<FeedSource>,
    /// Last poll time per feed URL.
    last_poll: HashMap<String, DateTime<Utc>>,
    /// Content hashes already seen (dedup).
    seen_hashes: dashmap::DashMap<String, ()>,
    http: reqwest::Client,
}

impl FeedIngester {
    pub fn new(sources: Vec<FeedSource>) -> Self {
        Self {
            last_poll: sources.iter().map(|s| (s.url.clone(), Utc::now())).collect(),
            sources,
            seen_hashes: dashmap::DashMap::new(),
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Check which feeds are due for polling.
    pub fn feeds_due(&self) -> Vec<&FeedSource> {
        let now = Utc::now();
        self.sources
            .iter()
            .filter(|s| {
                let last = self.last_poll.get(&s.url).copied().unwrap_or(Utc::now());
                (now - last).num_seconds().max(0) as u64 >= s.poll_interval_secs
            })
            .collect()
    }

    /// Fetch and parse a feed URL, returning new (non-duplicate) entries.
    /// Uses simple XML tag extraction — no external XML crate required.
    pub async fn fetch_feed(&self, source: &FeedSource) -> Result<Vec<FeedEntry>, String> {
        let resp = self
            .http
            .get(&source.url)
            .header("Accept", "application/rss+xml, application/atom+xml, text/xml")
            .send()
            .await
            .map_err(|e| format!("feed fetch failed for {}: {e}", source.url))?;

        if !resp.status().is_success() {
            return Err(format!("feed {} returned {}", source.url, resp.status()));
        }

        let body = resp
            .text()
            .await
            .map_err(|e| format!("feed body read failed: {e}"))?;

        let entries = self.parse_feed_xml(&body, source);
        let mut new_entries = Vec::new();
        for entry in entries {
            if !self.seen_hashes.contains_key(&entry.content_hash) {
                self.seen_hashes.insert(entry.content_hash.clone(), ());
                new_entries.push(entry);
            }
        }
        Ok(new_entries)
    }

    /// Minimal XML tag extraction for RSS <item> and Atom <entry> elements.
    fn parse_feed_xml(&self, xml: &str, source: &FeedSource) -> Vec<FeedEntry> {
        let mut entries = Vec::new();

        // Try RSS <item> blocks first, then Atom <entry> blocks.
        let blocks: Vec<&str> = if xml.contains("<item>") || xml.contains("<item ") {
            xml.split("<item")
                .skip(1)
                .filter_map(|s| s.split("</item>").next())
                .collect()
        } else {
            xml.split("<entry")
                .skip(1)
                .filter_map(|s| s.split("</entry>").next())
                .collect()
        };

        for block in blocks {
            let title = extract_tag(block, "title").unwrap_or_default();
            let content = extract_tag(block, "description")
                .or_else(|| extract_tag(block, "content"))
                .or_else(|| extract_tag(block, "summary"))
                .unwrap_or_default();
            let link = extract_tag(block, "link");

            if title.is_empty() && content.is_empty() {
                continue;
            }

            let hash = DataInjector::content_hash(&title, &content);
            entries.push(FeedEntry {
                title,
                content,
                link,
                published: None,
                content_hash: hash,
                source_url: source.url.clone(),
                category: source.default_category.clone(),
                tags: source.default_tags.clone(),
            });
        }

        entries
    }

    /// Convert a `FeedEntry` into an `InjectionItem`.
    pub fn to_injection_item(entry: &FeedEntry) -> InjectionItem {
        let mut metadata = HashMap::new();
        if let Some(ref link) = entry.link {
            metadata.insert("source_link".into(), link.clone());
        }
        metadata.insert("source_url".into(), entry.source_url.clone());
        metadata.insert("content_hash".into(), entry.content_hash.clone());

        InjectionItem {
            source: InjectionSource::RssFeed,
            title: entry.title.clone(),
            content: entry.content.clone(),
            category: entry.category.clone(),
            tags: entry.tags.clone(),
            metadata,
            received_at: entry.published.unwrap_or_else(Utc::now),
        }
    }

    /// Number of unique entries seen so far.
    pub fn seen_count(&self) -> usize {
        self.seen_hashes.len()
    }
}

/// Extract text between `<tag>...</tag>` (simple, no nested same-name tags).
fn extract_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let start = xml.find(&open)?;
    let after_open = &xml[start..];
    // Skip past the opening tag's `>`
    let content_start = after_open.find('>')? + 1;
    let inner = &after_open[content_start..];
    let end = inner.find(&close)?;
    let text = inner[..end].trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // -- RVF container tests (preserved) --

    #[test]
    fn test_rvf_container_has_segments() {
        let embedding = vec![0.1f32, 0.2, 0.3, 0.4];
        let tags = vec!["test".to_string()];
        let witness_chain = rvf_crypto::create_witness_chain(&[
            rvf_crypto::WitnessEntry {
                prev_hash: [0u8; 32],
                action_hash: rvf_crypto::shake256_256(b"test"),
                timestamp_ns: 1000,
                witness_type: 0x01,
            },
        ]);
        let dp_proof = r#"{"epsilon":1.0,"delta":1e-5}"#;
        let redaction_log = r#"{"entries":[],"total_redactions":0}"#;

        let input = RvfPipelineInput {
            memory_id: "test-id",
            embedding: &embedding,
            title: "Test Title",
            content: "Test content",
            tags: &tags,
            category: "pattern",
            contributor_id: "test-contributor",
            witness_chain: Some(&witness_chain),
            dp_proof_json: Some(dp_proof),
            redaction_log_json: Some(redaction_log),
        };

        let container = build_rvf_container(&input).expect("build should succeed");
        let seg_count = count_segments(&container);
        assert!(seg_count >= 3, "expected >= 3 segments, got {seg_count}");
        assert_eq!(seg_count, 5);
    }

    #[test]
    fn test_rvf_container_minimal() {
        let embedding = vec![1.0f32; 128];
        let tags = vec![];
        let input = RvfPipelineInput {
            memory_id: "min-id",
            embedding: &embedding,
            title: "Minimal",
            content: "Content",
            tags: &tags,
            category: "solution",
            contributor_id: "anon",
            witness_chain: None,
            dp_proof_json: None,
            redaction_log_json: None,
        };
        let container = build_rvf_container(&input).expect("build should succeed");
        let seg_count = count_segments(&container);
        assert_eq!(seg_count, 2);
    }

    // -- Cloud pipeline tests --

    #[test]
    fn test_pubsub_decode_push() {
        use base64::Engine;
        let data_b64 = base64::engine::general_purpose::STANDARD.encode(b"hello world");
        let envelope = PubSubPushEnvelope {
            message: PubSubPushMessage {
                data: Some(data_b64),
                attributes: HashMap::from([("source".into(), "test".into())]),
                message_id: "msg-001".into(),
                publish_time: None,
            },
            subscription: "projects/test/subscriptions/test-sub".into(),
        };
        let msg = PubSubClient::decode_push(envelope).unwrap();
        assert_eq!(msg.data, b"hello world");
        assert_eq!(msg.message_id, "msg-001");
        assert_eq!(msg.attributes.get("source").unwrap(), "test");
    }

    #[test]
    fn test_data_injector_dedup() {
        let injector = DataInjector::new();
        let item = InjectionItem {
            source: InjectionSource::Webhook,
            title: "Test Title".into(),
            content: "Test Content".into(),
            category: Some("pattern".into()),
            tags: vec!["test".into()],
            metadata: HashMap::new(),
            received_at: Utc::now(),
        };

        let r1 = injector.process(&item);
        assert!(r1.accepted);
        assert!(!r1.duplicate);
        assert_eq!(r1.stage_reached, "ready_for_embed");
        assert_eq!(injector.new_items_count(), 1);

        // Same item again should be deduplicated
        let r2 = injector.process(&item);
        assert!(!r2.accepted);
        assert!(r2.duplicate);
        assert_eq!(r2.stage_reached, "dedup");
        assert_eq!(injector.new_items_count(), 1);
    }

    #[test]
    fn test_data_injector_validation() {
        let injector = DataInjector::new();
        let item = InjectionItem {
            source: InjectionSource::PubSub,
            title: "".into(),
            content: "has content".into(),
            category: None,
            tags: vec![],
            metadata: HashMap::new(),
            received_at: Utc::now(),
        };
        let r = injector.process(&item);
        assert!(!r.accepted);
        assert_eq!(r.stage_reached, "validate");
        assert!(r.error.is_some());
    }

    #[test]
    fn test_content_hash_deterministic() {
        let h1 = DataInjector::content_hash("title", "content");
        let h2 = DataInjector::content_hash("title", "content");
        assert_eq!(h1, h2);

        let h3 = DataInjector::content_hash("title", "different");
        assert_ne!(h1, h3);
    }

    #[tokio::test]
    async fn test_scheduler_due_tasks() {
        let config = SchedulerConfig {
            train_item_threshold: 5,
            train_interval_secs: 0,     // immediately due
            drift_interval_secs: 0,     // immediately due
            transfer_interval_secs: 99999,
            graph_rebalance_secs: 99999,
            cleanup_interval_secs: 99999,
            attractor_interval_secs: 99999,
            prune_quality_threshold: 0.3,
        };
        let scheduler = OptimizationScheduler::new(config);

        // With 0-second intervals, training and drift should be due
        let due = scheduler.due_tasks(0).await;
        assert!(due.contains(&"training".to_string()));
        assert!(due.contains(&"drift_monitoring".to_string()));
        assert!(!due.contains(&"graph_rebalancing".to_string()));

        // Mark training complete, verify cycle count
        scheduler.mark_completed("training").await;
        assert_eq!(scheduler.cycles_completed(), 1);
    }

    #[tokio::test]
    async fn test_metrics_collector() {
        let mc = MetricsCollector::new();
        mc.record_received();
        mc.record_received();
        mc.record_processed();
        mc.record_failed();
        mc.set_queue_depth(42);
        mc.record_injection().await;

        let scheduler = OptimizationScheduler::new(SchedulerConfig::default());
        let snap = mc.snapshot(&scheduler).await;
        assert_eq!(snap.messages_received, 2);
        assert_eq!(snap.messages_processed, 1);
        assert_eq!(snap.messages_failed, 1);
        assert_eq!(snap.queue_depth, 42);
        assert!(snap.injections_per_minute > 0.0);
    }

    #[test]
    fn test_extract_tag() {
        let xml = r#"<title>Hello World</title><link>https://x.com</link>"#;
        assert_eq!(extract_tag(xml, "title"), Some("Hello World".into()));
        assert_eq!(extract_tag(xml, "link"), Some("https://x.com".into()));
        assert_eq!(extract_tag(xml, "missing"), None);
    }

    #[test]
    fn test_feed_entry_to_injection_item() {
        let entry = FeedEntry {
            title: "Article".into(),
            content: "Body text".into(),
            link: Some("https://example.com/1".into()),
            published: None,
            content_hash: "abc123".into(),
            source_url: "https://example.com/feed".into(),
            category: Some("science".into()),
            tags: vec!["ai".into()],
        };
        let item = FeedIngester::to_injection_item(&entry);
        assert_eq!(item.source, InjectionSource::RssFeed);
        assert_eq!(item.title, "Article");
        assert_eq!(item.metadata.get("source_link").unwrap(), "https://example.com/1");
    }

    #[test]
    fn test_feed_parse_rss_xml() {
        let ingester = FeedIngester::new(vec![]);
        let source = FeedSource {
            url: "https://example.com/feed".into(),
            poll_interval_secs: 300,
            default_category: Some("news".into()),
            default_tags: vec!["rss".into()],
        };
        let xml = r#"
        <rss><channel>
          <item>
            <title>First Post</title>
            <description>Content of the first post</description>
            <link>https://example.com/1</link>
          </item>
          <item>
            <title>Second Post</title>
            <description>Content of the second post</description>
          </item>
        </channel></rss>"#;

        let entries = ingester.parse_feed_xml(xml, &source);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].title, "First Post");
        assert_eq!(entries[0].category, Some("news".into()));
        assert_eq!(entries[1].title, "Second Post");
        assert!(entries[0].content_hash != entries[1].content_hash);
    }
}
