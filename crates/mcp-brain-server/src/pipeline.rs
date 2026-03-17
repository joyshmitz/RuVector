//! Cloud-native data pipeline for real-time injection and optimization.
//!
//! Also contains the RVF container construction pipeline (ADR-075 Phase 5).

use chrono::{DateTime, Utc};
use rvf_types::SegmentFlags;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;

// ── RVF Container Construction (ADR-075 Phase 5) ────────────────────

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

/// Build an RVF container. Returns serialized bytes (64-byte-aligned segments).
pub fn build_rvf_container(input: &RvfPipelineInput<'_>) -> Result<Vec<u8>, String> {
    let flags = SegmentFlags::empty();
    let mut out = Vec::new();
    let mut sid: u64 = 1;
    // VEC (0x01)
    let mut vec_payload = Vec::with_capacity(input.embedding.len() * 4);
    for &v in input.embedding { vec_payload.extend_from_slice(&v.to_le_bytes()); }
    out.extend_from_slice(&rvf_wire::write_segment(0x01, &vec_payload, flags, sid)); sid += 1;
    // META (0x07)
    let meta = serde_json::json!({
        "memory_id": input.memory_id, "title": input.title, "content": input.content,
        "tags": input.tags, "category": input.category, "contributor_id": input.contributor_id,
    });
    let mp = serde_json::to_vec(&meta).map_err(|e| format!("Failed to serialize RVF metadata: {e}"))?;
    out.extend_from_slice(&rvf_wire::write_segment(0x07, &mp, flags, sid)); sid += 1;
    // WITNESS (0x0A)
    if let Some(c) = input.witness_chain {
        out.extend_from_slice(&rvf_wire::write_segment(0x0A, c, flags, sid)); sid += 1;
    }
    // DiffPrivacyProof (0x34)
    if let Some(p) = input.dp_proof_json {
        out.extend_from_slice(&rvf_wire::write_segment(0x34, p.as_bytes(), flags, sid)); sid += 1;
    }
    // RedactionLog (0x35)
    if let Some(l) = input.redaction_log_json {
        out.extend_from_slice(&rvf_wire::write_segment(0x35, l.as_bytes(), flags, sid));
        let _ = sid;
    }
    Ok(out)
}

/// Count segments in a serialized RVF container.
pub fn count_segments(container: &[u8]) -> usize {
    let (mut count, mut off) = (0, 0);
    while off + 64 <= container.len() {
        let plen = u64::from_le_bytes(container[off+16..off+24].try_into().unwrap_or([0u8;8])) as usize;
        off += rvf_wire::calculate_padded_size(64, plen);
        count += 1;
    }
    count
}

// ── Cloud Pub/Sub ────────────────────────────────────────────────────

/// A decoded Pub/Sub message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PubSubMessage {
    pub data: Vec<u8>,
    pub attributes: HashMap<String, String>,
    pub message_id: String,
    pub publish_time: Option<DateTime<Utc>>,
}

/// Push envelope from Cloud Pub/Sub (HTTP POST body).
#[derive(Debug, Deserialize)]
pub struct PubSubPushEnvelope {
    pub message: PubSubPushMsg,
    pub subscription: String,
}

#[derive(Debug, Deserialize)]
pub struct PubSubPushMsg {
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
        Self {
            use_metadata_server: std::env::var("PUBSUB_EMULATOR_HOST").is_err(),
            project_id, subscription_id,
            http: reqwest::Client::builder().timeout(std::time::Duration::from_secs(30))
                .build().unwrap_or_default(),
        }
    }

    /// Decode a push-envelope into a `PubSubMessage`.
    pub fn decode_push(envelope: PubSubPushEnvelope) -> Result<PubSubMessage, String> {
        use base64::Engine;
        let data = match envelope.message.data {
            Some(b64) => base64::engine::general_purpose::STANDARD.decode(&b64)
                .map_err(|e| format!("base64 decode failed: {e}"))?,
            None => Vec::new(),
        };
        Ok(PubSubMessage {
            data, attributes: envelope.message.attributes,
            message_id: envelope.message.message_id, publish_time: envelope.message.publish_time,
        })
    }

    /// Acknowledge messages by ack_id (pull mode).
    pub async fn acknowledge(&self, ack_ids: &[String]) -> Result<(), String> {
        if ack_ids.is_empty() { return Ok(()); }
        let url = format!(
            "https://pubsub.googleapis.com/v1/projects/{}/subscriptions/{}:acknowledge",
            self.project_id, self.subscription_id
        );
        let mut req = self.http.post(&url).json(&serde_json::json!({ "ackIds": ack_ids }));
        if self.use_metadata_server {
            if let Some(t) = get_metadata_token(&self.http).await { req = req.bearer_auth(t); }
        }
        let resp = req.send().await.map_err(|e| format!("ack failed: {e}"))?;
        if !resp.status().is_success() { return Err(format!("ack returned {}", resp.status())); }
        Ok(())
    }
}

/// Fetch access token from GCE metadata server.
async fn get_metadata_token(http: &reqwest::Client) -> Option<String> {
    #[derive(Deserialize)]
    struct T { access_token: String }
    let r = http.get("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token")
        .header("Metadata-Flavor", "Google").send().await.ok()?;
    if !r.status().is_success() { return None; }
    Some(r.json::<T>().await.ok()?.access_token)
}

// ── Data Injection Pipeline ──────────────────────────────────────────

/// Source of injected data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum InjectionSource { PubSub, BatchUpload, RssFeed, Webhook }

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

/// Processes incoming data: validate -> embed -> dedup -> store -> graph-update -> train-check
#[derive(Debug)]
pub struct DataInjector {
    seen_hashes: dashmap::DashMap<String, DateTime<Utc>>,
    new_items_since_train: AtomicU64,
}

impl DataInjector {
    pub fn new() -> Self {
        Self { seen_hashes: dashmap::DashMap::new(), new_items_since_train: AtomicU64::new(0) }
    }

    /// Compute a SHA-256 content hash for deduplication.
    pub fn content_hash(title: &str, content: &str) -> String {
        let mut h = Sha256::new();
        h.update(title.as_bytes()); h.update(b"|"); h.update(content.as_bytes());
        hex::encode(h.finalize())
    }

    /// Run the injection pipeline for a single item.
    pub fn process(&self, item: &InjectionItem) -> InjectionResult {
        if item.title.is_empty() || item.content.is_empty() {
            return InjectionResult { item_hash: String::new(), accepted: false, duplicate: false,
                stage_reached: "validate".into(), error: Some("title and content must be non-empty".into()) };
        }
        let hash = Self::content_hash(&item.title, &item.content);
        if self.seen_hashes.contains_key(&hash) {
            return InjectionResult { item_hash: hash, accepted: false, duplicate: true,
                stage_reached: "dedup".into(), error: None };
        }
        self.seen_hashes.insert(hash.clone(), Utc::now());
        self.new_items_since_train.fetch_add(1, Ordering::Relaxed);
        InjectionResult { item_hash: hash, accepted: true, duplicate: false,
            stage_reached: "ready_for_embed".into(), error: None }
    }

    pub fn new_items_count(&self) -> u64 { self.new_items_since_train.load(Ordering::Relaxed) }
    pub fn reset_train_counter(&self) { self.new_items_since_train.store(0, Ordering::Relaxed); }
    pub fn dedup_set_size(&self) -> usize { self.seen_hashes.len() }
}

impl Default for DataInjector { fn default() -> Self { Self::new() } }

// ── Optimization Scheduler ───────────────────────────────────────────

/// Configuration for optimization cycle intervals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub train_item_threshold: u64,
    pub train_interval_secs: u64,
    pub drift_interval_secs: u64,
    pub transfer_interval_secs: u64,
    pub graph_rebalance_secs: u64,
    pub cleanup_interval_secs: u64,
    pub attractor_interval_secs: u64,
    pub prune_quality_threshold: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            train_item_threshold: 100, train_interval_secs: 300, drift_interval_secs: 900,
            transfer_interval_secs: 1800, graph_rebalance_secs: 3600,
            cleanup_interval_secs: 86400, attractor_interval_secs: 1200,
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
            config, cycles_completed: AtomicU64::new(0),
            last_train: RwLock::new(now), last_drift_check: RwLock::new(now),
            last_transfer: RwLock::new(now), last_graph_rebalance: RwLock::new(now),
            last_cleanup: RwLock::new(now), last_attractor: RwLock::new(now),
        }
    }

    /// Check which optimization tasks are due.
    pub async fn due_tasks(&self, new_item_count: u64) -> Vec<String> {
        let now = Utc::now();
        let ss = |ts: &DateTime<Utc>| (now - *ts).num_seconds().max(0) as u64;
        let mut due = Vec::new();
        if new_item_count >= self.config.train_item_threshold
            || ss(&*self.last_train.read().await) >= self.config.train_interval_secs
        { due.push("training".into()); }
        if ss(&*self.last_drift_check.read().await) >= self.config.drift_interval_secs { due.push("drift_monitoring".into()); }
        if ss(&*self.last_transfer.read().await) >= self.config.transfer_interval_secs { due.push("cross_domain_transfer".into()); }
        if ss(&*self.last_graph_rebalance.read().await) >= self.config.graph_rebalance_secs { due.push("graph_rebalancing".into()); }
        if ss(&*self.last_cleanup.read().await) >= self.config.cleanup_interval_secs { due.push("memory_cleanup".into()); }
        if ss(&*self.last_attractor.read().await) >= self.config.attractor_interval_secs { due.push("attractor_analysis".into()); }
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

    pub fn cycles_completed(&self) -> u64 { self.cycles_completed.load(Ordering::Relaxed) }
}

// ── Health & Metrics ─────────────────────────────────────────────────

/// Pipeline metrics snapshot for Cloud Monitoring.
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
    recent_injections: RwLock<Vec<(i64, u64)>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self { received: AtomicU64::new(0), processed: AtomicU64::new(0),
            failed: AtomicU64::new(0), queue_depth: AtomicU64::new(0),
            recent_injections: RwLock::new(Vec::new()) }
    }
    pub fn record_received(&self) { self.received.fetch_add(1, Ordering::Relaxed); }
    pub fn record_processed(&self) { self.processed.fetch_add(1, Ordering::Relaxed); }
    pub fn record_failed(&self) { self.failed.fetch_add(1, Ordering::Relaxed); }
    pub fn set_queue_depth(&self, d: u64) { self.queue_depth.store(d, Ordering::Relaxed); }

    pub async fn record_injection(&self) {
        let now = Utc::now().timestamp();
        let mut w = self.recent_injections.write().await;
        w.push((now, 1));
        w.retain(|(ts, _)| *ts >= now - 300);
    }

    pub async fn injections_per_minute(&self) -> f64 {
        let w = self.recent_injections.read().await;
        if w.is_empty() { return 0.0; }
        let total: u64 = w.iter().map(|(_, c)| c).sum();
        let span = ((Utc::now().timestamp() - w[0].0) as f64 / 60.0).max(1.0 / 60.0);
        total as f64 / span
    }

    pub async fn snapshot(&self, sched: &OptimizationScheduler) -> PipelineMetrics {
        PipelineMetrics {
            messages_received: self.received.load(Ordering::Relaxed),
            messages_processed: self.processed.load(Ordering::Relaxed),
            messages_failed: self.failed.load(Ordering::Relaxed),
            injections_per_minute: self.injections_per_minute().await,
            last_training_time: Some(*sched.last_train.read().await),
            last_drift_check: Some(*sched.last_drift_check.read().await),
            last_transfer: Some(*sched.last_transfer.read().await),
            queue_depth: self.queue_depth.load(Ordering::Relaxed),
            optimization_cycles_completed: sched.cycles_completed(),
        }
    }
}

impl Default for MetricsCollector { fn default() -> Self { Self::new() } }

// ── Feed Ingestion (RSS/Atom) ────────────────────────────────────────

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
    last_poll: HashMap<String, DateTime<Utc>>,
    seen_hashes: dashmap::DashMap<String, ()>,
    http: reqwest::Client,
}

impl FeedIngester {
    pub fn new(sources: Vec<FeedSource>) -> Self {
        let lp = sources.iter().map(|s| (s.url.clone(), Utc::now())).collect();
        Self { sources, last_poll: lp, seen_hashes: dashmap::DashMap::new(),
            http: reqwest::Client::builder().timeout(std::time::Duration::from_secs(30))
                .build().unwrap_or_default() }
    }

    pub fn feeds_due(&self) -> Vec<&FeedSource> {
        let now = Utc::now();
        self.sources.iter().filter(|s| {
            let last = self.last_poll.get(&s.url).copied().unwrap_or(now);
            (now - last).num_seconds().max(0) as u64 >= s.poll_interval_secs
        }).collect()
    }

    /// Fetch and parse a feed URL, returning new (non-duplicate) entries.
    pub async fn fetch_feed(&self, source: &FeedSource) -> Result<Vec<FeedEntry>, String> {
        let resp = self.http.get(&source.url)
            .header("Accept", "application/rss+xml, application/atom+xml, text/xml")
            .send().await.map_err(|e| format!("feed fetch failed for {}: {e}", source.url))?;
        if !resp.status().is_success() { return Err(format!("feed {} returned {}", source.url, resp.status())); }
        let body = resp.text().await.map_err(|e| format!("feed body read failed: {e}"))?;
        Ok(self.parse_feed_xml(&body, source).into_iter()
            .filter(|e| { if self.seen_hashes.contains_key(&e.content_hash) { false }
                else { self.seen_hashes.insert(e.content_hash.clone(), ()); true } }).collect())
    }

    fn parse_feed_xml(&self, xml: &str, source: &FeedSource) -> Vec<FeedEntry> {
        let blocks: Vec<&str> = if xml.contains("<item>") || xml.contains("<item ") {
            xml.split("<item").skip(1).filter_map(|s| s.split("</item>").next()).collect()
        } else {
            xml.split("<entry").skip(1).filter_map(|s| s.split("</entry>").next()).collect()
        };
        blocks.iter().filter_map(|block| {
            let title = extract_tag(block, "title").unwrap_or_default();
            let content = extract_tag(block, "description")
                .or_else(|| extract_tag(block, "content"))
                .or_else(|| extract_tag(block, "summary")).unwrap_or_default();
            if title.is_empty() && content.is_empty() { return None; }
            let hash = DataInjector::content_hash(&title, &content);
            Some(FeedEntry { title, content, link: extract_tag(block, "link"), published: None,
                content_hash: hash, source_url: source.url.clone(),
                category: source.default_category.clone(), tags: source.default_tags.clone() })
        }).collect()
    }

    /// Convert a `FeedEntry` into an `InjectionItem`.
    pub fn to_injection_item(entry: &FeedEntry) -> InjectionItem {
        let mut meta = HashMap::new();
        if let Some(ref l) = entry.link { meta.insert("source_link".into(), l.clone()); }
        meta.insert("source_url".into(), entry.source_url.clone());
        meta.insert("content_hash".into(), entry.content_hash.clone());
        InjectionItem { source: InjectionSource::RssFeed, title: entry.title.clone(),
            content: entry.content.clone(), category: entry.category.clone(),
            tags: entry.tags.clone(), metadata: meta,
            received_at: entry.published.unwrap_or_else(Utc::now) }
    }

    pub fn seen_count(&self) -> usize { self.seen_hashes.len() }
}

fn extract_tag(xml: &str, tag: &str) -> Option<String> {
    let start = xml.find(&format!("<{}", tag))?;
    let after = &xml[start..];
    let cs = after.find('>')? + 1;
    let inner = &after[cs..];
    let end = inner.find(&format!("</{}>", tag))?;
    let text = inner[..end].trim();
    if text.is_empty() { None } else { Some(text.to_string()) }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rvf_container_has_segments() {
        let embedding = vec![0.1f32, 0.2, 0.3, 0.4];
        let tags = vec!["test".to_string()];
        let wc = rvf_crypto::create_witness_chain(&[rvf_crypto::WitnessEntry {
            prev_hash: [0u8; 32], action_hash: rvf_crypto::shake256_256(b"test"),
            timestamp_ns: 1000, witness_type: 0x01,
        }]);
        let input = RvfPipelineInput {
            memory_id: "test-id", embedding: &embedding, title: "Test Title",
            content: "Test content", tags: &tags, category: "pattern",
            contributor_id: "test-contributor", witness_chain: Some(&wc),
            dp_proof_json: Some(r#"{"epsilon":1.0,"delta":1e-5}"#),
            redaction_log_json: Some(r#"{"entries":[],"total_redactions":0}"#),
        };
        let container = build_rvf_container(&input).unwrap();
        assert_eq!(count_segments(&container), 5);
    }

    #[test]
    fn test_rvf_container_minimal() {
        let embedding = vec![1.0f32; 128];
        let input = RvfPipelineInput {
            memory_id: "min-id", embedding: &embedding, title: "Minimal",
            content: "Content", tags: &[], category: "solution", contributor_id: "anon",
            witness_chain: None, dp_proof_json: None, redaction_log_json: None,
        };
        assert_eq!(count_segments(&build_rvf_container(&input).unwrap()), 2);
    }

    #[test]
    fn test_pubsub_decode_push() {
        use base64::Engine;
        let envelope = PubSubPushEnvelope {
            message: PubSubPushMsg {
                data: Some(base64::engine::general_purpose::STANDARD.encode(b"hello world")),
                attributes: HashMap::from([("source".into(), "test".into())]),
                message_id: "msg-001".into(), publish_time: None,
            },
            subscription: "projects/test/subscriptions/test-sub".into(),
        };
        let msg = PubSubClient::decode_push(envelope).unwrap();
        assert_eq!(msg.data, b"hello world");
        assert_eq!(msg.message_id, "msg-001");
    }

    #[test]
    fn test_data_injector_dedup() {
        let inj = DataInjector::new();
        let item = InjectionItem { source: InjectionSource::Webhook, title: "T".into(),
            content: "C".into(), category: Some("p".into()), tags: vec![],
            metadata: HashMap::new(), received_at: Utc::now() };
        let r1 = inj.process(&item);
        assert!(r1.accepted && !r1.duplicate && r1.stage_reached == "ready_for_embed");
        let r2 = inj.process(&item);
        assert!(!r2.accepted && r2.duplicate && r2.stage_reached == "dedup");
        assert_eq!(inj.new_items_count(), 1);
    }

    #[test]
    fn test_data_injector_validation() {
        let inj = DataInjector::new();
        let item = InjectionItem { source: InjectionSource::PubSub, title: "".into(),
            content: "c".into(), category: None, tags: vec![], metadata: HashMap::new(),
            received_at: Utc::now() };
        let r = inj.process(&item);
        assert!(!r.accepted && r.stage_reached == "validate" && r.error.is_some());
    }

    #[test]
    fn test_content_hash_deterministic() {
        assert_eq!(DataInjector::content_hash("a", "b"), DataInjector::content_hash("a", "b"));
        assert_ne!(DataInjector::content_hash("a", "b"), DataInjector::content_hash("a", "c"));
    }

    #[tokio::test]
    async fn test_scheduler_due_tasks() {
        let sched = OptimizationScheduler::new(SchedulerConfig {
            train_item_threshold: 5, train_interval_secs: 0, drift_interval_secs: 0,
            transfer_interval_secs: 99999, graph_rebalance_secs: 99999,
            cleanup_interval_secs: 99999, attractor_interval_secs: 99999,
            prune_quality_threshold: 0.3,
        });
        let due = sched.due_tasks(0).await;
        assert!(due.contains(&"training".to_string()) && due.contains(&"drift_monitoring".to_string()));
        assert!(!due.contains(&"graph_rebalancing".to_string()));
        sched.mark_completed("training").await;
        assert_eq!(sched.cycles_completed(), 1);
    }

    #[tokio::test]
    async fn test_metrics_collector() {
        let mc = MetricsCollector::new();
        mc.record_received(); mc.record_received(); mc.record_processed();
        mc.record_failed(); mc.set_queue_depth(42); mc.record_injection().await;
        let snap = mc.snapshot(&OptimizationScheduler::new(SchedulerConfig::default())).await;
        assert_eq!(snap.messages_received, 2);
        assert_eq!(snap.messages_processed, 1);
        assert_eq!(snap.messages_failed, 1);
        assert_eq!(snap.queue_depth, 42);
        assert!(snap.injections_per_minute > 0.0);
    }

    #[test]
    fn test_extract_tag() {
        assert_eq!(extract_tag("<title>Hello</title>", "title"), Some("Hello".into()));
        assert_eq!(extract_tag("<x>y</x>", "z"), None);
    }

    #[test]
    fn test_feed_entry_to_injection_item() {
        let e = FeedEntry { title: "A".into(), content: "B".into(),
            link: Some("https://x.com/1".into()), published: None, content_hash: "h".into(),
            source_url: "https://x.com/f".into(), category: Some("s".into()), tags: vec![] };
        let item = FeedIngester::to_injection_item(&e);
        assert_eq!(item.source, InjectionSource::RssFeed);
        assert_eq!(item.metadata.get("source_link").unwrap(), "https://x.com/1");
    }

    #[test]
    fn test_feed_parse_rss_xml() {
        let ing = FeedIngester::new(vec![]);
        let src = FeedSource { url: "https://x.com/f".into(), poll_interval_secs: 300,
            default_category: Some("news".into()), default_tags: vec!["rss".into()] };
        let xml = "<rss><channel><item><title>A</title><description>B</description></item>\
            <item><title>C</title><description>D</description></item></channel></rss>";
        let entries = ing.parse_feed_xml(xml, &src);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].title, "A");
        assert_eq!(entries[0].category, Some("news".into()));
        assert_ne!(entries[0].content_hash, entries[1].content_hash);
    }
}
