//! GitHub Gist publisher for autonomous brain discoveries.
//!
//! Only publishes **truly novel** findings — requires:
//! - Minimum novelty score (new inferences, not just restated knowledge)
//! - Minimum evidence threshold (enough observations to be credible)
//! - Strange loop quality gate (meta-cognitive self-assessment)
//! - Rate limited to 1 gist per hour
//!
//! Each gist includes formal verification links, witness chain hashes,
//! and links back to π.ruv.io for independent verification.

use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

// ── Novelty thresholds ──
// Tuned for current brain state (~2600 memories, 10 categories, 11 inference rules).
// These will publish roughly once per day when data is flowing, less when static.
/// Minimum new inferences: forward-chained claims not in any single memory
const MIN_NEW_INFERENCES: usize = 2;
/// Minimum evidence observations
const MIN_EVIDENCE: usize = 100;
/// Minimum strange loop quality score
const MIN_STRANGE_LOOP_SCORE: f32 = 0.008;
/// Minimum propositions extracted in this cycle
const MIN_PROPOSITIONS: usize = 5;
/// Minimum SONA patterns — 0 means SONA isn't required (it needs trajectory data)
const MIN_SONA_PATTERNS: usize = 0;
/// Minimum Pareto front growth — evolution must have found new solutions
const MIN_PARETO_GROWTH: usize = 1;

/// A discovery worthy of publishing.
///
/// The key distinction: `findings` should contain the actual discovered knowledge
/// (e.g., "category X relates_to category Y with 0.92 confidence"), not process
/// metrics (e.g., "10 propositions extracted"). The gist template formats
/// findings as the intellectual contribution, not a status report.
#[derive(Debug, Clone, Serialize)]
pub struct Discovery {
    pub title: String,
    pub category: String,
    pub abstract_text: String,
    /// The actual discovered knowledge — propositions, inferences, cross-domain connections.
    /// Each entry should be a substantive claim, not a metric.
    pub findings: Vec<String>,
    /// Methodology narrative
    pub methodology: Vec<String>,
    pub evidence_count: usize,
    pub confidence: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// IDs of supporting memories (for verification links)
    pub witness_memory_ids: Vec<String>,
    /// Witness chain hashes from RVF containers
    pub witness_hashes: Vec<String>,
    /// Strange loop self-assessment score
    pub strange_loop_score: f32,
    /// Number of new symbolic inferences (novelty signal)
    pub new_inferences: usize,
    /// Number of propositions extracted
    pub propositions_extracted: usize,
    /// SONA patterns that contributed
    pub sona_patterns: usize,
    /// Pareto front growth (new solutions found)
    pub pareto_growth: usize,
    /// Whether curiosity-driven exploration found this
    pub curiosity_triggered: bool,
    /// Self-reflection narrative from internal voice
    pub self_reflection: String,
    /// The actual propositions discovered (subject, predicate, object, confidence)
    pub propositions: Vec<(String, String, String, f64)>,
    /// The actual inferences derived
    pub inferences: Vec<String>,
}

impl Discovery {
    /// Check if this discovery meets the novelty bar for publishing.
    pub fn is_publishable(&self) -> bool {
        self.new_inferences >= MIN_NEW_INFERENCES
            && self.evidence_count >= MIN_EVIDENCE
            && self.strange_loop_score >= MIN_STRANGE_LOOP_SCORE
            && self.propositions_extracted >= MIN_PROPOSITIONS
            && self.pareto_growth >= MIN_PARETO_GROWTH
            && !self.inferences.is_empty()
    }

    /// Explain why a discovery was or wasn't published.
    pub fn novelty_report(&self) -> String {
        let checks: Vec<(&str, bool, String)> = vec![
            ("inferences", self.new_inferences >= MIN_NEW_INFERENCES,
             format!("{}/{}", self.new_inferences, MIN_NEW_INFERENCES)),
            ("evidence", self.evidence_count >= MIN_EVIDENCE,
             format!("{}/{}", self.evidence_count, MIN_EVIDENCE)),
            ("strange_loop", self.strange_loop_score >= MIN_STRANGE_LOOP_SCORE,
             format!("{:.4}/{:.4}", self.strange_loop_score, MIN_STRANGE_LOOP_SCORE)),
            ("propositions", self.propositions_extracted >= MIN_PROPOSITIONS,
             format!("{}/{}", self.propositions_extracted, MIN_PROPOSITIONS)),
            ("pareto_growth", self.pareto_growth >= MIN_PARETO_GROWTH,
             format!("{}/{}", self.pareto_growth, MIN_PARETO_GROWTH)),
            ("has_inferences", !self.inferences.is_empty(),
             format!("{} items", self.inferences.len())),
        ];

        let failed: Vec<String> = checks.iter()
            .filter(|(_, ok, _)| !ok)
            .map(|(name, _, val)| format!("{} {}", name, val))
            .collect();

        if failed.is_empty() {
            "NOVEL: all thresholds met".to_string()
        } else {
            format!("NOT NOVEL: {}", failed.join(", "))
        }
    }
}

/// Response from GitHub Gist API
#[derive(Debug, Deserialize)]
struct GistResponse {
    html_url: String,
    #[allow(dead_code)]
    id: String,
}

/// Gist publisher with rate limiting and novelty gating
pub struct GistPublisher {
    token: String,
    last_publish: Mutex<Option<Instant>>,
    min_interval: Duration,
    published_count: Mutex<u64>,
    /// Titles of previously published discoveries (dedup within session)
    published_titles: Mutex<Vec<String>>,
}

impl GistPublisher {
    /// Create from env var GITHUB_GIST_PAT; returns None if not set.
    pub fn from_env() -> Option<Self> {
        let token = std::env::var("GITHUB_GIST_PAT").ok()?;
        if token.is_empty() {
            return None;
        }
        Some(Self {
            token,
            last_publish: Mutex::new(None),
            min_interval: Duration::from_secs(14400), // 4 hour minimum between gists
            published_count: Mutex::new(0),
            published_titles: Mutex::new(Vec::new()),
        })
    }

    /// Check if we can publish (rate limit + dedup)
    pub fn can_publish(&self, title: &str) -> bool {
        // Rate limit
        let last = self.last_publish.lock();
        if let Some(t) = *last {
            if t.elapsed() < self.min_interval {
                return false;
            }
        }
        // Dedup: don't publish same title twice
        let titles = self.published_titles.lock();
        !titles.iter().any(|t| t == title)
    }

    pub fn published_count(&self) -> u64 {
        *self.published_count.lock()
    }

    /// Attempt to publish a discovery. Returns:
    /// - Ok(Some(url)) if published
    /// - Ok(None) if not novel enough or rate limited
    /// - Err if API failed
    pub async fn try_publish(&self, discovery: &Discovery) -> Result<Option<String>, String> {
        if !discovery.is_publishable() {
            tracing::debug!(
                "Discovery not publishable: {}",
                discovery.novelty_report()
            );
            return Ok(None);
        }
        if !self.can_publish(&discovery.title) {
            tracing::debug!("Gist publish rate limited or duplicate title");
            return Ok(None);
        }

        let filename = format!(
            "pi-brain-discovery-{}.md",
            discovery.timestamp.format("%Y%m%d-%H%M%S")
        );

        // Use Gemini to rewrite the raw discovery into a polished article
        let raw_content = format_academic_gist(discovery);
        let content = match rewrite_with_gemini(discovery, &raw_content).await {
            Ok(polished) => {
                tracing::info!("Gemini rewrote discovery ({} → {} chars)", raw_content.len(), polished.len());
                polished
            }
            Err(e) => {
                tracing::warn!("Gemini rewrite failed ({}), using raw content", e);
                raw_content
            }
        };

        let body = serde_json::json!({
            "description": format!("π Brain Discovery: {}", discovery.title),
            "public": true,
            "files": {
                filename: {
                    "content": content
                }
            }
        });

        let client = reqwest::Client::new();
        let resp = client
            .post("https://api.github.com/gists")
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "pi-brain/0.1")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("HTTP error: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!(
                "GitHub API {}: {}",
                status,
                &text[..text.len().min(200)]
            ));
        }

        let gist: GistResponse = resp
            .json()
            .await
            .map_err(|e| format!("Parse error: {}", e))?;

        *self.last_publish.lock() = Some(Instant::now());
        *self.published_count.lock() += 1;
        self.published_titles
            .lock()
            .push(discovery.title.clone());

        tracing::info!(
            "Published discovery gist: {} -> {} (novelty: {})",
            discovery.title,
            gist.html_url,
            discovery.novelty_report()
        );

        Ok(Some(gist.html_url))
    }
}

/// Format a discovery as an academic-style markdown document with verification.
/// Format a discovery as an academic-style document focused on the actual
/// discovered knowledge, not pipeline metrics.
fn format_academic_gist(d: &Discovery) -> String {
    // Format propositions as a knowledge table
    let propositions_md = if d.propositions.is_empty() {
        String::new()
    } else {
        let rows: Vec<String> = d.propositions.iter().map(|(s, p, o, c)| {
            format!("| {} | {} | {} | {:.2} |", s, p, o, c)
        }).collect();
        format!(
            "| Subject | Relation | Object | Confidence |\n\
             |---------|----------|--------|------------|\n\
             {}\n",
            rows.join("\n")
        )
    };

    // Format inferences as numbered claims
    let inferences_md = d.inferences.iter().enumerate()
        .map(|(i, inf)| format!("{}. {}", i + 1, inf))
        .collect::<Vec<_>>()
        .join("\n");

    // Format findings (the high-level insights)
    let findings_md = d.findings.iter().enumerate()
        .map(|(i, f)| format!("{}. {}", i + 1, f))
        .collect::<Vec<_>>()
        .join("\n");

    // Witness links
    let witness_md = d.witness_memory_ids.iter().take(5)
        .zip(d.witness_hashes.iter().take(5).chain(std::iter::repeat(&String::new())))
        .map(|(id, hash)| {
            let short = &id[..id.len().min(8)];
            if hash.is_empty() {
                format!("| [`{}`](https://pi.ruv.io/v1/memories/{}) | — |", short, id)
            } else {
                format!("| [`{}`](https://pi.ruv.io/v1/memories/{}) | `{}` |", short, id, &hash[..hash.len().min(16)])
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"# {title}

> **Domain:** {category} · **Generated:** {timestamp} by [π Brain](https://pi.ruv.io)
> Autonomous discovery from {evidence} knowledge observations

---

## Abstract

{abstract_text}

## Discovered Knowledge

### Novel Inferences

The following claims were derived by forward-chaining symbolic reasoning
over extracted propositions. These are **new knowledge** not present in
any single input observation — they emerge from combining evidence across
multiple sources:

{inferences}

### Extracted Propositions

Symbolic knowledge extracted from {evidence} observations across {n_clusters}
domain clusters:

{propositions}

## Cross-Domain Insights

{findings}

## Internal Deliberation

The brain's Internal Voice reflected on this learning cycle:

> {self_reflection}

## Verification

### Witness Chain

| Memory | Hash |
|--------|------|
{witnesses}

### Reproduce

```bash
curl -H "Authorization: Bearer KEY" "https://pi.ruv.io/v1/propositions"
curl -H "Authorization: Bearer KEY" "https://pi.ruv.io/v1/memories/search?q={category}&limit=10"
curl -H "Authorization: Bearer KEY" "https://pi.ruv.io/v1/cognitive/status"
```

---

*Autonomously generated by [π Brain](https://pi.ruv.io). No human authored this content.
{evidence} observations · {n_inferences} inferences · {n_props} propositions · strange loop {sl:.4}*
"#,
        title = d.title,
        category = d.category,
        timestamp = d.timestamp.format("%Y-%m-%d %H:%M UTC"),
        evidence = d.evidence_count,
        abstract_text = d.abstract_text,
        inferences = if inferences_md.is_empty() { "No novel inferences this cycle.".to_string() } else { inferences_md },
        n_clusters = d.propositions.len().max(1),
        propositions = if propositions_md.is_empty() { "No propositions extracted.".to_string() } else { propositions_md },
        findings = if findings_md.is_empty() { "No cross-domain insights this cycle.".to_string() } else { findings_md },
        self_reflection = d.self_reflection,
        witnesses = if witness_md.is_empty() { "| — | — |".to_string() } else { witness_md },
        n_inferences = d.new_inferences,
        n_props = d.propositions_extracted,
        sl = d.strange_loop_score,
    )
}

/// Use Gemini to rewrite a raw discovery into a polished, human-readable article.
/// Falls back to raw content if Gemini is unavailable.
async fn rewrite_with_gemini(discovery: &Discovery, raw_content: &str) -> Result<String, String> {
    let api_key = std::env::var("GEMINI_API_KEY")
        .map_err(|_| "GEMINI_API_KEY not set".to_string())?;
    let model = std::env::var("GEMINI_MODEL")
        .unwrap_or_else(|_| "gemini-2.5-flash".to_string());

    // Build a concise summary of what was discovered for the prompt
    let inferences_summary = discovery.inferences.iter()
        .take(5)
        .map(|i| format!("- {}", i))
        .collect::<Vec<_>>()
        .join("\n");

    let propositions_summary = discovery.propositions.iter()
        .take(10)
        .map(|(s, p, o, c)| format!("- {} {} {} (confidence: {:.2})", s, p, o, c))
        .collect::<Vec<_>>()
        .join("\n");

    let findings_summary = discovery.findings.iter()
        .take(5)
        .map(|f| format!("- {}", f))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
r#"You are the editorial voice of the π Brain — an autonomous AI knowledge system at pi.ruv.io.

Rewrite the following raw discovery data into a polished academic-style GitHub Gist article. The article must be:

1. **Accessible**: Start with a plain-language introduction that anyone can understand — what was discovered and why it matters
2. **Technical**: Include the formal symbolic reasoning chain, propositions, and inference rules
3. **Verifiable**: Include the witness chain hashes and API links for independent verification
4. **Honest**: If the confidence is low or the finding is speculative, say so clearly

Structure:
- Title (compelling, specific — not generic)
- Plain-language summary (2-3 sentences, no jargon)
- Key discoveries (what was actually found, in human terms)
- Technical details (propositions, inference chains, confidence scores)
- Verification (witness hashes, API endpoints)
- Citation block

Raw data:

**Inferences derived:**
{inferences}

**Propositions extracted:**
{propositions}

**Cross-domain findings:**
{findings}

**Self-reflection:**
{reflection}

**Stats:** {evidence} observations, {n_inferences} inferences, {n_props} propositions, strange loop score {sl:.4}, {sona} SONA patterns

**Witness hashes:** {witnesses}

**Witness memory IDs:** {memory_ids}

CRITICAL rules for honest scientific communication:
- Use the ACTUAL content from the findings and inferences above — don't invent facts
- NEVER use the word "causes" or "causal" unless confidence >= 80% AND temporal evidence exists
- For confidence < 50%: use "shows weak co-occurrence with", "may be loosely associated with"
- For confidence 50-65%: use "is associated with", "co-occurs with"
- For confidence 65-80%: use "may influence", "appears to be linked to"
- For confidence >= 80%: use "strongly associated with", "likely influences"
- Frame findings as HYPOTHESES, not conclusions. Use "suggests", "indicates", "appears"
- Be explicit about limitations: low vote coverage, small evidence sets, no temporal validation
- The article is from the π Brain's perspective ("we identified", "our analysis suggests")
- Include a "Limitations" section that honestly states what this does NOT prove
- Include links to https://pi.ruv.io for verification
- End with a proper BibTeX citation block
- Keep it under 2000 words
- Output ONLY the markdown article, no preamble

Write the article now:"#,
        inferences = inferences_summary,
        propositions = propositions_summary,
        findings = findings_summary,
        reflection = discovery.self_reflection,
        evidence = discovery.evidence_count,
        n_inferences = discovery.new_inferences,
        n_props = discovery.propositions_extracted,
        sl = discovery.strange_loop_score,
        sona = discovery.sona_patterns,
        witnesses = discovery.witness_hashes.iter().take(5)
            .map(|h| format!("`{}`", h))
            .collect::<Vec<_>>()
            .join(", "),
        memory_ids = discovery.witness_memory_ids.iter().take(5)
            .map(|id| format!("`{}`", &id[..id.len().min(8)]))
            .collect::<Vec<_>>()
            .join(", "),
    );

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, api_key
    );

    let grounding = std::env::var("GEMINI_GROUNDING")
        .unwrap_or_else(|_| "true".to_string()) == "true";

    let mut body = serde_json::json!({
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "maxOutputTokens": 8192,
            "temperature": 0.3
        }
    });

    if grounding {
        body["tools"] = serde_json::json!([{"google_search": {}}]);
    }

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Gemini HTTP error: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Gemini API {}: {}", status, &text[..text.len().min(200)]));
    }

    let json: serde_json::Value = resp.json().await
        .map_err(|e| format!("Gemini parse error: {}", e))?;

    // Extract text from Gemini response
    let text = json
        .get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.get(0))
        .and_then(|p| p.get("text"))
        .and_then(|t| t.as_str())
        .ok_or("No text in Gemini response".to_string())?;

    // Append verification footer that Gemini might omit
    let footer = format!(
        "\n\n---\n\n\
         *This article was autonomously generated by the [π Brain](https://pi.ruv.io) \
         cognitive system and editorially refined by Gemini. The underlying data, \
         propositions, and inference chains are machine-derived from {} observations. \
         No human authored or curated the findings.*\n\n\
         **Live Dashboard:** [π.ruv.io](https://pi.ruv.io) · \
         **API:** [/v1/status](https://pi.ruv.io/v1/status) · \
         **Verify:** [/v1/propositions](https://pi.ruv.io/v1/propositions)\n",
        discovery.evidence_count
    );

    Ok(format!("{}{}", text.trim(), footer))
}
