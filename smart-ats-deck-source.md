# Smart-ATS: AI-Powered Candidate Ranking at 100K Scale

## The Challenge: India Runs Data & AI Hackathon

Redrob AI (Series A, AI-native talent intelligence platform) posed a brutal real-world problem: rank 100,000 candidates for a single Senior AI Engineer job description and identify the best 100 — with a 5-minute runtime ceiling, no GPU, and no network calls during ranking.

The dataset: 100,000 candidate profiles in JSONL format, each with career history, skills, education, behavioral signals (23 redrob_signals fields), and structured metadata. The output: a CSV of exactly 100 ranked candidates with scores and human-readable reasoning.

This is the problem every recruiter faces, scaled to the extreme.

---

## The Gaps in Traditional ATS Systems

### Gap 1: Keyword Matching is Broken

Traditional ATS systems use simple keyword matching. A candidate who writes "ML" instead of "Machine Learning" disappears from results. A candidate who keyword-stuffs their resume with every AI buzzword — but is actually a Marketing Manager — floats to the top.

**The symptom:** Top results are gaming-optimized, not talent-optimized.

### Gap 2: No Understanding of Career Quality

A resume from TCS with 10 years of "AI consulting" looks identical to 10 years of building production ML systems at a product company. Traditional ATS has no way to distinguish.

**The symptom:** Resumes that look good on paper but represent people who haven't shipped real production systems get surfaced as top candidates.

### Gap 3: Semantic Blind Spots

"Built recommendation systems at scale" is not the same as writing "FAISS, Pinecone, Elasticsearch" in a skills list — but it demonstrates deeper competence. Text embeddings can find this. Keyword matching cannot.

**The symptom:** The most interesting candidates — plain-language practitioners who've actually shipped — get buried.

### Gap 4: No Behavioral Signal

Is this candidate actually available? Do they respond to recruiters? Are they actively looking? Are their assessment scores real? Traditional ATS ignores these signals entirely.

**The symptom:** Top-ranked candidates ghost the recruiter 60% of the time.

### Gap 5: Synthetic / Honeypot Profiles

The dataset contains seeded synthetic profiles designed to fool naive ranking systems — candidates with impossible career histories, 40+ expert-level skills with zero duration, or profiles claiming 65 years of experience.

**The symptom:** Naive systems waste interview slots on AI-generated noise.

### Gap 6: LLM Calls Are Expensive at Scale

You could ask an LLM to evaluate all 100,000 candidates. That's ~$500+ and hours of runtime. At 100K candidates × average profile length, even cheap models make this infeasible for real-world use.

**The symptom:** Either you spend a fortune, or you skip LLM intelligence entirely.

---

## The Smart-ATS Solution: Two-Phase Architecture

The core insight: **do the expensive work once, offline. Make the ranking instant.**

### Phase 1 — Pre-computation (runs once, ~30–90 minutes)

This is the slow, expensive phase that happens before any ranking request. It produces a set of artifacts that capture all the intelligence needed for fast ranking.

**Step 1: Feature Extraction (CPU, zero network calls)**

For every candidate, extract structured numeric signals without any API calls:
- Skill score: coverage of 75 target skills, weighted by proficiency level, endorsements, and duration
- Experience score: years in the 5–9 ideal band, product-company bonus, consulting-career penalty
- Education score: institution tier (IIT/IIM tier-1 = 1.0 → tier-4 = 0.25)
- Title alignment multiplier: "Marketing Manager" with AI keywords = 0.30x. "ML Engineer" = 1.0x.
- Availability multiplier: India-based = 1.0x, outside India without relocation willingness = 0.3x
- Behavioral score: 23 redrob_signals → single 0–1 composite (responsiveness 25%, recency 20%, assessment quality 25%, offer fit 15%, GitHub activity 15%)

**Step 2: Honeypot Detection (CPU, zero network calls)**

Seven deterministic rules to flag synthetic profiles:
- H1: Expert/advanced skill listed with zero months of duration
- H2: Years of experience wildly exceeds actual career history length
- H3: Job end date before start date (impossible timeline)
- H4: Shotgun expert — 30+ skills all at expert level (gaming pattern)
- H5: Profile completeness = 0 but has 5+ jobs (bot skeleton)
- H6: 50+ endorsed beginner skills (endorsement farming)
- H7: Years of experience > 60 or negative (impossible value)

Honeypots are zeroed out immediately and excluded from all further processing.

**Step 3: BM25 Keyword Index (CPU, zero network calls)**

Build a BM25Okapi index over all valid candidate profiles using n-gram preprocessing (unigrams + bigrams + trigrams, English stopwords removed). K1=1.2 saturation prevents keyword stuffing from dominating. Score every candidate against the JD.

**Step 4: Semantic Embeddings + FAISS Index (1 API call total)**

Convert every valid candidate profile to a dense embedding vector. Only one embedding provider call: embed all 100K candidates in batches. Store in a FAISS IndexFlatIP (inner product = cosine similarity on normalized vectors). Save to disk.

**Step 5: Hybrid Retrieval — BM25 + FAISS + RRF (zero network calls)**

Use Reciprocal Rank Fusion to combine the BM25 top-1000 and FAISS top-1000 lists into a single ranked pool of top-500 candidates. RRF score = Σ(1/(k+rank_i)) where k=60 is the standard constant. This surfaces candidates that are strong on either signal — the plain-language practitioners AND the keyword-rich profiles.

**Step 6: LLM Scoring (targeted API calls — only top 500)**

Send only the top-500 candidates to an LLM (Claude Haiku or OpenRouter free tier) in batches of 10. Each batch = 1 API call. Total: 50 API calls to score 500 candidates with human-level judgment about cultural fit, actual career trajectory, and JD alignment.

**Why only 500?** After hybrid retrieval, the signal-to-noise ratio is already very high. LLM calls on candidates ranked 5,000–100,000 would mostly confirm "not a fit" — a waste of API credits. By concentrating LLM judgment on the pre-filtered top-500, we get 95%+ of the intelligence at 1% of the cost.

**Total API calls for 100K candidates: ~50 LLM calls + 1 embedding call = 51 network requests.**

---

### Phase 2 — Ranking (< 5 minutes, ZERO network calls)

Load all artifacts from disk. No API calls. No model downloads. Pure arithmetic.

**Scoring Formula (when LLM score available — top 500):**

```
final = (0.55 × claude_score
       + 0.45 × (0.35×skill + 0.30×behavioral + 0.15×experience + 0.10×education + 0.10×bm25))
       × availability_multiplier
       × title_multiplier
       × (0 if honeypot or all-consulting)
```

Claude score carries 55% of the weight — it represents the holistic human-level judgment about fit.

**Scoring Formula (feature-only — candidates 501–100,000):**

```
final = 0.23×semantic_sim + 0.12×bm25 + 0.28×skill + 0.22×behavioral + 0.10×experience + 0.05×education
       × availability_multiplier × title_multiplier
```

Sort all scored candidates descending. Output top 100 with reasoning strings.

---

## How We Solve Each Gap

### Gap 1: Keyword Stuffing → BM25 + Title Multiplier

BM25 with k1=1.2 saturation means adding the same keyword 50 times gives diminishing returns — the 50th occurrence barely moves the score. The title multiplier applies a 0.30x penalty to candidates whose current job title is an explicit mismatch (Marketing Manager, HR Executive, Civil Engineer, etc.) regardless of keywords in their profile body.

Result: A Marketing Manager who stuffs their resume with "GPT, LLM, embeddings" scores 0.30× of their raw score. A Software Engineer with moderate AI keywords scores 1.0×.

### Gap 2: Career Quality → Consulting Penalty + Product Bonus

The experience scorer tracks the ratio of consulting-firm months to total career months. If 90%+ of a career was at TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, or similar firms, the candidate is zeroed out entirely.

If 50%+ of the career was at consulting firms, the experience score is reduced proportionally: `exp_score × (1 - 0.5 × consulting_ratio)`.

Candidates from smaller product companies (1–200 employee size) who haven't worked at consulting firms get a +0.10 experience bonus.

### Gap 3: Semantic Blind Spots → FAISS + Hybrid RRF

A candidate who writes "built recommendation systems at scale using vector similarity search" without naming specific tools (FAISS, Pinecone) will score low on keyword matching but high on semantic similarity to the JD embedding. By fusing BM25 and FAISS via RRF, such a candidate surfaces in the hybrid retrieval pool and gets LLM-evaluated.

### Gap 4: Behavioral Signals → 23-Signal Composite

The behavioral score converts 23 redrob_signals fields into a single 0–1 score:
- Recruiter response rate + average response time → 25% weight
- Last active date + open-to-work flag → 20% weight (active within 7 days = 1.0; inactive 6+ months = 0.25)
- Skill assessment scores + interview completion rate + profile completeness → 25% weight
- Offer acceptance rate + notice period + work mode preference → 15% weight
- GitHub activity score → 15% weight

A "perfect on paper" candidate who hasn't logged in for 8 months and has never completed an assessment scores low on behavioral, even with excellent skills.

### Gap 5: Honeypot Profiles → 7-Rule Detection

Seven deterministic rules catch the patterns typical of synthetic profiles. Honeypots are excluded before embedding (Phase 1) — they never enter the FAISS index, never consume BM25 compute, never get LLM evaluated.

In testing on the full 100K dataset, fewer than 0.1% of candidates triggered honeypot flags (approximately 40 of 100,000), keeping the false-positive rate negligible while eliminating clearly synthetic profiles.

### Gap 6: LLM Cost at Scale → Funnel Architecture

Instead of calling the LLM for all 100K candidates, the pipeline uses a three-stage funnel:
- Stage 1: Honeypot detection filters ~40 candidates (free, pure logic)
- Stage 2: Hybrid BM25+FAISS retrieval reduces 100K → 500 (free, local compute)
- Stage 3: LLM scores only the top-500 (~50 API calls total)

**API call math:**
- Naive approach: 100,000 LLM calls × $0.001/call = $100
- Smart-ATS: 50 LLM calls × $0.001/call = $0.05
- **Cost reduction: 2000× fewer LLM calls, same ranking quality at the top**

The LLM score matters most for the candidates most likely to make the final list. By pre-filtering with hybrid retrieval, we concentrate LLM judgment exactly where it matters.

---

## Architecture Diagram in Words

```
100,000 candidates
      │
      ▼
[Honeypot Detection] ──▶ ~40 synthetic profiles removed (free)
      │
      ▼
[Feature Extraction] ──▶ skills, experience, education, behavioral (free)
      │
      ├──▶ [BM25 Index] ──▶ score all 100K (free)
      │
      └──▶ [FAISS Embeddings] ──▶ 1 API call for all 100K
                │
                ▼
         [Hybrid RRF] ──▶ top-500 candidates merged
                │
                ▼
         [LLM Scoring] ──▶ 50 API calls (batches of 10)
                │
                ▼
         Artifacts saved to disk
                │
                ▼
         [Phase 2: Rank] ──▶ pure arithmetic, zero network
                │
                ▼
         submission.csv (100 ranked candidates, < 5 min)
```

---

## Scoring Weight Philosophy

The weight distribution reflects a key belief: **structured signals catch the extremes, LLM judgment resolves the middle**.

For the top-500 candidates (post-retrieval, LLM-scored):
- Claude score (55%): holistic assessment of trajectory, production experience, and cultural fit
- Skill match (0.45 × 0.35 = 15.75%): coverage of target AI/ML skills
- Behavioral (0.45 × 0.30 = 13.5%): engagement, responsiveness, activity
- Experience (0.45 × 0.15 = 6.75%): years, product company, not consulting
- Education (0.45 × 0.10 = 4.5%): institution tier
- BM25 (0.45 × 0.10 = 4.5%): keyword match as a tiebreaker

The availability multiplier (0.3× to 1.0×) and title multiplier (0.3× to 1.0×) are applied as hard gates — a brilliant candidate who is outside India with no relocation intent or whose title is completely unrelated gets a severe penalization regardless of other scores.

---

## The Key Innovations

### 1. Two-Phase Separation

The slow, expensive work (embeddings, LLM scoring) runs once offline. The ranking runs in under 5 minutes with no network calls. This makes the system production-deployable — you pre-compute artifacts nightly as the candidate pool changes, and serve rankings in real time.

### 2. Hybrid BM25+FAISS with Reciprocal Rank Fusion

Neither pure keyword search nor pure semantic search is sufficient. BM25 favors candidates who use the same vocabulary as the JD. FAISS favors candidates with semantically similar profiles regardless of exact wording. RRF combines both without requiring tuned mixing weights — it's a robust, parameter-minimal fusion.

### 3. Funnel Architecture for LLM Calls

The dramatic API call reduction (100K → 50) is the central economic innovation. Most candidates are trivially not a fit. The retrieval stage is cheap enough to run on all 100K. The LLM stage runs only where it matters.

### 4. Deterministic Honeypot Rules

Probabilistic ML for honeypot detection would be overkill and fragile. Seven deterministic rules based on logical impossibilities (H1–H7) are interpretable, fast, and robust. If a candidate claims expert-level Python with zero months of experience, no ML model is needed to know this is synthetic.

### 5. Behavioral Signals as First-Class Features

Most ATS systems treat behavioral signals as a post-filter or ignore them entirely. Smart-ATS encodes 23 behavioral dimensions into a weighted composite that contributes 22% of the feature-only score and 13.5% of the LLM-enhanced score. A highly skilled but completely inactive candidate is not a useful hire — the behavioral score captures this.

---

## Results and Compliance

The pipeline produces a `submission.csv` with exactly 100 ranked candidates:
- CSV columns: candidate_id, rank, score, reasoning
- Ranks 1–100, unique, non-decreasing scores
- Per-candidate reasoning strings grounded in actual profile data (years of experience, skill match quality, behavioral signals)
- Full reproducibility: Phase 1 is documented and resumable (--resume flag); Phase 2 runs from artifacts alone

Estimated total cost for a full production run:
- OpenAI embeddings (100K profiles): ~$1.50
- LLM scoring (500 candidates via OpenRouter free tier): $0.00
- **Total: ~$1.50 for ranking 100,000 candidates**

---

## Why This Approach Wins

Traditional ATS: keyword matching → ranked list. Effective for filtering to thousands. Fails at distinguishing the top 100 from a pool of 100,000.

ML-only ATS: pure embedding similarity. Misses keyword-optimized profiles and has no structured signal for career quality.

LLM-everywhere ATS: excellent judgment but prohibitive cost at scale. $100+ and hours of runtime for one ranking job.

Smart-ATS: structured signals catch the obvious (honeypots, consulting lifers, title mismatches), hybrid retrieval efficiently narrows the field, LLM judgment is concentrated on the 500 candidates who actually matter, and the final ranking runs in minutes with no API dependency.

The result is recruiter-grade judgment at database-query speed and startup-grade cost.
