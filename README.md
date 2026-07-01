# 🧠 Smart-ATS — Production-Grade AI Candidate Ranking Pipeline

> **Redrob AI Hackathon · India Runs Data & AI Challenge**  
> Ranking 100,000 candidates against a Senior AI Engineer JD using hybrid retrieval, LLM scoring, and behavioural signal fusion.

---

## 📌 TL;DR

| Metric | Value |
|---|---|
| Candidates processed | 100,000 |
| Final ranked output | Top 100 |
| Embedding model | `text-embedding-3-small` (OpenAI, 1536-dim) |
| LLM scoring model | `gpt-4o-mini` |
| Retrieval strategy | Hybrid BM25 + FAISS with Reciprocal Rank Fusion (RRF) |
| LLM-backed rows in final top-100 | **100 / 100** |
| Ranking runtime (after precompute) | **< 1 minute, CPU-only, zero network calls** |
| Honeypot detection rules | 7 (H1–H7) |

---

## 🗂️ Repository Structure

```
smart-ats/
├── challenge/
│   ├── precompute.py          # Offline pipeline: embed → BM25 → FAISS → LLM score
│   ├── rank.py                # Fast ranking: loads artifacts, scores, writes CSV
│   ├── retrieval.py           # BM25Retriever + Reciprocal Rank Fusion
│   ├── features.py            # Skill / experience / education feature extraction
│   ├── behavioral.py          # Behavioural signal scoring (redrob_signals)
│   ├── honeypot.py            # 7-rule impossible-profile detector
│   ├── jd.py                  # JD text + scoring weights
│   ├── artifacts/             # Pre-computed artifacts (gitignored — large files)
│   │   ├── faiss_index.pkl    # FAISS IndexFlatIP (1536-dim, 99,960 candidates)
│   │   ├── candidate_features.jsonl  # Structured features for all 100K
│   │   ├── llm_scores.jsonl       # gpt-4o-mini scores for top-2000 shortlist
│   │   ├── bm25_scores.jsonl         # Normalised BM25 scores for all 99,960
│   │   └── jd_embedding.npy          # Pre-computed JD embedding vector
│   ├── submission.csv         # Final ranked output (100 rows)
│   └── requirements.txt
├── submission_metadata.yaml
├── COMPLIANCE_CHECKLIST.md
└── README.md
```

---

## 🏗️ Architecture

```
                   ┌──────────────────────────────────────────────────────┐
                   │              PHASE 1 — PRECOMPUTE (offline, ~20 min) │
                   └──────────────────────────────────────────────────────┘

  candidates.jsonl (100,000)
          │
          ▼
  ┌────────────────────┐      ┌──────────────────────────────────────────┐
  │  Feature Extract   │─────▶│  candidate_features.jsonl                │
  │  + Honeypot Det.   │      │  skill_score, experience_score,          │
  │  (40 honeypots     │      │  behavioral_score, education_score,      │
  │   detected)        │      │  title_multiplier, availability_mult,    │
  └────────────────────┘      │  is_honeypot, purely_consulting          │
          │                   └──────────────────────────────────────────┘
          │ 99,960 valid candidates
          ▼
  ┌──────────────────────────┐      ┌──────────────────────────────────┐
  │  OpenAI Embeddings       │─────▶│  FAISS IndexFlatIP               │
  │  text-embedding-3-small  │      │  dim=1536, 99,960 vectors        │
  │  batch_size=256          │      │  faiss_index.pkl (~350 MB)       │
  └──────────────────────────┘      └──────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────┐      ┌──────────────────────────────────┐
  │  BM25 Indexing           │─────▶│  bm25_scores.jsonl               │
  │  unigram + bigram        │      │  Normalised BM25 scores          │
  │  + trigram n-grams       │      │  for all 99,960 candidates       │
  └──────────────────────────┘      └──────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                  Hybrid Retrieval — RRF Fusion                   │
  │                                                                  │
  │   BM25 top-4000  ──┐                                            │
  │                     ├─▶  RRF score = Σ 1/(k + rankᵢ)  ──▶  Top-2000
  │   FAISS top-4000 ──┘    k = 60  (Cormack et al., 2009)         │
  └──────────────────────────────────────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────┐      ┌──────────────────────────────────┐
  │  gpt-4o-mini Scoring     │─────▶│  llm_scores.jsonl             │
  │  2,000 candidates        │      │  claude_score ∈ [0,1]            │
  │  batch_size=10           │      │  fit_summary (1-2 sentences)     │
  │  ~200 API calls          │      │  is_disqualified (bool)          │
  └──────────────────────────┘      └──────────────────────────────────┘


                   ┌──────────────────────────────────────────────────────┐
                   │          PHASE 2 — RANK (< 1 min · CPU · no network) │
                   └──────────────────────────────────────────────────────┘

  Load artifacts → Hybrid BM25+FAISS RRF → 2,000-candidate pool
          │
          ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  SCORING FORMULA                                                     │
  │                                                                      │
  │  IF gpt-4o-mini score available (top-2000 shortlist):                │
  │    score = (0.55 × claude_score                                      │
  │           + 0.45 × (0.35 × skill_score                              │
  │                   + 0.30 × behavioral_score                         │
  │                   + 0.15 × experience_score                         │
  │                   + 0.10 × education_score                          │
  │                   + 0.10 × bm25_norm))                              │
  │           × availability_multiplier                                  │
  │           × title_multiplier                                         │
  │                                                                      │
  │  honeypot detected    → score = 0.0                                  │
  │  purely consulting    → score = 0.0                                  │
  │  is_disqualified=True → score = 0.0                                  │
  └──────────────────────────────────────────────────────────────────────┘
          │
          ▼
  Sort descending → top-100 → submission.csv
```

---

## 🔬 Technical Deep Dive

### 1. Hybrid Retrieval with Reciprocal Rank Fusion

Pure FAISS retrieval misses candidates whose strength is keyword-precise rather than semantically broad; pure BM25 misses paraphrased equivalents. We fuse both with **Reciprocal Rank Fusion** (Cormack et al., 2009):

```
RRF_score(d) = Σᵢ  1 / (k + rankᵢ(d))       k = 60
```

With `k=60`, this smoothly merges ranked lists so candidates appearing consistently high in both BM25 and FAISS rise to the top, while candidates appearing only in one list are appropriately penalised. The fused top-2000 is sent to the LLM for deep judgment.

BM25 is indexed with **unigram + bigram + trigram** n-grams after English stopword removal, giving sensitivity to compound technical terms like `vector_database`, `embedding_based_retrieval`, `learning_to_rank`, and `reciprocal_rank_fusion` that single-token BM25 would miss entirely.

### 2. Embedding Architecture

All 99,960 non-honeypot candidates are embedded using **OpenAI `text-embedding-3-small`** (1536 dimensions, normalised L2 → unit sphere). FAISS `IndexFlatIP` performs exact cosine similarity (inner product on unit vectors equals cosine). The JD is embedded with the same model and stored as `jd_embedding.npy`, ensuring the vector space is shared.

1536-dim embeddings provide substantially richer semantic representation than 384-dim local models — capturing nuanced distinctions between a Research Scientist who has never shipped production code and an Applied ML Engineer who has owned the full ML lifecycle end to end.

### 3. LLM Scoring (gpt-4o-mini)

The 2,000 candidates retrieved by hybrid RRF are each scored by `gpt-4o-mini` against the full JD. The model produces per candidate:

- `claude_score ∈ [0, 1]` — calibrated holistic fit score
- `fit_summary` — 1–2 sentence reasoning grounded in specific career evidence
- `is_disqualified` — explicit flag for JD "do not want" patterns

The LLM prompt is engineered to reward:
- ✅ Production deployment of ML models at scale (not just notebooks or papers)
- ✅ Embedding-based retrieval, ranking pipelines, evaluation frameworks (NDCG, MAP)
- ✅ Honest acknowledgment of gaps where relevant

And penalise:
- ❌ Pure research / academic careers without shipping production code
- ❌ Entire career at named consulting firms (TCS, Infosys, Wipro, Cognizant, Capgemini, Accenture)
- ❌ CV / Speech / Robotics specialists without NLP/IR production background
- ❌ Title inflation without shipped product evidence

**Result: 100 / 100 final ranked candidates carry a genuine gpt-4o-mini judgment.**

### 4. Feature Engineering

| Feature | Signal | Notes |
|---|---|---|
| `skill_score` | JD keyword match weighted by proficiency × duration × endorsements | Composite over matched skills |
| `experience_score` | Years in 5–9y ideal JD band; product-company bonus | Band-scored, capped [0, 1] |
| `behavioral_score` | Response rate, recency, GitHub activity, platform assessment scores | From `redrob_signals` |
| `education_score` | Degree tier (PhD / MTech / BTech), CS-adjacent field bonus | Tier lookup table |
| `title_multiplier` | Title–role alignment | 1.0 (ML titles) · 0.5 (adjacent) · 0.3 (mismatch) |
| `availability_multiplier` | Last-active recency, notice period, open-to-work flag | Range [0.5, 1.0] |

### 5. Honeypot Detection (Rules H1–H7)

| Rule | Description | Trigger |
|---|---|---|
| H1 | Expert/Advanced skill with `duration_months = 0` | ≥ 3 occurrences |
| H2 | Stated YoE > 2× actual career history length AND gap > 10 years | Both conditions |
| H3 | Job `end_date` before `start_date` | Any single entry |
| H4 | >30 skills all at "expert" proficiency | Shotgun expert pattern |
| H5 | `profile_completeness_score = 0` with ≥ 5 career entries | Bot skeleton |
| H6 | >50 endorsed-beginner skills | Endorsement farming |
| H7 | `years_of_experience < 0` or `> 60` | Impossible value |

Detected honeypots are excluded from the FAISS index entirely and score `0.0` in ranking. **40 honeypots detected across 100,000 candidates (0.04%).**

---

## 🚀 Quickstart

### Prerequisites

```bash
pip install -r challenge/requirements.txt
```

Key dependencies: `faiss-cpu`, `openai`, `rank-bm25`, `numpy`, `sentence-transformers`

### Environment Setup

Create `challenge/.env`:

```env
OPENAI_API_KEY=your_key_here

EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

LLM_PROVIDER=openai
OPENAI_CHAT_MODEL=gpt-4o-mini
```

### Phase 1 — Precompute (run once, ~20 min, ~$2.50 in API costs)

```bash
cd challenge

export $(grep -v '^#' .env | grep '=' | xargs)

python precompute.py \
  --candidates path/to/candidates.jsonl \
  --top-k 2000 \
  --batch-size 10
```

Resumable with `--resume` if interrupted — skips already-completed steps.

### Phase 2 — Rank (< 1 min, fully offline)

```bash
python rank.py \
  --candidates path/to/candidates.jsonl \
  --artifacts artifacts/ \
  --out submission.csv
```

Zero network calls. Zero GPU. Runs in under a minute on any CPU.

---

## 📊 Results

```
Total candidates:            100,000
Honeypots detected:               40  (0.04%)
Valid candidates indexed:     99,960
LLM shortlist (RRF top-k):    2,000
LLM-backed final rows:       100/100  ✅
Ranking runtime:               < 1 min (CPU-only, no network)
Embedding dimensions:            1536  (text-embedding-3-small)
FAISS index size:              ~350 MB
```

### Top 10 Ranked Candidates

| Rank | Score | Candidate | YoE | Title | Company |
|------|-------|-----------|-----|-------|---------|
| 1 | 0.812 | CAND_0043228 | 6.8y | Applied ML Engineer | Zoho |
| 2 | 0.782 | CAND_0000031 | 6.0y | Recommendation Systems Engineer | Swiggy |
| 3 | 0.781 | CAND_0061257 | 8.0y | Staff Machine Learning Engineer | LinkedIn |
| 4 | 0.773 | CAND_0018499 | 7.2y | Senior ML Engineer | Zomato |
| 5 | 0.773 | CAND_0042029 | 6.5y | Senior Data Scientist | Flipkart |
| 6 | 0.772 | CAND_0005649 | 7.4y | Senior Data Scientist | Sarvam AI |
| 7 | 0.770 | CAND_0041669 | 8.0y | Recommendation Systems Engineer | CRED |
| 8 | 0.768 | CAND_0068811 | 8.0y | Applied ML Engineer | Freshworks |
| 9 | 0.767 | CAND_0094759 | 8.6y | Lead AI Engineer | Meta |
| 10 | 0.764 | CAND_0096172 | 5.2y | NLP Engineer | Krutrim |

---

## 🎯 Why This Beats a Naive Embedding Ranker

| Approach | Problem |
|---|---|
| FAISS-only ranking | Misses keyword-strong candidates; retrieval pool diverges from LLM-scored shortlist |
| BM25-only | Misses semantically equivalent candidates using different vocabulary |
| LLM on all 100K | Computationally infeasible; prohibitive API cost |
| **Hybrid RRF + LLM on top-2K** | ✅ Fast retrieval, semantic richness, calibrated LLM judgment |

**The critical insight that drove our biggest quality jump:** the FAISS retrieval pool in `rank.py` and the LLM-scored shortlist in `precompute.py` must use the **same hybrid RRF retrieval strategy**. In an earlier version, `precompute.py` built its LLM shortlist via BM25+FAISS RRF while `rank.py` retrieved candidates via FAISS-only — the two pools barely overlapped. Result: only 11/100 final rows had genuine LLM judgment. After aligning both phases to hybrid RRF, **100/100 final rows carry real gpt-4o-mini reasoning.**

---

## 🛡️ JD Trap Handling

The JD contains deliberate traps to test ranker robustness:

| Trap | Our Handling |
|---|---|
| **Keyword stuffing** — AI keywords but wrong role (e.g. Marketing Manager) | `title_multiplier = 0.3` applied before final score |
| **Plain-language Tier-5** — no buzzwords but genuine production rec-sys experience | BM25 trigram + FAISS semantic both surface these profiles independently |
| **Consulting disqualifier** — entire career at named IT services firms | `purely_consulting` flag → `score = 0.0` regardless of LLM score |
| **Behaviorally inactive** — perfect on paper but dormant 12+ months | `behavioral_score` + `availability_multiplier` compound down-weight |
| **Experience band** — JD specifies 5–9 years | `experience_score` peaks in band; outliers still considered if LLM score is strong |
| **Research-only** — academic papers, no shipped product | LLM prompt explicitly penalises pure-research careers without production evidence |
| **CV / Speech / Robotics without NLP/IR** | LLM explicitly prompted to flag; low `skill_score` also catches this |

---

## ⚙️ Compute Constraints Compliance

| Constraint | Spec Limit | Actual |
|---|---|---|
| Ranking wall-clock | ≤ 5 minutes | **< 1 minute** |
| RAM during ranking | ≤ 16 GB | ~1–2 GB |
| GPU required | Must be CPU-only | ✅ `faiss-cpu`, no CUDA |
| Network during ranking | None allowed | ✅ Zero network calls in `rank.py` |
| Intermediate disk | ≤ 5 GB | ~350 MB total |

---

## 🔄 Reproducibility

```bash
# Full end-to-end from raw candidates.jsonl:

cd challenge
export $(grep -v '^#' .env | grep '=' | xargs)

# Phase 1 (~20 min, requires OpenAI API key)
python precompute.py \
  --candidates path/to/candidates.jsonl \
  --top-k 2000 \
  --batch-size 10

# Phase 2 (< 1 min, fully offline)
python rank.py \
  --candidates path/to/candidates.jsonl \
  --artifacts artifacts/ \
  --out submission.csv
```

**Estimated Phase 1 cost:** ~$2.50 total (embeddings for 100K at `text-embedding-3-small` rates + gpt-4o-mini scoring for 2K candidates).

---

## 📁 Key Files

| File | Description |
|---|---|
| `challenge/submission.csv` | Final 100-row ranked output |
| `challenge/precompute.py` | Full offline pipeline (embed → BM25 → FAISS → LLM) |
| `challenge/rank.py` | Fast ranker (< 1 min, no network) |
| `challenge/retrieval.py` | BM25Retriever + RRF hybrid retrieval |
| `challenge/honeypot.py` | 7-rule honeypot detector (H1–H7) |
| `challenge/features.py` | Feature extraction (skill, experience, education) |
| `challenge/behavioral.py` | Behavioural signal scoring |
| `challenge/jd.py` | JD text + scoring weights |
| `submission_metadata.yaml` | Team + compute declaration |
| `COMPLIANCE_CHECKLIST.md` | Full spec compliance audit |

---

## 👥 Team — Omni Coders

**Rishi Kesh** · **Narra Satya Sai Charan**

---

*Built for the [India Runs Data & AI Challenge](https://hack2skill.com/event/india_runs) by Redrob × hack2skill.*
