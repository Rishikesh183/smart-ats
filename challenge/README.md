# Challenge — Semantic Candidate Ranking

Two-phase system:
1. **precompute.py** — offline, one-time, can use API calls and take hours  
2. **rank.py** — fast (<5 min), no network, reads artifacts → outputs `submission.csv`

---

## Setup

```bash
cd challenge/
pip install -r requirements.txt

# Copy env vars
cp ../backend/.env .env
# Make sure ANTHROPIC_API_KEY is set
```

---

## Step 1 — Pre-compute (run once, can resume)

```bash
python precompute.py \
  --candidates "../[PUB] India_runs_data_and_ai_challenge/candidates.jsonl" \
  --out artifacts/ \
  --top-k 500 \
  --batch-size 5 \
  --resume
```

This produces:
- `artifacts/candidate_features.jsonl` — numeric features for all 100K candidates
- `artifacts/faiss_index.pkl` — FAISS cosine-similarity index
- `artifacts/claude_scores.jsonl` — Claude scores for the top-500 FAISS candidates

**Resumable**: if it's interrupted, re-run with `--resume` to continue from where it left off.

**Cost estimate**: ~500 Claude Sonnet calls at 5 candidates each = 100 API calls ≈ $1–2 USD.

---

## Step 2 — Rank (fast, no network)

```bash
python rank.py \
  --candidates "../[PUB] India_runs_data_and_ai_challenge/candidates.jsonl" \
  --out submission.csv \
  --artifacts artifacts/
```

Outputs `submission.csv` with exactly 100 rows:
```
candidate_id,rank,score,reasoning
CAND_0001234,1,0.921456,"Strong retrieval/ranking background; 7y experience; strong AI/ML skill match"
...
```

---

## Scoring formula

When Claude score is available (top-500):
```
final = (0.55 × claude_score
       + 0.45 × (0.40×skill + 0.35×behavioral + 0.15×experience + 0.10×education))
      × availability_multiplier
```

When Claude score is NOT available (outside top-500):
```
final = (0.35×semantic_similarity
       + 0.28×skill_score
       + 0.22×behavioral_score
       + 0.10×experience_score
       + 0.05×education_score)
      × availability_multiplier
```

Automatic disqualification (score = 0):
- Honeypot profiles (impossible career data)
- Purely consulting career (≥90% at TCS/Infosys/Wipro etc.)
- Claude marks `is_disqualified: true`

---

## File structure

```
challenge/
  jd.py              — JD text + TARGET_SKILLS + constants
  features.py        — Feature extraction from JSONL schema
  honeypot.py        — Impossible-profile detection
  behavioral.py      — redrob_signals → 0-1 behavioral score
  precompute.py      — Offline pipeline (FAISS + Claude)
  rank.py            — Fast ranking → submission.csv
  requirements.txt
  artifacts/         — Created by precompute.py
    faiss_index.pkl
    candidate_features.jsonl
    claude_scores.jsonl
```
