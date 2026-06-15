# Smart ATS — Semantic Candidate Ranking

An AI recruiting pipeline that ranks candidates the way a great recruiter would — by understanding what a role *actually needs*, not by matching keywords.

**Prime directive: don't lose a good candidate.**

---

## What it does

1. **Pass 1 — Rubric Builder**: LLM reads the JD and outputs a structured, editable rubric (parameters, weights, anchors, hard gates). A recruiter can inspect and tweak it before any candidate is scored.

2. **Stage 1 — Semantic Retrieval**: Embeds all candidate profiles with `sentence-transformers`, builds a FAISS index, retrieves the top-N most semantically similar candidates. Banded into `advance / rescue / drop` — equivalent stacks (Vue≈React, SvelteKit≈Next.js, Remix≈Next.js) surface naturally.

3. **Stage 2 — Intelligent Gates**: Only true must-haves (work authorization, mandatory licenses) can drop a candidate. Never kills candidates for missing a specific keyword.

4. **Stage 3 — Evidence-Grounded Scoring**: LLM scores each candidate on each rubric parameter with exact verbatim evidence spans. No score exists without a cited source.

5. **Stage 4a — Critic Agent** (high/extra-high modes): Re-scores rescue-band candidates who fell just below the retrieval cutoff. Recovers candidates with equivalent stacks, career switchers, and low-keyword-but-high-signal writers.

6. **Stage 4b — Comparative Re-rank**: LLM produces an internally consistent final ordering of the top finalists.

7. **Output**: Ranked shortlist with explainable evidence per parameter + rescued-candidates report.

---

## Recall demo (the headline metric)

The planted-candidate recall test injects 5 known-strong-but-keyword-poor candidates:

| Candidate | Why keyword-poor |
|---|---|
| C002 Priya Sharma | Vue/Nuxt instead of React/Next.js |
| C003 Marcus Williams | Career switcher, Next.js learner |
| C009 Dmitri Volkov | SvelteKit (architecturally identical to Next.js) |
| C014 Gabriela Souza | Remix (same SSR paradigm as Next.js) |
| C020 Yuki Tanaka | Built Next.js adapter but no "Next.js projects" |

Expected results:

| Mode | Planted recall | Shortlisted | Rescued |
|---|---|---|---|
| normal | ~40–60% | ~8–12 | 0 |
| high | ~60–80% | ~10–14 | 2–4 |
| extra_high | ~80–100% | ~12–16 | 3–5 |

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- An Anthropic API key (or OpenRouter)

### Local (no Docker)

```bash
# 1. Clone and enter the project
cd smart-ats

# 2. Copy env and fill in your API key
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY=sk-ant-...

# 3. Backend
cd backend
pip install -r requirements.txt

# Generate the synthetic dataset (already committed, but re-runnable)
python data/generate_dataset.py

# Verify everything works
python -m app.data.ingest

# Start the API server
uvicorn app.main:app --reload --port 8000

# 4. Frontend (new terminal)
cd ../frontend
npm install
npm run dev
# → http://localhost:3000
```

### Docker (one-command)

```bash
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY=sk-ant-...
docker compose up --build
# → Frontend: http://localhost:3000
# → API:      http://localhost:8000
```

> **Note:** First start downloads the `all-MiniLM-L6-v2` embedding model (~80MB). The Docker image pre-downloads it at build time.

---

## Running the eval harness

```bash
cd backend

# Recall test across all modes (takes ~5-10 min, uses LLM for all modes)
python -m eval.planted

# Explainability + determinism check
python -m eval.checks
```

Or trigger via the UI: click **📊 Eval Harness** in the top-right, then **Run Eval**.

---

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/rubric` | POST | Build rubric from JD `{"job_description": "..."}` |
| `/rubric` | PUT | Accept edited rubric |
| `/rank` | POST | Run pipeline `{"job_description": "...", "mode": "normal\|high\|extra_high", "rubric": {...}}` |
| `/result` | GET | Fetch last result |
| `/eval` | POST | Run eval harness `{"modes": ["normal", "high", "extra_high"]}` |
| `/candidates` | GET | List all candidates (debug) |

---

## Project structure

```
smart-ats/
├── .env.example
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── data/
│   │   ├── generate_dataset.py   # generates candidates.csv
│   │   └── candidates.csv        # 20 synthetic candidates (5 planted)
│   ├── app/
│   │   ├── config.py             # settings (pydantic-settings)
│   │   ├── models.py             # all pydantic models
│   │   ├── main.py               # FastAPI app + endpoints
│   │   ├── run.py                # pipeline orchestrator
│   │   ├── llm/
│   │   │   ├── client.py         # LLMClient interface (Anthropic + OpenRouter)
│   │   │   └── prompts.py        # all prompts (rubric, scorer, critic, rerank)
│   │   ├── data/
│   │   │   ├── ingest.py         # load CSV/JSON dataset
│   │   │   └── normalize.py      # → CandidateProfile
│   │   └── pipeline/
│   │       ├── rubric.py         # Pass 1
│   │       ├── extract.py        # Stage 0
│   │       ├── retrieval.py      # Stage 1 + FAISS + rescue band
│   │       ├── gates.py          # Stage 2
│   │       ├── scorer.py         # Stage 3
│   │       ├── critic.py         # Stage 4a (rescue band critic)
│   │       └── rerank.py         # Stage 4b (comparative re-rank)
│   └── eval/
│       ├── planted.py            # recall test
│       └── checks.py             # explainability + determinism
└── frontend/
    └── src/
        ├── app/
        │   ├── layout.tsx
        │   ├── page.tsx          # main UI (JD → rubric → ranked list)
        │   └── globals.css
        ├── components/
        │   ├── CandidateCard.tsx # expandable card with evidence per parameter
        │   ├── RubricEditor.tsx  # inline rubric editing modal
        │   └── EvalPanel.tsx     # recall metrics UI
        └── lib/
            └── api.ts            # typed API client
```

---

## Key design principles implemented

- **Semantic equivalence**: Vue≈React, SvelteKit≈Next.js, Remix≈Next.js — the LLM justifies equivalence direction in every score.
- **Rubric-first**: JD understanding and candidate scoring are separate passes. The rubric is a first-class, editable artifact.
- **Evidence or nothing**: Every score cites verbatim spans. Missing evidence → lower confidence, not inflated score.
- **Funnel for cost**: Cheap embeddings filter first (~500→50); expensive LLM scoring runs only on survivors.
- **Recall knob**: Three modes trade compute for fewer missed candidates. The UI is honest about this trade-off.

---

## Swapping providers

The `LLMClient` in `app/llm/client.py` is the only file with provider-specific code. To switch:

```env
# .env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=anthropic/claude-3-haiku
```

Embeddings can be swapped by changing `EMBEDDING_MODEL` in `.env` — any `sentence-transformers` model works. The FAISS index is re-built and cached automatically.
