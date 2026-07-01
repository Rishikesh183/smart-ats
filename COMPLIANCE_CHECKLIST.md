# Compliance Checklist — vs. official hack2skill submission spec

Sourced directly from the bundled `submission_spec.docx`, `job_description.docx`, `redrob_signals_doc.docx`, `README.docx`, `candidate_schema.json`, and `submission_metadata_template.yaml`. Cross-checked against the current state of `precompute.py`, `rank.py`, `honeypot.py`, `requirements.txt`, `README.md`, and `submission_metadata.yaml`.

Legend: ✅ Compliant · ⚠️ Partial / needs verification · ❌ Gap, needs action

## 1. CSV format (Spec §2–3)

| Item | Status | Notes |
|---|---|---|
| Columns `candidate_id,rank,score,reasoning` in order | ✅ | `rank.py` writes this exact header. |
| Exactly 100 data rows | ✅ | `validate_submission.py` confirms. |
| Ranks 1–100, each exactly once | ✅ | Validator confirms. |
| Unique `candidate_id`, exists in `candidates.jsonl` | ✅ | Validator confirms; ids drawn from the real pool. |
| Score non-increasing with rank | ✅ | Validator confirms. |
| Deterministic tie-break | ✅ | `rank.py` sorts by `(-score, candidate_id)`. |
| **Filename** = registered participant/team ID (e.g. `team_xxx.csv`) | ❌ | Current file is `submission.csv`. The literal participant ID format hasn't been confirmed from the portal. Rename before upload. |
| UTF-8 encoding | ✅ | Default Python CSV write is UTF-8. |

## 2. Reasoning quality (Spec §3, Stage 4 manual review)

| Check | Status | Notes |
|---|---|---|
| Non-empty, non-templated reasoning | ✅ | Per-row reasoning strings, vary by candidate. |
| References specific facts (years, title, skills, signals) | ✅ | Confirmed in spot-checked rows. |
| Connects to JD requirements, not generic praise | ⚠️ | True for the ~11 rows with genuine LLM (`claude_score`) backing; the other ~89 rows are generated from the feature-only fallback template and are more formulaic. Worth a manual skim of a 10-row random sample before submitting, since that's exactly what Stage 4 does. |
| No hallucinated skills/employers | ⚠️ | Not yet specifically audited row-by-row; LLM-scored rows already passed sanity inspection earlier in this session, fallback rows are template-built from real fields so risk is low but unverified. |
| Rank-consistent tone | ✅ | Confirmed on spot-checked top candidates (mediocre-fit language now correctly tracks mid-pack scores after the `claude_score` fix). |

## 3. Compute constraints during ranking (Spec §3)

| Constraint | Status | Notes |
|---|---|---|
| ≤5 min wall-clock | ✅ | `rank.py` runs in well under a minute once artifacts exist. |
| ≤16GB RAM | ⚠️ | Likely fine (FAISS index + feature files are tens of MB), but never explicitly measured. |
| CPU only, no GPU | ✅ | `faiss-cpu`, no CUDA calls in `rank.py`. |
| No network calls during ranking | ✅ | `rank.py` only reads local artifact files — confirmed by code read, no API clients instantiated in the ranking path. |
| ≤5GB intermediate disk state | ✅ | `artifacts/` totals ~345MB (`faiss_index.pkl` 295MB is the bulk). |

## 4. Honeypot detection (Spec §7, redrob_signals_doc)

| Item | Status | Notes |
|---|---|---|
| 7 rules implemented (H1 expert-skill/0-duration, H2 years-vs-history mismatch, H3 end<start date, H4 shotgun-expert, H5 bot-skeleton, H6 endorsement-farming, H7 impossible years) | ✅ | `honeypot.py`, unchanged from original design after this session's investigation. |
| Coverage vs. spec's "~80 seeded honeypots" example pattern ("8 years of experience at a company founded 3 years ago") | ❌ | `candidate_schema.json` has **no "company founded year" field anywhere** — this exact example can't be checked from the data we're given, so it's not implementable as literally described. This is a real gap but may be inherent to the dataset, not a code bug — current rules only catch internally-inconsistent profiles (skills/dates/career math), not employer-plausibility. |
| Honeypots excluded from FAISS index / scoring pool | ✅ | `precompute.py` filters `is_honeypot` candidates out before building the index. |
| False-positive rate kept low (don't over-flag legit noisy data) | ✅ | Verified this session — two candidate rules (salary min>max, last_active<signup) were tested and reverted after they flagged ~25% of the dataset as honeypots; the real signal in this dataset is much rarer (40/100,000 with the current 7 rules). |
| Honeypot rate in final top-100 < 10% | ⚠️ | Never explicitly counted against the current `submission.csv`. Quick scriptable check before final submission. |

## 5. Job-description "traps" (job_description.docx)

| Trap | Status | Notes |
|---|---|---|
| Keyword-stuffing trap (AI keywords but wrong title, e.g. "Marketing Manager") | ✅ | `rank.py` applies a 0.3x title-mismatch multiplier. |
| Plain-language Tier-5 trap (no buzzwords, but real production rec-sys experience) | ⚠️ | Depends on semantic similarity + LLM judgment correctly surfacing these profiles; not separately tested with a targeted example. |
| Consulting disqualifier scoped to "entire career," not current employer | ⚠️ | Confirmed conceptually in `rank.py`'s scoring earlier in the session, not freshly re-verified this pass. |
| Down-weight behaviorally-inactive "perfect-on-paper" candidates | ✅ | `availability_multiplier` / behavioral component in `rank.py` does this. |

## 6. Hidden scoring formula alignment (Spec §4)

| Item | Status | Notes |
|---|---|---|
| System optimizes for ranking quality at top of list (NDCG@10 weighted 0.50) | ⚠️ | Architecture mismatch found this session: `precompute.py`'s LLM-scoring shortlist (hybrid BM25+FAISS RRF, top-500) and `rank.py`'s actual retrieval pool (FAISS-only top-2000, BM25 folded in only as a scoring feature) only partially overlap. Result: only ~11/100 final rows currently carry a genuine LLM judgment; the rest fall back to feature-only scoring. This directly affects ranking quality at the top, which is 80% of the composite score (NDCG@10 + NDCG@50). **Recommended fix, not yet applied**: make `rank.py`'s retrieval use the same hybrid BM25+FAISS RRF approach as `precompute.py`, so the candidates actually scored by the LLM are the ones most likely to end up in the final top-100. |

## 7. Repo & reproducibility (Spec §10.3)

| Item | Status | Notes |
|---|---|---|
| README.md with setup + exact reproduce commands | ⚠️ | README exists and documents Phase 1/Phase 2 commands, but it's stale: describes OpenAI embeddings (`text-embedding-3-small`) + Claude/OpenRouter LLM scoring. The actual `.env` now points to **local Ollama** for everything (`EMBEDDING_PROVIDER=ollama`, `OLLAMA_EMBED_MODEL=embeddinggemma`, `OPENAI_BASE_URL=http://localhost:11434/v1`, `OPENAI_CHAT_MODEL=qwen2.5:3b`). Needs an update before submission — this is also a Stage 5 interview risk ("contradicts submitted code" is an explicit elimination criterion). |
| Full source code, no hidden/manual steps | ⚠️ | Source is committed, but `git status` currently shows **uncommitted changes** to `precompute.py`, `rank.py`, and `requirements.txt` (this session's `claude_score` sanitization fix and ollama/sentence-transformers deps). Must commit before the repo reflects what actually produced the submission. |
| Pre-computed artifacts OR a script to produce them | ❌ | `.gitignore` excludes `challenge/artifacts*/` entirely — a fresh clone has **no FAISS index, no feature/BM25/claude scores**. `rank.py` cannot run standalone without them. The fallback (`precompute.py` regenerates them) depends on a **locally running Ollama instance with two specific models pre-pulled** — not something a grader's sandboxed reproduction environment is likely to have, and pulling multi-GB models requires network access the ranking step isn't allowed to use. This is the single highest-severity finding: as configured today, a grader cloning the repo cannot reproduce the CSV at all. Needs one of: (a) commit/LFS the artifacts, (b) host them as a release asset with a download step, or (c) document a from-scratch Ollama setup path and accept it's a pre-computation-only dependency (allowed to exceed 5 min, but must still be genuinely runnable by a third party). |
| `requirements.txt` with versions | ✅ | Present, minimum-version pinned, now includes `ollama` and `sentence-transformers`. |
| `submission_metadata.yaml` at repo root | ⚠️ | Present, but several fields are stale/incomplete — see next section. |

## 8. submission_metadata.yaml vs. portal/template (Spec §10.2)

| Field | Status | Notes |
|---|---|---|
| `team_name` | ✅ | "smart-ats" |
| `primary_contact` / `team_members` | ⚠️ | Only lists "Rishi" as the sole member/Team Lead. If more than one person (e.g. you) did engineering work on this submission, the team roster should reflect that — Stage 5 interview checks who can actually defend the code. |
| `github_repo` | ✅ | Points to a real, specific repo URL. |
| `sandbox_link` | ❌ | **Empty** (`""`, marked `TODO`). This is a mandatory field (Spec §10.2/10.5) — "submissions without a working sandbox link are flagged at Stage 1." Needs a HuggingFace Space / Streamlit / Replit / Colab / Docker / Binder demo that runs the ranker end-to-end on a ≤100-candidate sample in ≤5 min CPU. Not yet started. |
| `reproduce_command` | ❌ | Currently `python challenge/rank.py --candidates ./candidates.jsonl --out ./submission.csv` run implicitly from repo root — but `rank.py`'s `--artifacts` flag defaults to `artifacts/` relative to the **current working directory**, and the real artifacts live at `challenge/artifacts/`. As written, this command would fail to find them. Either add `--artifacts challenge/artifacts/` or document `cd challenge` first (which is what the README's own version correctly does). |
| `compute` block | ⚠️ | Says "Local Windows machine" / "Windows 11" — doesn't match the Mac-based reruns done in this session. Should reflect whichever machine actually produced the final submitted CSV. |
| `ai_tools_used` / `ai_usage_summary` / `methodology_summary` | ❌ | Describes Claude (Anthropic API) scoring 500 candidates and OpenAI embeddings — **stale**. The actual pipeline run uses local Ollama models (`qwen2.5:3b`, `embeddinggemma`) for everything, with no real Claude/OpenAI API calls in the scoring path. This is the kind of declaration-vs-code mismatch the spec explicitly flags as a stronger negative signal than the AI use itself (§10.4) — needs a rewrite to match reality before submission. |
| `declarations` | ⚠️ | `reproduction_tested: true` should be re-verified given the reproduce-command bug above. |

## 9. What's genuinely solid right now

Git history shows real iterative commits across multiple PRs (10+ commits, not a single dump) — this is exactly what Stage 4's "git history authenticity" check is looking for. The CSV format, rank/tie-break logic, and core honeypot/title/behavioral heuristics are sound and validated. The claude_score out-of-range bug found and fixed this session was a real correctness issue that's now resolved with a permanent guard in `precompute.py`.

## Highest-priority action items, in order

1. Fix the retrieval-pool mismatch in `rank.py` so most of the top-100 actually carries a genuine LLM score (currently ~11/100) — directly affects the hidden NDCG@10/50 scoring.
2. Commit `precompute.py` / `rank.py` / `requirements.txt` changes from this session.
3. Decide how `rank.py` will find its artifacts on a fresh clone — un-gitignore them (likely needs Git LFS for the 295MB FAISS index) or document/host them another way. Without this, Stage 3 reproduction fails outright.
4. Rewrite `README.md` and `submission_metadata.yaml` to describe the actual Ollama-based pipeline, not the original OpenAI/Claude design.
5. Set up the mandatory sandbox/demo link (Spec §10.5) and fill in `sandbox_link`.
6. Fix `reproduce_command` to actually work from repo root (or specify `cd challenge` first).
7. Rename the final CSV to the registered participant ID before upload.
8. Spot-check honeypot rate in the current `submission.csv` top-100, and audit a random 10-row reasoning sample the way Stage 4 will.
