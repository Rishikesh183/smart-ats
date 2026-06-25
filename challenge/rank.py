"""
Fast ranking script — runs in < 5 minutes, NO network calls.
Loads pre-computed artifacts from precompute.py, embeds the JD once,
does FAISS retrieval, applies the scoring formula, outputs submission.csv.

Usage:
  python rank.py \
    --candidates "../[PUB] India_runs_data_and_ai_challenge/candidates.jsonl" \
    --out submission.csv \
    [--artifacts artifacts/] \
    [--top-n 100]

Scoring formula (when Claude score is available):
  final = (CLAUDE_WEIGHT × claude_score
           + FEATURE_WEIGHT × (
               w_skill  × skill_score
             + w_behav  × behavioral_score
             + w_exp    × experience_score
             + w_edu    × education_score
           )) × availability_multiplier × (0 if honeypot else 1)

When Claude score is NOT available (candidate outside top-K):
  final = (w_semantic × semantic_sim
           + w_skill   × skill_score
           + w_behav   × behavioral_score
           + w_exp     × experience_score
           + w_edu     × education_score
          ) × availability_multiplier
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from behavioral import compute_behavioral_score
from features import build_profile_text, extract_features
from honeypot import detect_honeypot
from jd import (
    CLAUDE_WEIGHT,
    FEATURE_WEIGHT,
    JD_TEXT,
    WEIGHTS,
)

SUBMISSION_ROWS = 100


# ─── embedding (local only — no network) ─────────────────────────────────────

def get_local_embedder():
    """Load sentence-transformer from disk cache. Raises if not installed."""
    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        print(f"[rank] Loading embedder: {model_name} …")
        t0 = time.time()
        model = SentenceTransformer(model_name)
        print(f"[rank] Embedder ready in {time.time()-t0:.1f}s")

        def encode(texts: list[str]) -> np.ndarray:
            return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        return encode
    except ImportError:
        raise RuntimeError(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers"
        )


# ─── load artifacts ───────────────────────────────────────────────────────────

def load_artifacts(artifacts_dir: Path) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    """
    Returns:
      faiss_data    — {"index": faiss.Index, "candidate_ids": list[str]}
      feat_map      — {candidate_id: feature_dict}
      claude_map    — {candidate_id: {"claude_score": float, "fit_summary": str, "is_disqualified": bool}}
    """
    faiss_path = artifacts_dir / "faiss_index.pkl"
    features_path = artifacts_dir / "candidate_features.jsonl"
    scores_path = artifacts_dir / "claude_scores.jsonl"
    bm25_path = artifacts_dir / "bm25_scores.jsonl"

    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}\nRun precompute.py first.")
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}\nRun precompute.py first.")

    print("[rank] Loading FAISS index ...")
    with open(faiss_path, "rb") as f:
        faiss_data = pickle.load(f)

    print("[rank] Loading candidate features ...")
    feat_map: dict[str, dict] = {}
    with open(features_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                feat_map[rec["candidate_id"]] = rec

    claude_map: dict[str, dict] = {}
    if scores_path.exists():
        print("[rank] Loading Claude scores ...")
        with open(scores_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    cid = rec.get("candidate_id")
                    if cid:
                        claude_map[cid] = rec
        print(f"[rank] Claude scores loaded for {len(claude_map):,} candidates.")
    else:
        print("[rank] No Claude scores found -- using feature-only scoring.")

    bm25_map: dict[str, float] = {}
    if bm25_path.exists():
        print("[rank] Loading BM25 scores ...")
        with open(bm25_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    bm25_map[rec["candidate_id"]] = rec["bm25_score_norm"]
        print(f"[rank] BM25 scores loaded for {len(bm25_map):,} candidates.")

    return faiss_data, feat_map, claude_map, bm25_map


# ─── scoring formula ──────────────────────────────────────────────────────────

def compute_final_score(
    feat: dict,
    semantic_sim: float,
    claude_rec: dict | None,
    bm25_score_norm: float = 0.0,
) -> float:
    """
    Combine Claude score (if available) with feature scores.
    Returns final score in [0, 1].
    """
    # Honeypot → instant zero
    if feat.get("is_honeypot"):
        return 0.0

    # Purely consulting disqualifier
    if feat.get("purely_consulting"):
        return 0.0

    avail = feat.get("availability_multiplier", 1.0)
    title_mult = feat.get("title_multiplier", 1.0)

    if claude_rec is not None:
        # Claude says disqualified
        if claude_rec.get("is_disqualified"):
            return 0.0

        cs = float(claude_rec.get("claude_score", 0.5))
        # BM25 + feature component (weights sum to 1.0)
        feature_component = (
            0.35 * feat.get("skill_score", 0.0)
            + 0.30 * feat.get("behavioral_score", 0.0)
            + 0.15 * feat.get("experience_score", 0.0)
            + 0.10 * feat.get("education_score", 0.0)
            + 0.10 * bm25_score_norm
        )
        combined = CLAUDE_WEIGHT * cs + FEATURE_WEIGHT * feature_component
    else:
        # No Claude score -- feature-only (semantic + BM25 + structured signals)
        combined = (
            WEIGHTS["semantic"]    * semantic_sim
            + WEIGHTS.get("bm25", 0.0) * bm25_score_norm
            + WEIGHTS["skill_match"] * feat.get("skill_score", 0.0)
            + WEIGHTS["behavioral"]  * feat.get("behavioral_score", 0.0)
            + WEIGHTS["experience"]  * feat.get("experience_score", 0.0)
            + WEIGHTS["education"]   * feat.get("education_score", 0.0)
        )

    return round(min(1.0, max(0.0, combined * avail * title_mult)), 6)


# ─── reasoning builder ────────────────────────────────────────────────────────

def build_reasoning(feat: dict, claude_rec: dict | None, final_score: float) -> str:
    parts = []

    if claude_rec and claude_rec.get("fit_summary"):
        parts.append(claude_rec["fit_summary"])

    yoe = feat.get("years_of_experience", 0)
    if yoe:
        parts.append(f"{yoe}y experience")

    sk = feat.get("skill_score", 0)
    if sk >= 0.7:
        parts.append("strong AI/ML skill match")
    elif sk >= 0.4:
        parts.append("partial skill match")
    else:
        parts.append("limited skill overlap")

    behav = feat.get("behavioral_score", 0)
    if behav >= 0.7:
        parts.append("highly responsive/active")

    if feat.get("purely_consulting"):
        parts.append("disqualified: all-consulting career")
    if feat.get("is_honeypot"):
        parts.append("disqualified: impossible profile")

    return "; ".join(parts) if parts else f"score={final_score:.3f}"


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    ap.add_argument("--out", default="submission.csv", help="Output CSV path")
    ap.add_argument("--artifacts", default="artifacts", help="Artifacts directory")
    ap.add_argument("--top-n", type=int, default=SUBMISSION_ROWS, help="Rows in submission")
    args = ap.parse_args()

    t_start = time.time()
    artifacts_dir = Path(args.artifacts)

    # ── Load artifacts ────────────────────────────────────────────────────────
    faiss_data, feat_map, claude_map, bm25_map = load_artifacts(artifacts_dir)
    index = faiss_data["index"]
    ordered_ids: list[str] = faiss_data["candidate_ids"]

    # ── Load or compute JD embedding ─────────────────────────────────────────
    jd_emb_path = artifacts_dir / "jd_embedding.npy"
    if jd_emb_path.exists():
        print("[rank] Loading pre-computed JD embedding …")
        jd_emb = np.load(str(jd_emb_path)).astype(np.float32)
    else:
        print("[rank] No saved JD embedding — computing locally (must match precompute model) …")
        encode = get_local_embedder()
        jd_emb = encode([JD_TEXT]).astype(np.float32)

    # ── FAISS search — retrieve top candidates ────────────────────────────────
    # Retrieve more than 100 so we have buffer after filtering honeypots
    retrieve_n = min(len(ordered_ids), max(2000, args.top_n * 20))
    print(f"[rank] FAISS search: top-{retrieve_n} …")
    sim_scores, indices = index.search(jd_emb, retrieve_n)

    candidates_to_score: list[tuple[str, float]] = []
    for idx, sim in zip(indices[0], sim_scores[0]):
        if idx < 0:
            continue
        cid = ordered_ids[idx]
        candidates_to_score.append((cid, float(sim)))

    print(f"[rank] {len(candidates_to_score):,} candidates to score …")

    # ── Score each candidate ──────────────────────────────────────────────────
    results: list[dict] = []
    for cid, sem_sim in candidates_to_score:
        feat = feat_map.get(cid)
        if feat is None:
            continue
        claude_rec = claude_map.get(cid)
        bm25_norm = bm25_map.get(cid, 0.0)
        score = compute_final_score(feat, sem_sim, claude_rec, bm25_norm)
        if score <= 0:
            continue  # skip honeypots/disqualified immediately
        results.append({
            "candidate_id": cid,
            "score": score,
            "semantic_sim": sem_sim,
            "reasoning": build_reasoning(feat, claude_rec, score),
        })

    # ── Also score candidates NOT returned by FAISS but in feat_map ──────────
    # (edge case: very large retrieve_n already covers most; skip if we have enough)
    if len(results) < args.top_n:
        faiss_seen = {cid for cid, _ in candidates_to_score}
        for cid, feat in feat_map.items():
            if cid in faiss_seen:
                continue
            if feat.get("is_honeypot") or feat.get("purely_consulting"):
                continue
            claude_rec = claude_map.get(cid)
            score = compute_final_score(feat, 0.0, claude_rec, bm25_map.get(cid, 0.0))
            if score > 0:
                results.append({
                    "candidate_id": cid,
                    "score": score,
                    "semantic_sim": 0.0,
                    "reasoning": build_reasoning(feat, claude_rec, score),
                })

    # ── Sort and take top-N ───────────────────────────────────────────────────
    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[: args.top_n]

    # ── Write submission.csv ──────────────────────────────────────────────────
    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for rank, rec in enumerate(top, start=1):
            writer.writerow({
                "candidate_id": rec["candidate_id"],
                "rank": rank,
                "score": round(rec["score"], 6),
                "reasoning": rec["reasoning"],
            })

    elapsed = time.time() - t_start
    print(f"[rank] Done in {elapsed:.1f}s -- {len(top)} candidates ranked.")


if __name__ == "__main__":
    main()
