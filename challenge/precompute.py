"""
Offline pre-computation pipeline.
Run ONCE (can take 30-90 min for 100K candidates + Claude scoring).
Produces artifacts consumed by rank.py.

Usage:
  python precompute.py \
    --candidates "../[PUB] India_runs_data_and_ai_challenge/candidates.jsonl" \
    --out artifacts/ \
    [--top-k 500] \
    [--batch-size 5] \
    [--resume]

Output artifacts:
  artifacts/faiss_index.pkl         — FAISS index + ordered candidate_id list
  artifacts/candidate_features.jsonl — per-candidate feature dict (all 100K)
  artifacts/claude_scores.jsonl      — Claude scores for top-K (resumable)

Env vars needed (set in .env or shell):
  ANTHROPIC_API_KEY=...
  EMBEDDING_PROVIDER=local | openai
  OPENAI_API_KEY=...            (if openai)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from the challenge/ dir
sys.path.insert(0, str(Path(__file__).parent))

from behavioral import compute_behavioral_score
from features import build_profile_text, extract_features
from honeypot import detect_honeypot
from jd import JD_TEXT, WEIGHTS


# ─── embedding setup ─────────────────────────────────────────────────────────

def get_embedder():
    """Return whichever embedder is configured (local or OpenAI)."""
    provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        def encode_openai(texts: list[str], batch_size: int = 64) -> np.ndarray:
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                resp = client.embeddings.create(model=model, input=batch)
                vecs = [e.embedding for e in resp.data]
                all_embs.extend(vecs)
            arr = np.array(all_embs, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return arr / norms

        return encode_openai
    else:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        print(f"[precompute] Loading sentence-transformer: {model_name} …")
        model = SentenceTransformer(model_name)

        def encode_local(texts: list[str], batch_size: int = 64) -> np.ndarray:
            return model.encode(
                texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
            )

        return encode_local


# ─── Claude scoring ───────────────────────────────────────────────────────────

def build_claude_prompt(jd: str, candidates: list[dict]) -> str:
    cand_blocks = []
    for c in candidates:
        cid = c.get("candidate_id", "")
        text = c.get("profile_text", "")[:1200]  # cap to avoid token overflow
        cand_blocks.append(f"=== {cid} ===\n{text}")
    cands_text = "\n\n".join(cand_blocks)
    return f"""You are an expert technical recruiter. Rate each candidate for the following job description.

JOB DESCRIPTION:
{jd}

CANDIDATES:
{cands_text}

For each candidate output a JSON object with:
  "candidate_id": str
  "claude_score": float 0-1 (0=completely unfit, 1=perfect fit)
  "fit_summary": str (1-2 sentences, evidence-grounded)
  "is_disqualified": bool (true if explicit disqualifier applies)

Return a JSON array, one object per candidate, in the same order.
Return ONLY the JSON array, no markdown fences."""


def score_with_claude(
    candidates_batch: list[dict],
    jd: str,
    model: str = "claude-3-5-sonnet-20241022",
    max_retries: int = 3,
) -> list[dict]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    prompt = build_claude_prompt(jd, candidates_batch)

    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            # strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            results = json.loads(raw)
            if isinstance(results, list):
                return results
        except (json.JSONDecodeError, IndexError, Exception) as e:
            print(f"  [claude] attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    # fallback: return neutral scores
    return [
        {"candidate_id": c["candidate_id"], "claude_score": 0.5, "fit_summary": "Error", "is_disqualified": False}
        for c in candidates_batch
    ]


# ─── FAISS index ──────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> "faiss.Index":
    import faiss  # type: ignore
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# ─── main pipeline ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    ap.add_argument("--out", default="artifacts", help="Output directory")
    ap.add_argument("--top-k", type=int, default=500, help="Top-K for Claude scoring")
    ap.add_argument("--batch-size", type=int, default=5, help="Candidates per Claude call")
    ap.add_argument("--resume", action="store_true", help="Skip already-scored candidates")
    ap.add_argument("--model", default="claude-3-5-sonnet-20241022")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_path = out_dir / "candidate_features.jsonl"
    faiss_path = out_dir / "faiss_index.pkl"
    scores_path = out_dir / "claude_scores.jsonl"

    # ── Step 1: Load + extract features ──────────────────────────────────────
    if features_path.exists() and args.resume:
        print(f"[precompute] Resuming — loading features from {features_path}")
        all_features = []
        with open(features_path) as f:
            for line in f:
                all_features.append(json.loads(line))
    else:
        print(f"[precompute] Loading candidates from {args.candidates} …")
        raw_candidates = []
        with open(args.candidates) as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_candidates.append(json.loads(line))
        print(f"[precompute] {len(raw_candidates):,} candidates loaded.")

        print("[precompute] Extracting features + honeypot detection …")
        all_features = []
        honeypot_count = 0
        for cand in raw_candidates:
            feat = extract_features(cand)
            is_hp, reasons = detect_honeypot(cand)
            feat["is_honeypot"] = is_hp
            feat["honeypot_reasons"] = reasons

            # Behavioral score (doesn't need the full cand, just signals)
            signals = cand.get("redrob_signals", {})
            feat["behavioral_score"] = compute_behavioral_score(signals)

            all_features.append(feat)
            if is_hp:
                honeypot_count += 1

        print(f"[precompute] Honeypots detected: {honeypot_count}")

        # Save features
        with open(features_path, "w") as f:
            for feat in all_features:
                # don't store profile_text in features file (it's huge); it's in the embeddings
                row = {k: v for k, v in feat.items() if k != "profile_text"}
                f.write(json.dumps(row) + "\n")
        print(f"[precompute] Features saved → {features_path}")

    # ── Step 2: Build FAISS index ─────────────────────────────────────────────
    if faiss_path.exists() and args.resume:
        print(f"[precompute] Resuming — FAISS index already exists at {faiss_path}")
        with open(faiss_path, "rb") as f:
            faiss_data = pickle.load(f)
        index = faiss_data["index"]
        ordered_ids = faiss_data["candidate_ids"]
    else:
        encode = get_embedder()

        # Filter out honeypots before embedding (saves time + avoids noise)
        valid_features = [f for f in all_features if not f.get("is_honeypot")]
        print(f"[precompute] Embedding {len(valid_features):,} valid candidates …")

        texts = [f.get("profile_text", f.get("headline", "")) for f in valid_features]
        ordered_ids = [f["candidate_id"] for f in valid_features]

        # Embed in chunks to save memory
        chunk = 4096
        all_embs = []
        for i in range(0, len(texts), chunk):
            print(f"  embedding chunk {i//chunk + 1}/{(len(texts)-1)//chunk + 1} …")
            embs = encode(texts[i : i + chunk])
            all_embs.append(embs)
        embeddings = np.vstack(all_embs).astype(np.float32)

        print(f"[precompute] Building FAISS index (dim={embeddings.shape[1]}) …")
        index = build_faiss_index(embeddings)

        with open(faiss_path, "wb") as f:
            pickle.dump({"index": index, "candidate_ids": ordered_ids}, f)
        print(f"[precompute] FAISS index saved → {faiss_path}")

    # ── Step 3: JD embedding → retrieve top-K ────────────────────────────────
    print(f"[precompute] Retrieving top-{args.top_k} candidates for JD …")
    encode = get_embedder()
    jd_emb = encode([JD_TEXT]).astype(np.float32)

    scores_arr, indices = index.search(jd_emb, args.top_k)
    top_k_ids = [ordered_ids[i] for i in indices[0]]
    print(f"[precompute] Top-{args.top_k} candidates retrieved.")

    # ── Step 4: Claude scoring (resumable) ───────────────────────────────────
    already_scored: set[str] = set()
    if scores_path.exists() and args.resume:
        with open(scores_path) as f:
            for line in f:
                rec = json.loads(line)
                already_scored.add(rec["candidate_id"])
        print(f"[precompute] Resuming Claude scoring — {len(already_scored)} already done.")

    # Build feature map for quick lookup
    feat_map = {f["candidate_id"]: f for f in all_features}

    to_score = [cid for cid in top_k_ids if cid not in already_scored]
    print(f"[precompute] Scoring {len(to_score)} candidates with Claude …")

    # We need profile_text for the prompt — it may not be in features file if resume
    # Re-load from source if needed
    if to_score:
        # Check if profile_text is present in feat_map
        sample_feat = feat_map.get(to_score[0], {})
        if "profile_text" not in sample_feat:
            print("[precompute] Re-loading profile texts for Claude prompts …")
            with open(args.candidates) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cand = json.loads(line)
                    cid = cand.get("candidate_id", "")
                    if cid in feat_map:
                        feat_map[cid]["profile_text"] = build_profile_text(cand)

    batch = args.batch_size
    scored_count = 0
    with open(scores_path, "a") as score_file:
        for i in range(0, len(to_score), batch):
            batch_ids = to_score[i : i + batch]
            batch_feats = [feat_map[cid] for cid in batch_ids if cid in feat_map]

            results = score_with_claude(batch_feats, JD_TEXT, model=args.model)

            for rec in results:
                score_file.write(json.dumps(rec) + "\n")
            score_file.flush()

            scored_count += len(results)
            if (i // batch) % 10 == 0:
                print(f"  [{scored_count}/{len(to_score)}] scored …")

            # Rate-limit: Claude Sonnet burst limit
            time.sleep(0.5)

    print(f"[precompute] Done! All artifacts saved to {out_dir}/")
    print("  • candidate_features.jsonl")
    print("  • faiss_index.pkl")
    print("  • claude_scores.jsonl")


if __name__ == "__main__":
    main()
