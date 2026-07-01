"""
Offline pre-computation pipeline.
Run ONCE (can take 30-90 min for 100K candidates + LLM scoring).
Produces artifacts consumed by rank.py.

Usage:
  python precompute.py \
    --candidates "../[PUB] India_runs_data_and_ai_challenge/candidates.jsonl" \
    --out artifacts/ \
    [--top-k 500] [--batch-size 10] [--resume] [--skip-claude] [--no-bm25]

Output artifacts:
  artifacts/faiss_index.pkl           - FAISS index + candidate_id list
  artifacts/candidate_features.jsonl  - per-candidate feature dict (all 100K)
  artifacts/claude_scores.jsonl       - LLM scores for top-K (resumable)
  artifacts/bm25_scores.jsonl         - Normalized BM25 scores (all candidates)
  artifacts/feature_scores.csv        - Per-candidate features in CSV format
  artifacts/jd_embedding.npy          - JD embedding vector

Env vars:
  ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY
  EMBEDDING_PROVIDER=local | openai
  LLM_PROVIDER=claude | openrouter | openai
  OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
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
from jd import JD_TEXT, WEIGHTS


# ── embedding setup ───────────────────────────────────────────────────────────

def get_embedder():
    provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    if provider == "dummy":
        # Zero-cost test mode: random normalized 384-d vectors (no install needed)
        import hashlib
        def encode_dummy(texts, batch_size=64):
            vecs = []
            for t in texts:
                seed = int(hashlib.md5(t[:100].encode()).hexdigest(), 16) % (2**31)
                rng = np.random.RandomState(seed)
                v = rng.randn(384).astype(np.float32)
                vecs.append(v / np.linalg.norm(v))
            return np.array(vecs, dtype=np.float32)
        print("[precompute] DUMMY embedder active (test mode -- scores are meaningless)")
        return encode_dummy
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        def encode_openai(texts, batch_size=256):  # 256 * ~910 tok max = ~233k < 300k limit
            all_embs = []
            for i in range(0, len(texts), batch_size):
                # OpenAI rejects empty strings — replace with placeholder
                batch = [t if t and t.strip() else "[no profile]" for t in texts[i : i + batch_size]]
                for attempt in range(8):
                    try:
                        resp = client.embeddings.create(model=model, input=batch)
                        break
                    except Exception as _e:
                        if "429" in str(_e) or "rate_limit" in str(_e).lower():
                            wait = 15 * (attempt + 1)
                            print(f"[embed] Rate limited — waiting {wait}s (attempt {attempt+1}/8) ...")
                            time.sleep(wait)
                        else:
                            raise
                vecs = [e.embedding for e in resp.data]
                all_embs.extend(vecs)
            arr = np.array(all_embs, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return arr / norms

        return encode_openai
    elif provider == "ollama":
        import ollama
        model = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")

        def encode_ollama(texts, batch_size=64):
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                resp = ollama.embed(model=model, input=batch)
                vecs = resp["embeddings"]
                all_embs.extend(vecs)
            arr = np.array(all_embs, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return arr / norms

        return encode_ollama
    else:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        print(f"[precompute] Loading sentence-transformer: {model_name} ...")
        model = SentenceTransformer(model_name)

        def encode_local(texts, batch_size=64):
            return model.encode(
                texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
            )

        return encode_local


# ── LLM scoring ───────────────────────────────────────────────────────────────

def build_llm_prompt(jd, candidates):
    cand_blocks = []
    for c in candidates:
        cid = c.get("candidate_id", "")
        text = c.get("profile_text", "")[:1200]
        cand_blocks.append("=== " + cid + " ===\n" + text)
    cands_text = "\n\n".join(cand_blocks)
    return (
        "You are an expert technical recruiter. Rate each candidate for the following job description.\n\n"
        "JOB DESCRIPTION:\n" + jd + "\n\n"
        "CANDIDATES:\n" + cands_text + "\n\n"
        "For each candidate output a JSON object with:\n"
        "  candidate_id: str\n"
        "  claude_score: float 0-1 (0=completely unfit, 1=perfect fit)\n"
        "  fit_summary: str (1-2 sentences, evidence-grounded)\n"
        "  is_disqualified: bool (true if explicit disqualifier applies)\n\n"
        "Return a JSON array, one object per candidate, in the same order.\n"
        "Return ONLY the JSON array, no markdown fences."
    )


def _parse_llm_response(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def score_with_llm(candidates_batch, jd, model="claude-3-5-haiku-20241022", max_retries=3):
    provider = os.getenv("LLM_PROVIDER", "claude").lower()

    cand_blocks = []
    for c in candidates_batch:
        cid = c.get("candidate_id", "")
        text = c.get("profile_text", "")[:1200]
        cand_blocks.append("=== " + cid + " ===\n" + text)
    cands_text = "\n\n".join(cand_blocks)

    prompt = (
        "You are an expert technical recruiter. "
        "Rate each candidate for this Senior AI Engineer role.\n\n"
        "JOB DESCRIPTION:\n" + jd[:2000] + "\n\n"
        "CANDIDATES:\n" + cands_text + "\n\n"
        "Output a JSON array, one object per candidate:\n"
        "[{candidate_id, claude_score 0-1, fit_summary 1-2 sentences, is_disqualified bool}]\n"
        "Return ONLY the JSON array, no markdown."
    )

    for attempt in range(max_retries):
        try:
            if provider == "openrouter":
                from openai import OpenAI
                client = OpenAI(
                    api_key=os.getenv("OPENROUTER_API_KEY", ""),
                    base_url="https://openrouter.ai/api/v1",
                )
                or_model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
                resp = client.chat.completions.create(
                    model=or_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                )
                raw = resp.choices[0].message.content or ""
            elif provider == "openai":
                from openai import OpenAI
                client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY", ""),
                    base_url=os.getenv("OPENAI_BASE_URL") or None,
                )
                resp = client.chat.completions.create(
                    model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                )
                raw = resp.choices[0].message.content or ""
            else:
                import anthropic
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
                msg = client.messages.create(
                    model=model,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = msg.content[0].text

            results = _parse_llm_response(raw)
            if isinstance(results, list):
                # Sanity-clamp claude_score to [0,1]. Smaller local models (e.g. qwen2.5:3b)
                # sometimes ignore the "0-1 float" instruction and answer on a 0-10 scale
                # instead, which silently ceilings to a false 1.0 downstream. Rescale any
                # out-of-range value before it ever hits disk.
                for rec in results:
                    s = rec.get("claude_score")
                    if isinstance(s, (int, float)):
                        if s > 1:
                            s = s / 10.0
                        rec["claude_score"] = round(min(1.0, max(0.0, s)), 4)
                    else:
                        rec["claude_score"] = 0.5
                return results

        except Exception as e:
            err = str(e)
            print(f"  [llm] attempt {attempt+1} failed: {err[:120]}")
            if attempt < max_retries - 1:
                if "429" in err or "rate_limit" in err.lower():
                    wait = 10 * (2 ** attempt)
                elif "503" in err or "502" in err or "overloaded" in err.lower():
                    wait = 5 * (2 ** attempt)
                else:
                    wait = 2 ** attempt
                print(f"  [llm] waiting {wait}s ...")
                time.sleep(wait)

    return [
        {"candidate_id": c["candidate_id"], "claude_score": 0.5,
         "fit_summary": "Error", "is_disqualified": False}
        for c in candidates_batch
    ]


score_with_claude = score_with_llm  # backwards compat


# ── FAISS index ───────────────────────────────────────────────────────────────

def build_faiss_index(embeddings):
    import faiss  # type: ignore
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# ── main pipeline ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--top-k", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--model", default="claude-3-5-haiku-20241022")
    ap.add_argument("--skip-claude", action="store_true",
                    help="Skip LLM scoring (feature-only mode)")
    ap.add_argument("--no-bm25", action="store_true",
                    help="Skip BM25 hybrid retrieval (FAISS-only)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Randomly sample top_k from top_k*10 FAISS pool")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_path = out_dir / "candidate_features.jsonl"
    faiss_path = out_dir / "faiss_index.pkl"
    scores_path = out_dir / "claude_scores.jsonl"
    bm25_scores_path = out_dir / "bm25_scores.jsonl"
    feature_csv_path = out_dir / "feature_scores.csv"

    # ── Step 1: Load + extract features ──────────────────────────────────────
    if features_path.exists() and args.resume:
        print(f"[precompute] Resuming -- loading features from {features_path}")
        all_features = []
        with open(features_path) as f:
            for line in f:
                all_features.append(json.loads(line))
    else:
        print(f"[precompute] Loading candidates from {args.candidates} ...")
        raw_candidates = []
        with open(args.candidates) as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_candidates.append(json.loads(line))
        print(f"[precompute] {len(raw_candidates):,} candidates loaded.")

        print("[precompute] Extracting features + honeypot detection ...")
        all_features = []
        honeypot_count = 0
        for cand in raw_candidates:
            feat = extract_features(cand)
            is_hp, reasons = detect_honeypot(cand)
            feat["is_honeypot"] = is_hp
            feat["honeypot_reasons"] = reasons
            signals = cand.get("redrob_signals", {})
            feat["behavioral_score"] = compute_behavioral_score(signals)
            all_features.append(feat)
            if is_hp:
                honeypot_count += 1

        print(f"[precompute] Honeypots detected: {honeypot_count}")

        with open(features_path, "w") as f:
            for feat in all_features:
                row = {k: v for k, v in feat.items() if k != "profile_text"}
                f.write(json.dumps(row) + "\n")
        print(f"[precompute] Features saved -> {features_path}")

        # Export feature CSV for analysis
        csv_fields = [
            "candidate_id", "skill_score", "behavioral_score",
            "experience_score", "education_score", "title_multiplier",
            "availability_multiplier", "is_honeypot", "purely_consulting",
            "years_of_experience", "location",
        ]
        with open(feature_csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for feat in all_features:
                writer.writerow({k: feat.get(k, "") for k in csv_fields})
        print(f"[precompute] Feature CSV saved -> {feature_csv_path}")

    feat_map = {f["candidate_id"]: f for f in all_features}

    # ── Step 2: Build FAISS index ─────────────────────────────────────────────
    if faiss_path.exists() and args.resume:
        print(f"[precompute] Resuming -- FAISS index at {faiss_path}")
        with open(faiss_path, "rb") as f:
            faiss_data = pickle.load(f)
        index = faiss_data["index"]
        ordered_ids = faiss_data["candidate_ids"]
    else:
        encode = get_embedder()
        valid_features = [f for f in all_features if not f.get("is_honeypot")]
        print(f"[precompute] Embedding {len(valid_features):,} valid candidates ...")

        ordered_ids = [f["candidate_id"] for f in valid_features]
        texts = [f.get("profile_text", f.get("headline", "")) for f in valid_features]

        # In resume mode, profile_text is stripped from saved features.
        # Re-read candidates.jsonl to rebuild profile texts when needed.
        if not any(t.strip() for t in texts[:20]):
            print("[precompute] profile_text missing from features (resume mode) -- reloading from candidates file ...")
            text_lookup: dict[str, str] = {}
            with open(args.candidates) as _cf:
                for _line in _cf:
                    _line = _line.strip()
                    if not _line:
                        continue
                    _cand = json.loads(_line)
                    text_lookup[_cand["candidate_id"]] = build_profile_text(_cand)
            texts = [text_lookup.get(cid, "[no profile]") for cid in ordered_ids]
            print(f"[precompute] Texts reloaded for {sum(1 for t in texts if t.strip()):,} candidates.")

        chunk = 4096
        all_embs = []
        for i in range(0, len(texts), chunk):
            print(f"  chunk {i//chunk + 1}/{(len(texts)-1)//chunk + 1} ...")
            embs = encode(texts[i : i + chunk])
            all_embs.append(embs)
        embeddings = np.vstack(all_embs).astype(np.float32)

        print(f"[precompute] Building FAISS index (dim={embeddings.shape[1]}) ...")
        index = build_faiss_index(embeddings)

        with open(faiss_path, "wb") as f:
            pickle.dump({"index": index, "candidate_ids": ordered_ids}, f)
        print(f"[precompute] FAISS index saved -> {faiss_path}")

    # ── Step 2.5: BM25 scoring ────────────────────────────────────────────────
    bm25_sorted_ids = []

    if not args.no_bm25:
        if bm25_scores_path.exists() and args.resume:
            print(f"[precompute] Resuming -- loading BM25 scores from {bm25_scores_path}")
            bm25_score_map = {}
            with open(bm25_scores_path) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    bm25_score_map[rec["candidate_id"]] = rec["bm25_score_norm"]
            bm25_sorted_ids = sorted(bm25_score_map, key=lambda x: bm25_score_map[x], reverse=True)
            print(f"[precompute] BM25 scores loaded for {len(bm25_score_map):,} candidates.")
        else:
            try:
                from retrieval import BM25Retriever  # type: ignore

                has_text = any("profile_text" in f for f in all_features[:10])
                if not has_text:
                    print("[precompute] Re-reading profile texts for BM25 ...")
                    with open(args.candidates) as cf:
                        for line in cf:
                            line = line.strip()
                            if not line:
                                continue
                            cand = json.loads(line)
                            cid = cand.get("candidate_id", "")
                            if cid in feat_map:
                                feat_map[cid]["profile_text"] = build_profile_text(cand)

                bm25_profiles = [
                    {"candidate_id": f["candidate_id"],
                     "profile_text": f.get("profile_text", f.get("headline", ""))}
                    for f in all_features
                    if not f.get("is_honeypot")
                    and (f.get("profile_text") or f.get("headline"))
                ]

                print(f"[precompute] Building BM25 index over {len(bm25_profiles):,} profiles ...")
                retriever = BM25Retriever()
                retriever.build(bm25_profiles)

                print("[precompute] Scoring all candidates with BM25 ...")
                raw_scores = retriever.score_all(JD_TEXT)

                max_s = max(raw_scores.values()) if raw_scores else 1.0
                if max_s <= 0:
                    max_s = 1.0
                bm25_score_map = {cid: round(s / max_s, 6) for cid, s in raw_scores.items()}

                with open(bm25_scores_path, "w") as sf:
                    for cid, score in bm25_score_map.items():
                        sf.write(json.dumps({"candidate_id": cid, "bm25_score_norm": score}) + "\n")
                print(f"[precompute] BM25 scores saved -> {bm25_scores_path}")

                bm25_sorted_ids = sorted(bm25_score_map, key=lambda x: bm25_score_map[x], reverse=True)

            except ImportError:
                print("[precompute] WARNING: rank_bm25 not installed -- skipping BM25.")
                print("  Install: pip install rank-bm25")

    # ── Step 3: JD embedding -> retrieve top-K ────────────────────────────────
    jd_emb_path = out_dir / "jd_embedding.npy"
    print("[precompute] Embedding JD ...")
    encode = get_embedder()
    jd_emb = encode([JD_TEXT]).astype(np.float32)
    np.save(str(jd_emb_path), jd_emb)
    print(f"[precompute] JD embedding saved -> {jd_emb_path}")

    pool_size = min(len(ordered_ids), args.top_k * 2)
    _, indices = index.search(jd_emb, pool_size)
    faiss_ids = [ordered_ids[i] for i in indices[0] if i >= 0]

    if bm25_sorted_ids:
        from retrieval import hybrid_retrieve  # type: ignore
        bm25_pool = bm25_sorted_ids[:pool_size]
        top_k_ids = hybrid_retrieve(bm25_pool, faiss_ids, top_k=args.top_k)
        print(f"[precompute] Hybrid BM25+FAISS -> {len(top_k_ids)} candidates.")
    elif args.seed is not None:
        import random
        rng = random.Random(args.seed)
        pool_ids = list(faiss_ids)
        rng.shuffle(pool_ids)
        top_k_ids = pool_ids[:args.top_k]
        print(f"[precompute] Seed={args.seed}: sampled {len(top_k_ids)} from FAISS pool.")
    else:
        top_k_ids = faiss_ids[:args.top_k]
        print(f"[precompute] FAISS top-{len(top_k_ids)} candidates retrieved.")

    # ── Step 4: LLM scoring (resumable) ──────────────────────────────────────
    if args.skip_claude:
        print("[precompute] --skip-claude: skipping LLM scoring (feature-only mode).")
        print(f"[precompute] Done! Artifacts in {out_dir}/")
        print("  * candidate_features.jsonl")
        print("  * faiss_index.pkl")
        if not args.no_bm25 and bm25_sorted_ids:
            print("  * bm25_scores.jsonl")
        print("  * feature_scores.csv")
        return

    already_scored: set = set()
    if scores_path.exists() and args.resume:
        with open(scores_path) as f:
            for line in f:
                rec = json.loads(line)
                already_scored.add(rec["candidate_id"])
        print(f"[precompute] Resuming LLM scoring -- {len(already_scored)} already done.")

    to_score = [cid for cid in top_k_ids if cid not in already_scored]
    print(f"[precompute] Scoring {len(to_score)} candidates with LLM ({os.getenv('LLM_PROVIDER', 'claude')}) ...")

    if to_score:
        sample_feat = feat_map.get(to_score[0], {})
        if "profile_text" not in sample_feat:
            print("[precompute] Re-loading profile texts for LLM prompts ...")
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

            results = score_with_llm(batch_feats, JD_TEXT, model=args.model)

            for rec in results:
                score_file.write(json.dumps(rec) + "\n")
            score_file.flush()

            scored_count += len(results)
            if (i // batch) % 10 == 0:
                print(f"  [{scored_count}/{len(to_score)}] scored ...")

            time.sleep(0.5)

    print(f"[precompute] Done! Artifacts in {out_dir}/")
    print("  * candidate_features.jsonl")
    print("  * faiss_index.pkl")
    print("  * claude_scores.jsonl")
    if not args.no_bm25 and bm25_sorted_ids:
        print("  * bm25_scores.jsonl")
    print("  * feature_scores.csv")


if __name__ == "__main__":
    main()
