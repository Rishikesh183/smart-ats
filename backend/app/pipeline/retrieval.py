"""
Stage 1: Semantic retrieval + rescue band banding.

Embedding is handled by app.llm.embedder (pluggable: local or API).
FAISS index is cached to disk — embeddings are computed exactly once
per candidate set regardless of which backend is used.
"""
from __future__ import annotations
import hashlib
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from app.models import BandedCandidate, CandidateProfile, Mode

_index = None
_indexed_profiles: list[CandidateProfile] = []


def _cache_path(profiles: list[CandidateProfile]) -> Path:
    """Deterministic cache key: embedding model name + ordered candidate IDs."""
    from app.llm.embedder import get_embedder
    h = hashlib.md5(
        (get_embedder().model_name + "".join(p.candidate_id for p in profiles)).encode()
    ).hexdigest()[:12]
    return Path(f"/tmp/ats_faiss_{h}.pkl")


def build_index(profiles: list[CandidateProfile]) -> None:
    """Build (or load cached) FAISS index for all profiles."""
    global _index, _indexed_profiles

    cache = _cache_path(profiles)
    if cache.exists():
        logger.info(f"Loading FAISS index from cache: {cache}")
        with open(cache, "rb") as f:
            _index, _indexed_profiles = pickle.load(f)
        return

    logger.info(f"Building FAISS index for {len(profiles)} profiles")
    import faiss
    from app.llm.embedder import get_embedder

    embedder = get_embedder()
    texts = [p.full_text for p in profiles]
    embeddings = embedder.encode(texts, normalize=True)           # (N, dim) float32

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product = cosine on normalized vecs
    index.add(embeddings)

    _index = index
    _indexed_profiles = profiles

    with open(cache, "wb") as f:
        pickle.dump((index, profiles), f)
    logger.info(f"FAISS index built (dim={dim}, backend={embedder.model_name}) and cached")


def retrieve(
    job_description: str,
    profiles: list[CandidateProfile],
    top_k: Optional[int] = None,
    mode: Mode = "normal",
) -> list[BandedCandidate]:
    """
    Retrieve and band candidates for a JD.
    Returns BandedCandidates in descending similarity order.
    """
    from app.config import settings
    from app.llm.embedder import get_embedder

    if top_k is None:
        top_k = settings.retrieval_top_k

    global _index, _indexed_profiles
    if _index is None or _indexed_profiles != profiles:
        build_index(profiles)

    embedder = get_embedder()
    q_emb = embedder.encode([job_description], normalize=True)[0]   # (dim,)

    k = min(top_k, len(profiles))
    scores, indices = _index.search(q_emb.reshape(1, -1), k)
    scores, indices = scores[0], indices[0]

    advance_pct, rescue_pct = _get_thresholds(mode, settings)
    if len(scores):
        advance_threshold = float(np.percentile(scores, (1 - advance_pct) * 100))
        rescue_floor      = float(np.percentile(scores, (1 - rescue_pct)  * 100))
    else:
        advance_threshold = rescue_floor = 0.0

    logger.info(
        f"Stage 1 [{mode}]: retrieved={k} "
        f"advance≥{advance_threshold:.3f} rescue≥{rescue_floor:.3f}"
    )

    results: list[BandedCandidate] = []
    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        score_f = float(score)
        band = (
            "advance" if score_f >= advance_threshold else
            "rescue"  if score_f >= rescue_floor      else
            "drop"
        )
        results.append(BandedCandidate(
            profile=_indexed_profiles[idx],
            similarity_score=score_f,
            band=band,
        ))

    logger.info(
        f"  bands: advance={sum(1 for r in results if r.band=='advance')} "
        f"rescue={sum(1 for r in results if r.band=='rescue')} "
        f"drop={sum(1 for r in results if r.band=='drop')}"
    )
    return results


def _get_thresholds(mode: Mode, settings) -> tuple[float, float]:
    if mode == "normal":
        return settings.advance_percentile_normal, settings.rescue_floor_normal
    elif mode == "high":
        return settings.advance_percentile_high, settings.rescue_floor_high
    else:
        return settings.advance_percentile_extra_high, settings.rescue_floor_extra_high
