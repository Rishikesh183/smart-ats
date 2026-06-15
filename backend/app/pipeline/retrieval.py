"""
Stage 1: Semantic retrieval + rescue band banding.
- Embeds all profiles with sentence-transformers
- Builds FAISS index
- Retrieves top-K for a JD query
- Bands results into advance / rescue / drop by percentile
"""
from __future__ import annotations
import hashlib
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from app.models import BandedCandidate, CandidateProfile, Mode

# Lazy imports for heavy deps
_embedder = None
_index = None
_indexed_profiles: list[CandidateProfile] = []


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        from app.config import settings
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def _cache_path(profiles: list[CandidateProfile]) -> Path:
    """Deterministic cache path based on profile content hash."""
    from app.config import settings
    h = hashlib.md5(
        (settings.embedding_model + "".join(p.candidate_id for p in profiles)).encode()
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

    embedder = _get_embedder()
    texts = [p.full_text for p in profiles]
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner product = cosine similarity (with normalized vecs)
    index.add(embeddings)

    _index = index
    _indexed_profiles = profiles

    with open(cache, "wb") as f:
        pickle.dump((index, profiles), f)
    logger.info(f"FAISS index built (dim={dim}) and cached")


def retrieve(
    job_description: str,
    profiles: list[CandidateProfile],
    top_k: Optional[int] = None,
    mode: Mode = "normal",
) -> list[BandedCandidate]:
    """
    Retrieve and band candidates for a JD.
    Returns a list of BandedCandidate in descending similarity order.
    """
    from app.config import settings

    if top_k is None:
        top_k = settings.retrieval_top_k

    # Rebuild index if needed
    global _index, _indexed_profiles
    if _index is None or _indexed_profiles != profiles:
        build_index(profiles)

    # Embed query
    embedder = _get_embedder()
    q_emb = embedder.encode([job_description], normalize_embeddings=True)[0].astype(np.float32)

    # Search
    k = min(top_k, len(profiles))
    scores, indices = _index.search(q_emb.reshape(1, -1), k)
    scores = scores[0]
    indices = indices[0]

    # Get thresholds for this mode
    advance_pct, rescue_pct = _get_thresholds(mode, settings)

    # Band by percentile of retrieved scores
    if len(scores) > 0:
        advance_threshold = float(np.percentile(scores, (1 - advance_pct) * 100))
        rescue_floor = float(np.percentile(scores, (1 - rescue_pct) * 100))
    else:
        advance_threshold = 0.0
        rescue_floor = 0.0

    logger.info(
        f"Stage 1 [{mode}]: retrieved={k} "
        f"advance_threshold={advance_threshold:.3f} "
        f"rescue_floor={rescue_floor:.3f}"
    )

    results: list[BandedCandidate] = []
    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        profile = _indexed_profiles[idx]
        score_f = float(score)

        if score_f >= advance_threshold:
            band = "advance"
        elif score_f >= rescue_floor:
            band = "rescue"
        else:
            band = "drop"

        results.append(BandedCandidate(
            profile=profile,
            similarity_score=score_f,
            band=band,
        ))

    advance_count = sum(1 for r in results if r.band == "advance")
    rescue_count = sum(1 for r in results if r.band == "rescue")
    drop_count = sum(1 for r in results if r.band == "drop")
    logger.info(f"  bands: advance={advance_count} rescue={rescue_count} drop={drop_count}")

    return results


def _get_thresholds(mode: Mode, settings) -> tuple[float, float]:
    """Return (advance_percentile, rescue_floor_percentile) for this mode."""
    if mode == "normal":
        return settings.advance_percentile_normal, settings.rescue_floor_normal
    elif mode == "high":
        return settings.advance_percentile_high, settings.rescue_floor_high
    else:  # extra_high
        return settings.advance_percentile_extra_high, settings.rescue_floor_extra_high
