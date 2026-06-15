"""
Stage 0: Light extraction — loads and normalizes all profiles.
This is the cheapest stage; no LLM calls.
"""
from __future__ import annotations
from functools import lru_cache

from loguru import logger

from app.data.ingest import load_dataset
from app.data.normalize import normalize_dataframe
from app.models import CandidateProfile


@lru_cache(maxsize=1)
def get_all_profiles(dataset_path: str | None = None) -> list[CandidateProfile]:
    """Load and normalize all profiles. Cached so it only runs once per process."""
    df = load_dataset(dataset_path)
    profiles = normalize_dataframe(df)
    logger.info(f"Stage 0: {len(profiles)} profiles normalized")
    return profiles
