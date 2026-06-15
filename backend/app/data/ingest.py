"""
Stage 0: Load + inspect the candidate dataset.
Run as: python -m app.data.ingest
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from app.data.normalize import normalize_dataframe


def load_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Load CSV or JSON dataset and return a raw DataFrame."""
    if path is None:
        from app.config import settings
        path = settings.dataset_path

    p = Path(path)
    if not p.is_absolute():
        # resolve relative to backend/
        base = Path(__file__).parent.parent.parent
        p = base / path

    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {p}. Run `python data/generate_dataset.py` first.")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(p, dtype=str).fillna("")
    elif suffix in (".json", ".jsonl"):
        df = pd.read_json(p, dtype=str)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}")

    logger.info(f"Loaded {len(df)} rows from {p}")
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print schema, row count, and sample rows."""
    print("\n" + "=" * 60)
    print(f"DATASET SCHEMA  ({len(df)} rows × {len(df.columns)} columns)")
    print("=" * 60)
    print(f"\nColumns: {list(df.columns)}\n")
    print("Dtypes:")
    print(df.dtypes.to_string())
    print(f"\nNull counts:")
    print(df.isnull().sum().to_string())
    print("\nSample rows (first 3):")
    for _, row in df.head(3).iterrows():
        print("-" * 40)
        for col, val in row.items():
            display = str(val)[:120] + ("…" if len(str(val)) > 120 else "")
            print(f"  {col}: {display}")
    print("=" * 60)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_dataset(path)
    inspect_dataset(df)

    profiles = normalize_dataframe(df)
    print(f"\nNormalized {len(profiles)} CandidateProfile objects successfully.")
    print("\nSample profile:")
    p = profiles[0]
    print(f"  candidate_id: {p.candidate_id}")
    print(f"  skills: {p.skills[:5]}{'...' if len(p.skills) > 5 else ''}")
    print(f"  full_text length: {len(p.full_text)} chars")
    print(f"  career_history entries: {len(p.career_history)}")
    print(f"  platform_activity: {json.dumps(p.platform_activity)[:120]}")
    print(f"  behavioral_signals: {json.dumps(p.behavioral_signals)[:120]}")
