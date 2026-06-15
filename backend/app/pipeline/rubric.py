"""
Pass 1: Job Description → Rubric
Turns a JD into a structured, editable rubric.json.
"""
from __future__ import annotations
import json
from loguru import logger

from app.llm.client import get_client
from app.llm.prompts import RUBRIC_SYSTEM, RUBRIC_USER
from app.models import Rubric


def build_rubric(job_description: str) -> Rubric:
    """
    Call the LLM to parse a JD into a structured Rubric.
    Uses the fast model (cheaper) since this is a single pass.
    """
    logger.info("Pass 1: Building rubric from JD")
    client = get_client()

    prompt = RUBRIC_USER.format(job_description=job_description)
    raw = client.complete(
        messages=[{"role": "user", "content": prompt}],
        schema={},       # signals: parse JSON from response
        fast=False,      # use the scoring model for quality
        system=RUBRIC_SYSTEM,
    )

    # Validate and construct
    try:
        rubric = Rubric.model_validate(raw)
    except Exception as e:
        logger.error(f"Rubric validation failed: {e}\nRaw: {json.dumps(raw, indent=2)[:500]}")
        # Attempt repair: normalize weights to sum to 1.0
        if isinstance(raw, dict) and "parameters" in raw:
            params = raw["parameters"]
            total = sum(float(p.get("weight", 0)) for p in params)
            if total > 0 and abs(total - 1.0) > 0.001:
                logger.warning(f"Normalizing weights from {total:.3f} to 1.0")
                for p in params:
                    p["weight"] = round(float(p.get("weight", 0)) / total, 4)
                # Fix rounding error on last param
                diff = 1.0 - sum(p["weight"] for p in params)
                params[-1]["weight"] = round(params[-1]["weight"] + diff, 4)
            rubric = Rubric.model_validate(raw)
        else:
            raise

    logger.info(
        f"Rubric built: {rubric.role_title!r} | "
        f"{len(rubric.parameters)} params | "
        f"{len(rubric.hard_gates)} hard gates"
    )
    return rubric
