"""
Stage 3: Evidence-grounded LLM scoring.
Every score must cite evidence spans from the candidate profile.
"""
from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from app.llm.client import get_client
from app.llm.prompts import SCORER_SYSTEM, SCORER_USER
from app.models import CandidateProfile, CandidateScore, ParameterScore, Rubric


def _score_one(profile: CandidateProfile, rubric: Rubric) -> CandidateScore:
    """Score a single candidate against the rubric."""
    client = get_client()

    prompt = SCORER_USER.format(
        rubric_json=rubric.model_dump_json(indent=2),
        candidate_id=profile.candidate_id,
        profile_text=profile.full_text,
    )

    try:
        raw = client.complete(
            messages=[{"role": "user", "content": prompt}],
            schema={},
            fast=False,
            system=SCORER_SYSTEM,
        )
    except Exception as e:
        logger.error(f"Scorer LLM error for {profile.candidate_id}: {e}")
        # Return a minimal failed score so we don't crash the pipeline
        return CandidateScore(
            candidate_id=profile.candidate_id,
            passed_gates=False,
            gate_failures=["llm_error"],
            total_score=0.0,
            summary=f"LLM scoring failed: {e}",
        )

    # Parse parameter scores
    param_scores: list[ParameterScore] = []
    for ps_raw in raw.get("parameter_scores", []):
        # Look up weight from rubric in case LLM got it wrong
        rubric_param = next(
            (p for p in rubric.parameters if p.id == ps_raw.get("parameter_id")),
            None
        )
        weight = rubric_param.weight if rubric_param else float(ps_raw.get("weight", 0))

        # Ensure evidence list is non-empty — never allow evidence-free scores
        evidence = ps_raw.get("evidence", [])
        if not evidence:
            evidence = ["[No direct evidence found in profile]"]
            ps_raw["confidence"] = "low"

        param_scores.append(ParameterScore(
            parameter_id=ps_raw.get("parameter_id", "unknown"),
            score=float(ps_raw.get("score", 0)),
            max=float(ps_raw.get("max", 10)),
            weight=weight,
            evidence=evidence,
            reasoning=ps_raw.get("reasoning", ""),
            confidence=ps_raw.get("confidence", "low"),
        ))

    score = CandidateScore(
        candidate_id=raw.get("candidate_id", profile.candidate_id),
        passed_gates=raw.get("passed_gates", True),
        gate_failures=raw.get("gate_failures", []),
        parameter_scores=param_scores,
        summary=raw.get("summary", ""),
    )
    # total_score is computed by the model validator
    logger.debug(f"  Scored {profile.candidate_id}: {score.total_score:.1f}/100 gates={'OK' if score.passed_gates else 'FAIL'}")
    return score


def score_candidates(
    profiles: list[CandidateProfile],
    rubric: Rubric,
    max_workers: int = 4,
) -> list[CandidateScore]:
    """
    Score all profiles concurrently.
    Returns scores in the same order as input profiles.
    """
    logger.info(f"Stage 3: Scoring {len(profiles)} candidates (workers={max_workers})")
    results: dict[str, CandidateScore] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_score_one, profile, rubric): profile.candidate_id
            for profile in profiles
        }
        for future in as_completed(futures):
            cid = futures[future]
            try:
                score = future.result()
                results[cid] = score
            except Exception as e:
                logger.error(f"Unhandled error scoring {cid}: {e}")

    # Return in original order
    return [results[p.candidate_id] for p in profiles if p.candidate_id in results]
