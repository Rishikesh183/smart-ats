"""
Stage 4a: Critic agent — re-scores rescue-band candidates.
Promotes candidates whose true score clears the shortlist bar.
"""
from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from app.llm.client import get_client
from app.llm.prompts import CRITIC_SYSTEM, CRITIC_USER
from app.models import (
    BandedCandidate,
    CandidateScore,
    ParameterScore,
    RescuedEntry,
    RescuedReport,
    Rubric,
)


def _critic_score_one(banded: BandedCandidate, rubric: Rubric) -> tuple[CandidateScore, str]:
    """Run critic scoring on a single rescue-band candidate. Returns (score, why_rescued)."""
    client = get_client()
    profile = banded.profile

    prompt = CRITIC_USER.format(
        rubric_json=rubric.model_dump_json(indent=2),
        candidate_id=profile.candidate_id,
        profile_text=profile.full_text,
    )

    try:
        raw = client.complete(
            messages=[{"role": "user", "content": prompt}],
            schema={},
            fast=False,
            system=CRITIC_SYSTEM,
        )
    except Exception as e:
        logger.error(f"Critic LLM error for {profile.candidate_id}: {e}")
        empty = CandidateScore(
            candidate_id=profile.candidate_id,
            passed_gates=False,
            gate_failures=["llm_error"],
            summary=str(e),
        )
        return empty, ""

    param_scores: list[ParameterScore] = []
    for ps_raw in raw.get("parameter_scores", []):
        rubric_param = next(
            (p for p in rubric.parameters if p.id == ps_raw.get("parameter_id")),
            None
        )
        weight = rubric_param.weight if rubric_param else float(ps_raw.get("weight", 0))
        evidence = ps_raw.get("evidence", [])
        if not evidence:
            evidence = ["[No direct evidence found]"]

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
        rescued=True,
        summary=raw.get("summary", ""),
    )

    why_rescued = raw.get("why_rescued", "Rescue-band re-evaluation identified higher fit than initial retrieval.")
    logger.debug(f"  Critic [{profile.candidate_id}]: {score.total_score:.1f}/100 rescued={score.rescued}")
    return score, why_rescued


def run_critic(
    rescue_band: list[BandedCandidate],
    rubric: Rubric,
    shortlist_bar: float,
    max_workers: int = 4,
) -> tuple[list[CandidateScore], RescuedReport]:
    """
    Run the critic on all rescue-band candidates.
    Returns (promoted_scores, rescued_report).
    promoted_scores = candidates that cleared the shortlist_bar.
    """
    if not rescue_band:
        return [], RescuedReport()

    logger.info(f"Critic: re-scoring {len(rescue_band)} rescue-band candidates (bar={shortlist_bar:.1f})")

    promoted: list[CandidateScore] = []
    rescued_entries: list[RescuedEntry] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_critic_score_one, banded, rubric): banded
            for banded in rescue_band
        }
        for future in as_completed(futures):
            banded = futures[future]
            try:
                score, why_rescued = future.result()
            except Exception as e:
                logger.error(f"Critic error for {banded.profile.candidate_id}: {e}")
                continue

            if score.passed_gates and score.total_score >= shortlist_bar:
                promoted.append(score)
                # Collect all evidence from all parameters
                all_evidence = [
                    ev
                    for ps in score.parameter_scores
                    for ev in ps.evidence
                    if ev != "[No direct evidence found]"
                ][:5]  # Top 5 evidence spans

                rescued_entries.append(RescuedEntry(
                    candidate_id=score.candidate_id,
                    retrieval_rank=f"rescue band (similarity={banded.similarity_score:.3f})",
                    new_total_score=round(score.total_score, 1),
                    why_rescued=why_rescued,
                    evidence=all_evidence,
                ))
                logger.info(f"  ✓ Rescued {score.candidate_id}: {score.total_score:.1f}/100")
            else:
                reason = "gate failure" if not score.passed_gates else f"score {score.total_score:.1f} < bar {shortlist_bar:.1f}"
                logger.debug(f"  ✗ Not rescued {score.candidate_id}: {reason}")

    logger.info(f"Critic: {len(promoted)} candidates rescued from {len(rescue_band)} in rescue band")
    return promoted, RescuedReport(rescued=rescued_entries)
