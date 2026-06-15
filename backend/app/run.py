"""
Pipeline orchestrator: ties all stages together.
Entry point: run_pipeline(job_description, mode, rubric=None) → RankedShortlist
"""
from __future__ import annotations
import time
from typing import Optional

from loguru import logger

from app.models import (
    Mode,
    RankRequest,
    RankedShortlist,
    Rubric,
)
from app.pipeline.extract import get_all_profiles
from app.pipeline.gates import evaluate_gates
from app.pipeline.retrieval import build_index, retrieve


def run_pipeline(
    job_description: str,
    mode: Mode = "normal",
    rubric: Optional[Rubric] = None,
) -> RankedShortlist:
    """
    Full pipeline:
    Pass 1 → Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4 → shortlist
    """
    from app.config import settings
    from app.pipeline.rubric import build_rubric
    from app.pipeline.scorer import score_candidates
    from app.pipeline.critic import run_critic
    from app.pipeline.rerank import comparative_rerank
    from app.models import RescuedReport

    t0 = time.time()
    stats: dict = {"mode": mode, "timings": {}}

    # ── Pass 1: Build rubric ───────────────────────────────────────────────────
    if rubric is None:
        t = time.time()
        rubric = build_rubric(job_description)
        stats["timings"]["rubric_build"] = round(time.time() - t, 2)

    # ── Stage 0: Load profiles ─────────────────────────────────────────────────
    t = time.time()
    all_profiles = get_all_profiles()
    stats["total_candidates"] = len(all_profiles)
    stats["timings"]["extract"] = round(time.time() - t, 2)

    # ── Stage 1: Semantic retrieval + banding ──────────────────────────────────
    t = time.time()
    build_index(all_profiles)
    banded = retrieve(job_description, all_profiles, mode=mode)
    advance_band = [b for b in banded if b.band == "advance"]
    rescue_band  = [b for b in banded if b.band == "rescue"]
    stats["retrieved"] = len(banded)
    stats["advance_count"] = len(advance_band)
    stats["rescue_count"] = len(rescue_band)
    stats["timings"]["retrieval"] = round(time.time() - t, 2)

    # ── Stage 2: Intelligent gate filtering ────────────────────────────────────
    t = time.time()
    advance_profiles = []
    gate_dropped = []
    for b in advance_band:
        passed, failures = evaluate_gates(b.profile, rubric.hard_gates)
        if passed:
            advance_profiles.append(b.profile)
        else:
            gate_dropped.append((b.profile.candidate_id, failures))
            logger.info(f"Gate drop [{b.profile.candidate_id}]: {failures}")

    stats["gate_dropped"] = len(gate_dropped)
    stats["scoring_pool"] = len(advance_profiles)
    stats["timings"]["gates"] = round(time.time() - t, 2)

    # ── Stage 3: Evidence-grounded scoring ────────────────────────────────────
    t = time.time()
    scored = score_candidates(advance_profiles, rubric)

    # Also respect gate failures found by LLM during scoring
    passed_scored = [s for s in scored if s.passed_gates]
    gate_failed_by_llm = [s for s in scored if not s.passed_gates]
    stats["llm_gate_failures"] = len(gate_failed_by_llm)
    stats["timings"]["scoring"] = round(time.time() - t, 2)

    # ── Stage 4a: Critic (rescue band, high/extra_high modes) ─────────────────
    rescued_report = RescuedReport()
    rescued_scores: list = []

    if mode in ("high", "extra_high") and rescue_band:
        t = time.time()
        # Compute shortlist bar = score of the lowest-ranked advance candidate (or 40 minimum)
        if passed_scored:
            sorted_scored = sorted(passed_scored, key=lambda s: s.total_score)
            shortlist_bar = max(sorted_scored[0].total_score * 0.85, 40.0)
        else:
            shortlist_bar = 40.0

        rescued_scores, rescued_report = run_critic(rescue_band, rubric, shortlist_bar)
        stats["rescued_count"] = len(rescued_scores)
        stats["timings"]["critic"] = round(time.time() - t, 2)

    # ── Stage 4b: Comparative re-rank ─────────────────────────────────────────
    t = time.time()
    all_passing = passed_scored + rescued_scores
    finalist_count = (
        settings.finalist_count_extra_high if mode == "extra_high"
        else settings.finalist_count_normal
    )
    reranked = comparative_rerank(all_passing, rubric, finalist_count)
    stats["timings"]["rerank"] = round(time.time() - t, 2)
    stats["shortlisted"] = len(reranked)
    stats["total_time_s"] = round(time.time() - t0, 2)

    logger.info(
        f"Pipeline done [{mode}]: "
        f"{stats['total_candidates']} total → "
        f"{stats['advance_count']} advance → "
        f"{stats['shortlisted']} shortlisted | "
        f"rescued={len(rescued_report.rescued)} | "
        f"time={stats['total_time_s']}s"
    )

    return RankedShortlist(
        mode=mode,
        shortlist=reranked,
        rescued_report=rescued_report,
        rubric=rubric,
        stats=stats,
    )
