"""
Stage 4b: Comparative re-ranking of top finalists.
LLM produces an internally consistent ordering of the top N candidates.
"""
from __future__ import annotations
import json

from loguru import logger

from app.llm.client import get_client
from app.llm.prompts import RERANK_SYSTEM, RERANK_USER
from app.models import CandidateScore, Rubric


def comparative_rerank(
    scored_candidates: list[CandidateScore],
    rubric: Rubric,
    finalist_count: int,
) -> list[CandidateScore]:
    """
    Take the top `finalist_count` candidates (by total_score) and ask the LLM
    to produce a final, internally consistent ordering.
    Returns the full list with finalists re-ranked at the top.
    """
    if not scored_candidates:
        return scored_candidates

    # Sort by score descending, take finalists
    sorted_all = sorted(scored_candidates, key=lambda s: s.total_score, reverse=True)
    finalists = sorted_all[:finalist_count]
    rest = sorted_all[finalist_count:]

    if len(finalists) <= 1:
        return finalists + rest

    logger.info(f"Stage 4: Comparative re-rank of {len(finalists)} finalists")

    # Prepare compact summary for the LLM
    finalists_summary = [
        {
            "candidate_id": s.candidate_id,
            "total_score": round(s.total_score, 1),
            "summary": s.summary,
            "top_evidence": [
                ev
                for ps in sorted(s.parameter_scores, key=lambda p: p.score / p.max, reverse=True)[:3]
                for ev in ps.evidence[:1]
            ],
        }
        for s in finalists
    ]

    rubric_summary = f"{rubric.role_title}: {rubric.role_summary}"

    client = get_client()
    prompt = RERANK_USER.format(
        role_title=rubric.role_title,
        rubric_summary=rubric_summary,
        finalists_json=json.dumps(finalists_summary, indent=2),
    )

    try:
        raw = client.complete(
            messages=[{"role": "user", "content": prompt}],
            schema={},
            fast=False,
            system=RERANK_SYSTEM,
        )
    except Exception as e:
        logger.error(f"Re-rank LLM error: {e}. Using score-sorted order.")
        return finalists + rest

    ordered_ids: list[str] = raw.get("ordered_ids", [])
    rerank_notes: dict[str, str] = raw.get("rerank_notes", {})

    if not ordered_ids:
        logger.warning("Re-rank returned no ordered_ids; using score order")
        return finalists + rest

    # Build a lookup
    score_map = {s.candidate_id: s for s in finalists}

    reranked: list[CandidateScore] = []
    seen = set()
    for cid in ordered_ids:
        if cid in score_map and cid not in seen:
            s = score_map[cid]
            # Annotate with re-rank note
            if cid in rerank_notes:
                s.summary = s.summary + f" [Re-rank: {rerank_notes[cid]}]"
            reranked.append(s)
            seen.add(cid)

    # Any finalists not mentioned by the LLM go at the end (in score order)
    for s in finalists:
        if s.candidate_id not in seen:
            reranked.append(s)

    logger.info(f"Re-rank complete: {[s.candidate_id for s in reranked[:5]]}")
    return reranked + rest
