"""
FastAPI application — all endpoints.
POST /rubric     → build rubric from JD
PUT  /rubric     → accept edited rubric
POST /rank       → run full pipeline
GET  /health     → health check
POST /eval       → run eval harness
"""
from __future__ import annotations
import json
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from app.config import settings
from app.models import Mode, RankedShortlist, Rubric, RankRequest

app = FastAPI(title="Smart ATS", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── In-memory result store (single-recruiter demo) ─────────────────────────────
_last_result: Optional[RankedShortlist] = None
_last_rubric: Optional[Rubric] = None


# ── Request / Response schemas ─────────────────────────────────────────────────

class RubricRequest(BaseModel):
    job_description: str


class RankRequestBody(BaseModel):
    job_description: str
    mode: Mode = "normal"
    rubric: Optional[dict[str, Any]] = None   # if provided, skip Pass 1


class EvalRequest(BaseModel):
    modes: list[Mode] = ["normal", "high", "extra_high"]
    job_description: Optional[str] = None


# ── Startup: warm up embedder ──────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("Warming up embedding model and loading profiles...")
    try:
        from app.pipeline.extract import get_all_profiles
        from app.pipeline.retrieval import build_index
        profiles = get_all_profiles()
        build_index(profiles)
        logger.info(f"Ready: {len(profiles)} profiles indexed")
    except Exception as e:
        logger.warning(f"Startup warmup failed (continuing anyway): {e}")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/rubric")
def build_rubric_endpoint(body: RubricRequest) -> dict:
    """Pass 1: Build a rubric from a JD. Returns the rubric for recruiter inspection/editing."""
    global _last_rubric
    try:
        from app.pipeline.rubric import build_rubric
        rubric = build_rubric(body.job_description)
        _last_rubric = rubric
        return rubric.model_dump()
    except Exception as e:
        logger.error(f"Rubric build error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/rubric")
def update_rubric(rubric_data: dict) -> dict:
    """Accept an edited rubric from the recruiter UI."""
    global _last_rubric
    try:
        rubric = Rubric.model_validate(rubric_data)
        _last_rubric = rubric
        return {"status": "ok", "rubric": rubric.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/rank")
def rank_candidates(body: RankRequestBody) -> dict:
    """Run the full pipeline. Optionally accepts a pre-built/edited rubric."""
    global _last_result, _last_rubric

    rubric: Optional[Rubric] = None
    if body.rubric:
        try:
            rubric = Rubric.model_validate(body.rubric)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid rubric: {e}")
    elif _last_rubric:
        rubric = _last_rubric

    try:
        from app.run import run_pipeline
        result = run_pipeline(body.job_description, mode=body.mode, rubric=rubric)
        _last_result = result
        _last_rubric = result.rubric
        return result.model_dump()
    except Exception as e:
        logger.error(f"Rank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/result")
def get_last_result() -> dict:
    """Return the most recent ranking result."""
    if _last_result is None:
        raise HTTPException(status_code=404, detail="No results yet. POST /rank first.")
    return _last_result.model_dump()


@app.post("/eval")
def run_eval(body: EvalRequest) -> dict:
    """Run the eval harness and return recall metrics."""
    try:
        from eval.planted import run_recall_test, SAMPLE_JD
        jd = body.job_description or SAMPLE_JD
        results = run_recall_test(modes=body.modes)
        return {
            mode: result.model_dump()
            for mode, result in results.items()
        }
    except Exception as e:
        logger.error(f"Eval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/candidates")
def list_candidates() -> dict:
    """List all candidates (for debugging / UI display)."""
    from app.pipeline.extract import get_all_profiles
    profiles = get_all_profiles()
    return {
        "total": len(profiles),
        "candidates": [
            {
                "id": p.candidate_id,
                "name": p.raw.get("name", ""),
                "title": p.raw.get("title", ""),
                "skills": p.skills[:6],
            }
            for p in profiles
        ]
    }
