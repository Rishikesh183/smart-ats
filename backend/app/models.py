"""Pydantic models for the entire pipeline."""
from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator


# ── Candidate data ────────────────────────────────────────────────────────────

class CandidateProfile(BaseModel):
    candidate_id: str
    full_text: str                       # concatenated, cleaned — used for embedding
    career_history: list[dict[str, Any]] # roles, durations, titles
    skills: list[str]
    platform_activity: dict[str, Any]    # github/portfolio signals
    behavioral_signals: dict[str, Any]   # soft signals
    raw: dict[str, Any]                  # original row, untouched


# ── Rubric ────────────────────────────────────────────────────────────────────

class ScoringAnchor(BaseModel):
    high: str
    mid: str
    low: str


class RubricParameter(BaseModel):
    id: str
    label: str
    weight: float
    kind: Literal["nice_to_have", "important", "critical"]
    what_counts: str
    anchors: ScoringAnchor


class HardGate(BaseModel):
    id: str
    requirement: str
    rationale: str


class Rubric(BaseModel):
    role_title: str
    role_summary: str
    parameters: list[RubricParameter]
    hard_gates: list[HardGate] = Field(default_factory=list)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "Rubric":
        total = sum(p.weight for p in self.parameters)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Parameter weights must sum to 1.0, got {total:.3f}")
        return self


# ── Scoring ───────────────────────────────────────────────────────────────────

class ParameterScore(BaseModel):
    parameter_id: str
    score: float        # 0–10
    max: float = 10.0
    weight: float
    evidence: list[str] # exact spans from the profile
    reasoning: str
    confidence: Literal["high", "medium", "low"]


class CandidateScore(BaseModel):
    candidate_id: str
    passed_gates: bool
    gate_failures: list[str] = Field(default_factory=list)
    total_score: float = 0.0            # 0–100
    parameter_scores: list[ParameterScore] = Field(default_factory=list)
    rescued: bool = False
    summary: str = ""

    @model_validator(mode="after")
    def compute_total(self) -> "CandidateScore":
        if self.parameter_scores:
            self.total_score = sum(
                (ps.score / ps.max) * ps.weight for ps in self.parameter_scores
            ) * 100
        return self


# ── Retrieval banding ─────────────────────────────────────────────────────────

class BandedCandidate(BaseModel):
    profile: CandidateProfile
    similarity_score: float
    band: Literal["advance", "rescue", "drop"]


# ── Rescued report ────────────────────────────────────────────────────────────

class RescuedEntry(BaseModel):
    candidate_id: str
    retrieval_rank: str
    new_total_score: float
    why_rescued: str
    evidence: list[str]


class RescuedReport(BaseModel):
    rescued: list[RescuedEntry] = Field(default_factory=list)


# ── Pipeline request / result ─────────────────────────────────────────────────

Mode = Literal["normal", "high", "extra_high"]


class RankRequest(BaseModel):
    job_description: str
    mode: Mode = "normal"
    rubric: Optional[Rubric] = None     # if None, build from JD


class RankedShortlist(BaseModel):
    mode: Mode
    shortlist: list[CandidateScore]
    rescued_report: RescuedReport
    rubric: Rubric
    stats: dict[str, Any] = Field(default_factory=dict)


# ── Eval ─────────────────────────────────────────────────────────────────────

class EvalResult(BaseModel):
    mode: Mode
    total_candidates: int
    advanced: int
    rescued_count: int
    shortlisted: int
    planted_recall: Optional[float] = None    # % of planted candidates recovered
    planted_ids: list[str] = Field(default_factory=list)
    recovered_planted: list[str] = Field(default_factory=list)
    explainability_ok: bool = True
    explainability_failures: list[str] = Field(default_factory=list)
