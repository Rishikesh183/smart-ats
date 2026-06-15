"""
Normalization layer: maps raw dataset columns → CandidateProfile.
Handles missing fields gracefully — never fabricates data.
"""
from __future__ import annotations
import json
import re
from typing import Any

import pandas as pd

from app.models import CandidateProfile


# ── Column name aliases ───────────────────────────────────────────────────────
# Maps known column aliases to canonical names so we handle varied datasets.
COLUMN_MAP = {
    # id
    "candidate_id": "id", "cid": "id", "applicant_id": "id",
    # name
    "candidate_name": "name", "applicant_name": "name", "full_name": "name",
    # title
    "job_title": "title", "current_title": "title", "position": "title",
    # experience
    "years_experience": "years_exp", "experience_years": "years_exp", "yoe": "years_exp",
    # skills
    "technical_skills": "skills", "skill_set": "skills", "tech_stack": "skills",
    # work auth
    "work_authorization": "work_auth", "visa_status": "work_auth", "authorization": "work_auth",
    # summary / bio
    "bio": "summary", "profile_summary": "summary", "about": "summary",
    # experience detail
    "work_experience": "experience", "career_history": "experience", "jobs": "experience",
    # github / portfolio
    "github_url": "github", "portfolio": "github", "portfolio_url": "github",
    # education
    "education_background": "education", "degree": "education",
    # notes
    "planted_note": "notes", "remarks": "notes",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column aliases so downstream code can use canonical names."""
    rename = {col: COLUMN_MAP[col.lower()] for col in df.columns if col.lower() in COLUMN_MAP}
    return df.rename(columns=rename)


def _safe_get(row: dict, *keys: str, default: Any = "") -> Any:
    """Try multiple keys, return first hit or default."""
    for k in keys:
        v = row.get(k, "")
        if v and str(v).strip():
            return v
    return default


def _parse_skills(raw: str) -> list[str]:
    """Parse comma/semicolon separated skills list."""
    if not raw:
        return []
    # Handle JSON array
    if raw.strip().startswith("["):
        try:
            items = json.loads(raw)
            return [str(s).strip() for s in items if s]
        except json.JSONDecodeError:
            pass
    # Comma or semicolon separated
    parts = re.split(r"[,;]", raw)
    return [p.strip() for p in parts if p.strip()]


def _parse_experience(raw: str) -> list[dict[str, Any]]:
    """Parse experience JSON or return empty list."""
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except (json.JSONDecodeError, TypeError):
        pass
    # Plain text — wrap as single entry
    if raw.strip():
        return [{"raw_text": raw.strip()}]
    return []


def _extract_platform_activity(row: dict) -> dict[str, Any]:
    """Build platform_activity from available columns."""
    activity: dict[str, Any] = {}

    github_raw = _safe_get(row, "github", "github_url", "portfolio")
    if github_raw:
        activity["github"] = str(github_raw)
        # Extract follower count hint
        m = re.search(r"(\d[\d,.]+[KkMm]?)\s*follower", github_raw, re.I)
        if m:
            activity["github_followers_hint"] = m.group(1)
        # Extract star hints
        m = re.search(r"(\d[\d,.]+[KkMm]?)\s*star", github_raw, re.I)
        if m:
            activity["github_stars_hint"] = m.group(1)

    for col in ("portfolio", "linkedin", "website", "blog"):
        if row.get(col):
            activity[col] = row[col]

    return activity


def _extract_behavioral_signals(row: dict) -> dict[str, Any]:
    """Extract soft signals from the raw data."""
    signals: dict[str, Any] = {}

    summary = str(row.get("summary", ""))

    # Mentorship signals
    mentorship = re.search(r"mentor(?:ing|ed|s)?\s+(\d+)\s+(?:junior|engineer|developer|intern)", summary, re.I)
    if mentorship:
        signals["mentorship"] = f"Mentors {mentorship.group(1)} engineers (from summary)"

    # Team leadership
    team_lead = re.search(r"(?:led|managed|leading)\s+(?:team\s+of\s+)?(\d+)\s+(?:engineer|developer|person)", summary, re.I)
    if team_lead:
        signals["team_leadership"] = f"Led team of {team_lead.group(1)} (from summary)"

    # Community / speaking
    if re.search(r"(?:speak|talk|conference|meetup|open.source|contrib)", summary, re.I):
        signals["community"] = "Community involvement mentioned"

    return signals


def _build_full_text(row: dict, skills: list[str], experience: list[dict]) -> str:
    """
    Build a single clean string for embedding.
    Order: title + years → skills → summary → experience bullets → education + github.
    """
    parts: list[str] = []

    title = str(row.get("title", "")).strip()
    years = str(row.get("years_exp", "")).strip()
    if title:
        parts.append(f"Role: {title}" + (f" ({years} years experience)" if years else ""))

    skills_text = ", ".join(skills)
    if skills_text:
        parts.append(f"Skills: {skills_text}")

    summary = str(row.get("summary", "")).strip()
    if summary:
        parts.append(f"Summary: {summary}")

    # Flatten experience into text
    exp_lines: list[str] = []
    for job in experience:
        if isinstance(job, dict):
            role = job.get("role", "")
            company = job.get("company", "")
            yrs = job.get("years", "")
            header = " | ".join(filter(None, [role, company, f"{yrs}yr" if yrs else ""]))
            if header:
                exp_lines.append(header)
            for bullet in job.get("bullets", []):
                exp_lines.append(f"  • {bullet}")
            if "raw_text" in job:
                exp_lines.append(job["raw_text"])
    if exp_lines:
        parts.append("Experience:\n" + "\n".join(exp_lines))

    edu = str(row.get("education", "")).strip()
    if edu:
        parts.append(f"Education: {edu}")

    github = str(row.get("github", "")).strip()
    if github:
        parts.append(f"GitHub/Portfolio: {github}")

    work_auth = str(row.get("work_auth", "")).strip()
    if work_auth:
        parts.append(f"Work Authorization: {work_auth}")

    return "\n\n".join(parts)


def normalize_row(row: dict, idx: int) -> CandidateProfile:
    """Normalize a single raw row into a CandidateProfile."""
    candidate_id = str(_safe_get(row, "id", "candidate_id") or f"CAND_{idx:04d}")
    skills = _parse_skills(str(_safe_get(row, "skills", "technical_skills", default="")))
    experience = _parse_experience(str(_safe_get(row, "experience", "work_experience", "career_history", default="")))
    platform_activity = _extract_platform_activity(row)
    behavioral_signals = _extract_behavioral_signals(row)
    full_text = _build_full_text(row, skills, experience)

    return CandidateProfile(
        candidate_id=candidate_id,
        full_text=full_text,
        career_history=experience,
        skills=skills,
        platform_activity=platform_activity,
        behavioral_signals=behavioral_signals,
        raw=row,
    )


def normalize_dataframe(df: pd.DataFrame) -> list[CandidateProfile]:
    """Normalize an entire DataFrame into a list of CandidateProfile."""
    df = _normalize_columns(df)
    profiles: list[CandidateProfile] = []
    for idx, row in df.iterrows():
        try:
            profile = normalize_row(row.to_dict(), int(str(idx)))
            profiles.append(profile)
        except Exception as e:
            from loguru import logger
            logger.warning(f"Failed to normalize row {idx}: {e}")
    return profiles
