"""
Feature extraction from the competition JSONL schema.
Returns structured numeric features for each candidate — no LLM calls here.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from jd import (
    CONSULTING_FIRMS,
    EDUCATION_TIER_SCORE,
    PREFERRED_LOCATIONS,
    PROFICIENCY_SCORE,
    TARGET_SKILLS,
)

# ─── title alignment ─────────────────────────────────────────────────────────

# Titles that signal genuine AI/ML/Engineering background
_AI_TITLE_KEYWORDS = {
    "ai", "ml", "machine learning", "data scientist", "data science",
    "nlp", "deep learning", "research engineer", "applied scientist",
    "software engineer", "software developer", "backend engineer",
    "fullstack engineer", "full stack", "platform engineer",
    "mlops", "data engineer", "retrieval", "search engineer",
    "ranking engineer", "recommendation", "engineer", "architect",
    "tech lead", "principal engineer", "staff engineer",
}

# Titles that are explicit mismatches regardless of skills listed
_UNRELATED_TITLES = {
    "marketing manager", "marketing executive", "content writer", "content creator",
    "hr manager", "hr executive", "human resources", "recruiter",
    "accountant", "finance manager", "financial analyst", "chartered accountant",
    "graphic designer", "ui designer", "ux designer", "visual designer",
    "sales executive", "sales manager", "business development",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "operations manager", "supply chain", "logistics",
    "customer support", "customer success", "customer service",
    "project manager",   # ambiguous but not technical enough for this JD
}


def compute_title_alignment(current_title: str) -> float:
    """
    Returns a multiplier 0.3–1.0 based on how well the current job title
    aligns with a Senior AI Engineer role.
    The JD explicitly warns: Marketing Manager with all AI keywords ≠ fit.
    """
    title_lower = _lower(current_title)
    if not title_lower:
        return 0.70  # unknown — don't fully penalize

    # Exact unrelated match → heavy penalty
    for unrelated in _UNRELATED_TITLES:
        if unrelated in title_lower:
            return 0.30

    # AI/engineering keyword in title → no penalty
    for ai_kw in _AI_TITLE_KEYWORDS:
        if ai_kw in title_lower:
            return 1.00

    # Ambiguous (e.g. "consultant", "analyst") → mild penalty
    return 0.65


# ─── helpers ────────────────────────────────────────────────────────────────

def _lower(s: Any) -> str:
    return str(s).lower().strip() if s else ""


def _parse_date(s: Any) -> date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(str(s)[:10], fmt).date()
        except ValueError:
            pass
    return None


def _months_between(start: date | None, end: date | None) -> int:
    if not start:
        return 0
    end = end or date.today()
    return max(0, (end.year - start.year) * 12 + end.month - start.month)


# ─── profile text (for embedding) ────────────────────────────────────────────

def build_profile_text(cand: dict) -> str:
    """Concatenate all text fields into a single string for embedding."""
    parts: list[str] = []
    p = cand.get("profile", {})

    for field in ("headline", "summary", "current_title", "current_company",
                  "current_industry", "location"):
        v = p.get(field, "")
        if v:
            parts.append(str(v))

    for job in cand.get("career_history", []):
        bits = [job.get("title", ""), job.get("company", ""), job.get("description", "")]
        parts.append(" ".join(b for b in bits if b))

    for edu in cand.get("education", []):
        parts.append(f"{edu.get('degree','')} {edu.get('field_of_study','')} {edu.get('institution','')}")

    for sk in cand.get("skills", []):
        parts.append(sk.get("name", ""))

    return " ".join(parts)


# ─── skill match score ────────────────────────────────────────────────────────

def compute_skill_score(cand: dict) -> float:
    """
    Score 0-1 based on coverage of TARGET_SKILLS weighted by proficiency,
    endorsements, and duration_months.
    """
    skills = cand.get("skills", [])
    skill_map: dict[str, dict] = {}
    for sk in skills:
        name = _lower(sk.get("name", ""))
        skill_map[name] = sk

    target_set = [t.lower() for t in TARGET_SKILLS]

    total_possible = len(target_set)
    score = 0.0

    for target in target_set:
        # exact match
        match = skill_map.get(target)
        if match is None:
            # substring match (e.g. "embeddings" in "sentence-transformers / embeddings")
            for sk_name, sk in skill_map.items():
                if target in sk_name or sk_name in target:
                    match = sk
                    break

        if match:
            prof = PROFICIENCY_SCORE.get(_lower(match.get("proficiency", "")), 0.40)
            endorsements = min(1.0, match.get("endorsements", 0) / 20.0)
            months = min(1.0, match.get("duration_months", 0) / 36.0)
            # weighted: proficiency counts most, duration adds signal
            contribution = 0.60 * prof + 0.20 * endorsements + 0.20 * months
            score += contribution

    return score / total_possible if total_possible else 0.0


# ─── experience score ─────────────────────────────────────────────────────────

def compute_experience_score(cand: dict) -> tuple[float, bool]:
    """
    Returns (score 0-1, is_purely_consulting).
    Target: 5-9 years; 6-8 ideal. Penalty for all-consulting career.
    """
    p = cand.get("profile", {})
    years = p.get("years_of_experience", 0) or 0

    # Experience score (peak at 6-8 years, taper outside)
    if years <= 0:
        exp_score = 0.0
    elif years <= 3:
        exp_score = 0.30
    elif years <= 5:
        exp_score = 0.60
    elif years <= 9:
        # peak band 5-9
        exp_score = 0.80 + (min(years, 8) - 5) / 3 * 0.20
    elif years <= 12:
        exp_score = 0.90
    else:
        # overqualified
        exp_score = 0.70

    # Consulting detection
    history = cand.get("career_history", [])
    if not history:
        return exp_score, False

    consulting_months = 0
    total_months = 0

    for job in history:
        company = _lower(job.get("company", ""))
        months = job.get("duration_months", 0) or 0
        is_consulting = any(firm in company for firm in CONSULTING_FIRMS)

        total_months += months
        if is_consulting:
            consulting_months += months

    if total_months > 0:
        consulting_ratio = consulting_months / total_months
        is_purely_consulting = consulting_ratio >= 0.90
        if consulting_ratio > 0.5:
            exp_score *= (1.0 - 0.5 * consulting_ratio)
    else:
        is_purely_consulting = False

    # Bonus for product-company experience
    # Schema company_size values: "1-10","11-50","51-200","201-500","501-1000","1001-5000","5001-10000","10001+"
    # Smaller companies more likely to be product companies (excluding big consulting)
    product_sizes = {"1-10", "11-50", "51-200", "201-500", "501-1000"}
    product_months = sum(
        j.get("duration_months", 0) or 0
        for j in history
        if j.get("company_size", "") in product_sizes
        and not any(f in _lower(j.get("company", "")) for f in CONSULTING_FIRMS)
    )
    if product_months >= 36:
        exp_score = min(1.0, exp_score + 0.10)

    return exp_score, is_purely_consulting


# ─── education score ───────────────────────────────────────────────�