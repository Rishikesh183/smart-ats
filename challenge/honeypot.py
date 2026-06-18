"""
Honeypot / impossible-profile detection.
Returns True if the candidate profile is synthetic noise that should be discarded.

Rules (any single rule triggers a honeypot flag):
  H1: expert/advanced skill with duration_months == 0
  H2: years_of_experience > total career months / 12  (by a large margin)
  H3: career end_date before start_date on same job
  H4: skills list has > 30 items with all at expert level (shotgun expert)
  H5: profile_completeness_score == 0.0 but has extensive career history (bot skeleton)
  H6: total endorsed skills > 50 with all beginner (endorsement farming)
  H7: years_of_experience listed as > 60 or < 0 (impossible value)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


def _parse_date(s: Any) -> date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(str(s)[:10], fmt).date()
        except ValueError:
            pass
    return None


def _months_between(start: date | None, end: date | None) -> float:
    if not start:
        return 0.0
    end = end or date.today()
    return max(0.0, (end.year - start.year) * 12 + end.month - start.month)


def detect_honeypot(cand: dict) -> tuple[bool, list[str]]:
    """
    Returns (is_honeypot: bool, reasons: list[str]).
    If is_honeypot is True, final_score should be zeroed out.
    """
    reasons: list[str] = []
    skills = cand.get("skills", [])
    profile = cand.get("profile", {})
    history = cand.get("career_history", [])
    signals = cand.get("redrob_signals", {})

    years_exp = profile.get("years_of_experience", 0) or 0

    # ── H7: impossible years_of_experience value ──────────────────────────────
    if years_exp < 0 or years_exp > 60:
        reasons.append(f"H7: years_of_experience={years_exp} is impossible")

    # ── H1: expert/advanced skill but 0 months duration ──────────────────────
    high_prof = {"advanced", "expert"}
    h1_count = 0
    for sk in skills:
        prof = str(sk.get("proficiency", "")).lower()
        dur = sk.get("duration_months", None)
        if prof in high_prof and dur is not None and dur == 0:
            h1_count += 1
    if h1_count >= 3:
        reasons.append(f"H1: {h1_count} expert/advanced skills with duration_months=0")

    # ── H4: shotgun expert (> 30 skills all at expert level) ─────────────────
    if len(skills) > 30:
        expert_count = sum(
            1 for sk in skills
            if str(sk.get("proficiency", "")).lower() == "expert"
        )
        if expert_count > 25:
            reasons.append(f"H4: {expert_count} expert skills out of {len(skills)} — shotgun")

    # ── H6: endorsement farming (> 50 endorsed beginner skills) ─────────────
    endorsed_beginners = sum(
        1 for sk in skills
        if str(sk.get("proficiency", "")).lower() == "beginner"
        and (sk.get("endorsements", 0) or 0) > 10
    )
    if endorsed_beginners > 50:
        reasons.append(f"H6: {endorsed_beginners} endorsed-beginner skills")

    # ── H2: years_exp wildly exceeds career history length ──────────────────
    if history:
        total_career_months = sum(
            j.get("duration_months", 0) or 0 for j in history
        )
        # cross-check with date ranges if duration not populated
        if total_career_months == 0:
            for job in history:
                start = _parse_date(job.get("start_date"))
                end = _parse_date(job.get("end_date")) if not job.get("is_current") else None
                total_career_months += _months_between(start, end)

        career_years = total_career_months / 12.0
        if years_exp > 0 and career_years > 0:
            if years_exp > career_years * 2.0 and years_exp - career_years > 10:
                reasons.append(
                    f"H2: years_of_experience={years_exp} >> career history={career_years:.1f}y"
                )

    # ── H3: end_date before start_date ───────────────────────────────────────
    for job in history:
        start = _parse_date(job.get("start_date"))
        end = _parse_date(job.get("end_date"))
        if start and end and end < start:
            reasons.append(
                f"H3: career end_date {end} < start_date {start} at {job.get('company')}"
            )
            break  # one is enough

    # ── H5: completeness=0 but has large career history ──────────────────────
    completeness = signals.get("profile_completeness_score", None)
    if completeness is not None and float(completeness) == 0.0 and len(history) >= 5:
        reasons.append("H5: completeness_score=0 but has 5+ jobs — bot skeleton")

    is_honeypot = len(reasons) > 0
    return is_honeypot, reasons
