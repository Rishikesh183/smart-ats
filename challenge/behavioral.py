"""
Behavioral score from redrob_signals (23 fields).
Returns a single 0-1 float representing candidate engagement/quality signals.

Scoring rationale:
  - Responsiveness (response_rate, avg_response_time_hours)        → 25%
  - Recency / active job seeker (last_active_date, open_to_work)   → 20%
  - Quality signals (assessment scores, interview completion)       → 25%
  - Offer fit (offer_acceptance_rate, notice_period)               → 15%
  - AI-specific activity (github, profile completeness)            → 15%
"""

from __future__ import annotations

from datetime import date, datetime


def _parse_date(s) -> date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(str(s)[:10], fmt).date()
        except ValueError:
            pass
    return None


def _days_since(d: date | None) -> int:
    if not d:
        return 365  # assume inactive if no data
    return max(0, (date.today() - d).days)


def compute_behavioral_score(signals: dict) -> float:
    """
    Score redrob_signals into a single 0–1 value.
    All sub-scores are 0–1; final is a weighted sum.
    """
    if not signals:
        return 0.30  # neutral default when no data

    # ── Responsiveness (25%) ──────────────────────────────────────────────────
    response_rate = float(signals.get("recruiter_response_rate", 0.5) or 0.5)
    response_time = float(signals.get("avg_response_time_hours", 48) or 48)

    # response_rate is already 0-1 presumably (proportion)
    r_rate_score = min(1.0, max(0.0, response_rate))

    # response time: <4h = 1.0, 24h = 0.7, 48h = 0.5, 96h+ = 0.2
    if response_time <= 4:
        r_time_score = 1.0
    elif response_time <= 12:
        r_time_score = 0.85
    elif response_time <= 24:
        r_time_score = 0.70
    elif response_time <= 48:
        r_time_score = 0.55
    elif response_time <= 96:
        r_time_score = 0.35
    else:
        r_time_score = 0.20

    responsiveness = 0.60 * r_rate_score + 0.40 * r_time_score

    # ── Recency / active seeker (20%) ─────────────────────────────────────────
    last_active = _parse_date(signals.get("last_active_date"))
    days_inactive = _days_since(last_active)
    open_to_work = bool(signals.get("open_to_work_flag", False))

    if days_inactive <= 7:
        recency_score = 1.0
    elif days_inactive <= 30:
        recency_score = 0.85
    elif days_inactive <= 90:
        recency_score = 0.65
    elif days_inactive <= 180:
        recency_score = 0.45
    else:
        recency_score = 0.25

    recency_score = min(1.0, recency_score + (0.15 if open_to_work else 0.0))

    # ── Quality signals (25%) ─────────────────────────────────────────────────
    assessment_scores: dict = signals.get("skill_assessment_scores", {}) or {}
    interview_completion = float(signals.get("interview_completion_rate", 0.5) or 0.5)
    completeness = float(signals.get("profile_completeness_score", 0.5) or 0.5)

    if assessment_scores:
        # Normalize: assume scores are 0-100 scale
        avg_assessment = sum(assessment_scores.values()) / len(assessment_scores) / 100.0
        avg_assessment = min(1.0, max(0.0, avg_assessment))
    else:
        avg_assessment = 0.40  # no assessments taken

    quality = 0.40 * avg_assessment + 0.35 * interview_completion + 0.25 * completeness

    # ── Offer fit (15%) ───────────────────────────────────────────────────────
    offer_acceptance = float(signals.get("offer_acceptance_rate", 0.5) or 0.5)
    notice_days = int(signals.get("notice_period_days", 60) or 60)
    preferred_mode = str(signals.get("preferred_work_mode", "") or "").lower()

    # Notice period: <30d ideal (can join fast), 30-60 ok, 90+ penalty
    if notice_days <= 0:
        notice_score = 0.50  # unknown
    elif notice_days <= 30:
        notice_score = 1.0
    elif notice_days <= 60:
        notice_score = 0.80
    elif notice_days <= 90:
        notice_score = 0.60
    else:
        notice_score = 0.40

    # Hybrid or onsite is fine; remote-only gets mild penalty for this JD
    mode_score = 0.80 if "remote" in preferred_mode and "hybrid" not in preferred_mode else 1.0

    offer_fit = 0.50 * offer_acceptance + 0.35 * notice_score + 0.15 * mode_score

    # ── AI-specific activity (15%) ────────────────────────────────────────────
    github_score = float(signals.get("github_activity_score", 0) or 0)
    # assume 0-100 scale
    github_norm = min(1.0, github_score / 100.0)

    # profile completeness already used above; also reward verified skills
    ai_activity = github_norm

    # ── Weighted final ────────────────────────────────────────────────────────
    final = (
        0.25 * responsiveness
        + 0.20 * recency_score
        + 0.25 * quality
        + 0.15 * offer_fit
        + 0.15 * ai_activity
    )

    return round(min(1.0, max(0.0, final)), 4)
