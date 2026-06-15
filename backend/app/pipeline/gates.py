"""
Stage 2: Intelligent gate evaluation.
Only true hard gates (work auth, mandatory certs) can drop a candidate.
Scored parameters are never treated as gates here.
"""
from __future__ import annotations
import re

from loguru import logger

from app.models import CandidateProfile, HardGate


# Patterns that indicate a candidate IS authorized to work
_AUTH_POSITIVE = re.compile(
    r"\b(us\s+citizen|green\s+card|ead|work\s+permit|tn\s+visa|o-?1|h-?1b|authorized|valid|nafta)\b",
    re.I,
)

# Patterns that indicate a candidate is NOT authorized
_AUTH_NEGATIVE = re.compile(
    r"\b(not\s+authorized|expired|student\s+visa.*expired|opt\s+expired|invalid|no\s+work)\b",
    re.I,
)


def _check_work_authorization(profile: CandidateProfile) -> bool:
    """
    Heuristic check: is this candidate authorized to work?
    Errs on the side of inclusion (returns True if ambiguous).
    """
    work_auth = str(profile.raw.get("work_auth", "")).strip()
    full_text = profile.full_text

    if not work_auth and not full_text:
        return True  # no info → don't drop

    text = f"{work_auth} {full_text}"

    if _AUTH_NEGATIVE.search(text):
        return False
    if _AUTH_POSITIVE.search(text):
        return True
    # Ambiguous — don't drop; let scorer handle with lower confidence
    return True


def evaluate_gates(
    profile: CandidateProfile,
    gates: list[HardGate],
) -> tuple[bool, list[str]]:
    """
    Evaluate all hard gates for a candidate.
    Returns (passed: bool, failures: list[gate_id]).
    """
    failures: list[str] = []

    for gate in gates:
        gate_id = gate.id.lower()
        req = gate.requirement.lower()

        # Work authorization gate
        if any(kw in gate_id or kw in req for kw in ["work_auth", "authorization", "visa", "citizen"]):
            if not _check_work_authorization(profile):
                failures.append(gate.id)
                logger.debug(f"  Gate FAIL [{profile.candidate_id}]: {gate.id}")
            continue

        # For other gate types: attempt a keyword match in full_text
        # Only fail if there's EXPLICIT negative evidence — we never assume failure
        # (The LLM scorer handles nuanced gate checks in Stage 3)
        # So by default all non-work-auth gates pass at this stage
        # They will be re-checked semantically in the scorer prompt

    passed = len(failures) == 0
    return passed, failures
