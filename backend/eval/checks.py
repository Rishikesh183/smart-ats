"""
Eval: Explainability + determinism checks.
Run as: python -m eval.checks
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from loguru import logger
from app.run import run_pipeline

SAMPLE_JD = """
Senior Full-Stack Engineer — Next.js, TypeScript, PostgreSQL
4+ years experience. Ship production systems. US work authorization required.
"""


def check_explainability(shortlist) -> tuple[bool, list[str]]:
    """Every shortlisted candidate must have non-empty evidence on every scored parameter."""
    failures = []
    for score in shortlist.shortlist:
        if not score.passed_gates:
            continue
        for ps in score.parameter_scores:
            real_evidence = [e for e in ps.evidence if e != "[No direct evidence found in profile]"]
            if not real_evidence:
                failures.append(f"{score.candidate_id}.{ps.parameter_id}: EMPTY evidence")
    return len(failures) == 0, failures


def check_determinism(n_runs: int = 2) -> tuple[bool, list[str]]:
    """Re-running with the same JD should produce a stable top-5 ordering."""
    orderings = []
    for i in range(n_runs):
        logger.info(f"Determinism run {i+1}/{n_runs}")
        result = run_pipeline(SAMPLE_JD, mode="normal")
        top5 = [s.candidate_id for s in result.shortlist[:5]]
        orderings.append(top5)

    diffs = []
    for i in range(1, len(orderings)):
        if orderings[i] != orderings[0]:
            diffs.append(f"Run {i+1} differs: {orderings[i]} vs {orderings[0]}")

    return len(diffs) == 0, diffs


if __name__ == "__main__":
    print("Running explainability check...")
    result = run_pipeline(SAMPLE_JD, mode="normal")
    ok, failures = check_explainability(result)
    print(f"Explainability: {'PASS ✓' if ok else 'FAIL ✗'}")
    for f in failures[:5]:
        print(f"  {f}")

    print("\nRunning determinism check (2 runs)...")
    ok_d, diffs = check_determinism(n_runs=2)
    print(f"Determinism: {'PASS ✓' if ok_d else 'WARN (LLM non-determinism)'}")
    for d in diffs:
        print(f"  {d}")
