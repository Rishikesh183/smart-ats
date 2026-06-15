"""
Eval: Planted-candidate recall test.
Measures what % of known strong-but-keyword-poor candidates each mode recovers.
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from loguru import logger
from app.run import run_pipeline
from app.models import EvalResult, Mode

# These are candidates marked as PLANTED in our dataset.
# They have equivalent/superset stacks the cheap retrieval might miss.
PLANTED_CANDIDATE_IDS = ["C002", "C003", "C009", "C014", "C020"]

SAMPLE_JD = """
Senior Full-Stack Engineer

We are looking for a Senior Full-Stack Engineer to join our product team.

Requirements:
- 4+ years of professional software engineering experience
- Strong proficiency in React and Next.js
- Experience with TypeScript
- Backend experience with Node.js or Python
- Experience with databases (PostgreSQL or similar)
- Experience shipping production systems to real users

Nice to have:
- AWS or cloud platform experience
- Open source contributions
- Experience mentoring other engineers
- Track record of measurable performance improvements

Work authorization: Must be authorized to work in the United States.
"""


def run_recall_test(modes: list[Mode] | None = None) -> dict[Mode, EvalResult]:
    if modes is None:
        modes = ["normal", "high", "extra_high"]

    results: dict[Mode, EvalResult] = {}

    for mode in modes:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running eval: mode={mode}")
        logger.info(f"{'='*50}")

        shortlist = run_pipeline(SAMPLE_JD, mode=mode)
        shortlisted_ids = {s.candidate_id for s in shortlist.shortlist}
        rescued_ids = {r.candidate_id for r in shortlist.rescued_report.rescued}
        all_recovered = shortlisted_ids | rescued_ids

        recovered_planted = [pid for pid in PLANTED_CANDIDATE_IDS if pid in all_recovered]
        recall = len(recovered_planted) / len(PLANTED_CANDIDATE_IDS)

        # Explainability check
        explainability_failures = []
        for score in shortlist.shortlist:
            for ps in score.parameter_scores:
                if not ps.evidence or ps.evidence == ["[No direct evidence found in profile]"]:
                    explainability_failures.append(
                        f"{score.candidate_id}.{ps.parameter_id}: no evidence"
                    )

        result = EvalResult(
            mode=mode,
            total_candidates=shortlist.stats.get("total_candidates", 0),
            advanced=shortlist.stats.get("advance_count", 0),
            rescued_count=len(shortlist.rescued_report.rescued),
            shortlisted=len(shortlist.shortlist),
            planted_recall=round(recall, 3),
            planted_ids=PLANTED_CANDIDATE_IDS,
            recovered_planted=recovered_planted,
            explainability_ok=len(explainability_failures) == 0,
            explainability_failures=explainability_failures[:10],
        )
        results[mode] = result

        logger.info(f"  Planted recall [{mode}]: {recall:.0%} ({len(recovered_planted)}/{len(PLANTED_CANDIDATE_IDS)})")
        logger.info(f"  Recovered planted: {recovered_planted}")
        logger.info(f"  Shortlisted: {len(shortlist.shortlist)}")
        logger.info(f"  Rescued: {len(shortlist.rescued_report.rescued)}")
        logger.info(f"  Explainability OK: {result.explainability_ok}")

    return results


if __name__ == "__main__":
    results = run_recall_test()
    print("\n" + "="*60)
    print("RECALL TEST SUMMARY")
    print("="*60)
    for mode, result in results.items():
        print(f"\nMode: {mode}")
        print(f"  Planted recall:   {result.planted_recall:.0%}")
        print(f"  Recovered:        {result.recovered_planted}")
        print(f"  Shortlisted:      {result.shortlisted}")
        print(f"  Rescued:          {result.rescued_count}")
        print(f"  Explainability:   {'✓' if result.explainability_ok else '✗'}")
        if result.explainability_failures:
            for f in result.explainability_failures[:3]:
                print(f"    - {f}")
