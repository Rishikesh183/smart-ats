"""
View full candidate profiles from the JSONL dataset.

Usage:
  python inspect_candidates.py CAND_0011555 CAND_0087783

  Or set DEFAULT_TARGETS below and run with no args.

Set --candidates to point at a different jsonl file.
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_TARGETS = [
    # paste candidate IDs here to inspect without typing them every time
]

DEFAULT_CANDIDATES = "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"


def fmt_section(title):
    bar = "-" * 60
    return f"\n{bar}\n  {title}\n{bar}"


def show(cand):
    p = cand.get("profile", {})
    cid = cand.get("candidate_id", "?")

    print(fmt_section(f"CANDIDATE: {cid}"))

    # Basic info
    print(f"  Title    : {p.get('current_title', '-')}")
    print(f"  Company  : {p.get('current_company', '-')}")
    print(f"  Location : {p.get('location', '-')}")
    print(f"  Years XP : {p.get('years_of_experience', '-')}")
    print(f"  Headline : {p.get('headline', '-')}")
    reloc = p.get("open_to_relocation")
    if reloc is not None:
        print(f"  Relocate : {reloc}")

    # Summary
    summary = p.get("summary", "")
    if summary:
        print(f"\n  SUMMARY:")
        for line in summary[:600].split("\n"):
            print(f"    {line}")
        if len(summary) > 600:
            print("    [... truncated]")

    # Skills
    skills = cand.get("skills", [])
    if skills:
        print(f"\n  SKILLS ({len(skills)}):")
        for sk in skills:
            name = sk.get("name", "")
            prof = sk.get("proficiency", "")
            endorse = sk.get("endorsements", 0)
            months = sk.get("duration_months", 0)
            parts = [name]
            if prof:
                parts.append(prof)
            if endorse:
                parts.append(f"{endorse} endorsements")
            if months:
                parts.append(f"{months}mo")
            print(f"    - {' | '.join(parts)}")

    # Career
    history = cand.get("career_history", [])
    if history:
        print(f"\n  CAREER HISTORY ({len(history)} roles):")
        for job in history:
            title = job.get("title", "-")
            company = job.get("company", "-")
            months = job.get("duration_months", "?")
            size = job.get("company_size", "")
            desc = job.get("description", "")
            print(f"    [{months}mo] {title} @ {company}" + (f" ({size})" if size else ""))
            if desc:
                for line in desc[:300].split("\n"):
                    if line.strip():
                        print(f"         {line.strip()}")
                if len(desc) > 300:
                    print("         [... truncated]")

    # Education
    edu_list = cand.get("education", [])
    if edu_list:
        print(f"\n  EDUCATION:")
        for edu in edu_list:
            degree = edu.get("degree", "")
            field = edu.get("field_of_study", "")
            inst = edu.get("institution", "")
            tier = edu.get("institution_tier", "")
            year = edu.get("graduation_year", "")
            line = " ".join(x for x in [degree, field, "@", inst] if x and x != "@")
            extras = " ".join(x for x in [tier, str(year) if year else ""] if x)
            print(f"    - {line}" + (f" [{extras}]" if extras else ""))

    # Redrob signals
    signals = cand.get("redrob_signals", {})
    if signals:
        print(f"\n  REDROB SIGNALS:")
        for k, v in signals.items():
            print(f"    {k}: {v}")

    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", help="Candidate IDs to inspect")
    ap.add_argument(
        "--candidates",
        default=DEFAULT_CANDIDATES,
        help="Path to candidates.jsonl",
    )
    args = ap.parse_args()

    targets = set(args.targets) if args.targets else set(DEFAULT_TARGETS)
    if not targets:
        print("Usage: python inspect_candidates.py CAND_001 CAND_002 ...")
        print("Or set DEFAULT_TARGETS at top of file.")
        sys.exit(1)

    found = set()
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: candidates file not found: {candidates_path}")
        print("Try: python inspect_candidates.py --candidates sample_100.jsonl CAND_001")
        sys.exit(1)

    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            if cand.get("candidate_id") in targets:
                show(cand)
                found.add(cand["candidate_id"])
                if found == targets:
                    break

    missing = targets - found
    if missing:
        print(f"WARNING: not found: {missing}")


if __name__ == "__main__":
    main()
