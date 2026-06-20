"""
Extract a small random sample from the full candidates.jsonl for local testing.

Usage:
  python sample.py \
    --candidates "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" \
    --out sample_candidates.jsonl \
    --n 500 \
    [--seed 42]
"""

import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="Full candidates.jsonl path")
    ap.add_argument("--out", default="sample_candidates.jsonl")
    ap.add_argument("--n", type=int, default=500, help="Sample size")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[sample] Reading {args.candidates} …")
    all_lines = []
    with open(args.candidates, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_lines.append(line)

    print(f"[sample] Total candidates: {len(all_lines):,}")

    random.seed(args.seed)
    sampled = random.sample(all_lines, min(args.n, len(all_lines)))

    with open(args.out, "w", encoding="utf-8") as f:
        for line in sampled:
            f.write(line + "\n")

    print(f"[sample] Wrote {len(sampled)} candidates -> {args.out}")

    # Quick sanity: show first candidate's ID + title
    first = json.loads(sampled[0])
    p = first.get("profile", {})
    print(f"[sample] First: {first.get('candidate_id')} | {p.get('current_title')} | {p.get('location')}")


if __name__ == "__main__":
    main()
