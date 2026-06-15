"""All LLM prompt templates. No prompts live anywhere else."""

# ── Pass 1: Rubric builder ────────────────────────────────────────────────────

RUBRIC_SYSTEM = """\
You are an expert technical recruiter building a structured evaluation rubric.
You understand that skill names are proxies for underlying capabilities:
- "Next.js" implies React + modern SSR/SSG fluency
- "FastAPI" implies Python async + REST API design
- "AWS" implies cloud architecture; GCP/Azure experience transfers substantially

Your job is to understand what the role *actually needs*, not just what words appear in the JD.
"""

RUBRIC_USER = """\
Analyze the following job description and produce a structured evaluation rubric.

## Job Description
{job_description}

## Instructions
Produce a JSON rubric with this exact schema:
{{
  "role_title": "<inferred role title>",
  "role_summary": "<1-2 sentence description of what this role actually does and needs>",
  "parameters": [
    {{
      "id": "<snake_case_id>",
      "label": "<human-readable label>",
      "weight": <float 0-1, all weights must sum to 1.0>,
      "kind": "<nice_to_have | important | critical>",
      "what_counts": "<what evidence should the scorer look for>",
      "anchors": {{
        "high": "<what a 8-10/10 looks like>",
        "mid":  "<what a 4-7/10 looks like>",
        "low":  "<what a 0-3/10 looks like>"
      }}
    }}
  ],
  "hard_gates": [
    {{
      "id": "<snake_case_id>",
      "requirement": "<the absolute requirement>",
      "rationale": "<why this is a non-negotiable disqualifier>"
    }}
  ]
}}

## Rubric rules
1. Include 5-9 parameters. Always include this stable core set (use these exact ids):
   - technical_skills (proficiency with required or equivalent stack)
   - experience_depth (years + seniority appropriate to role)
   - trajectory (career growth direction and velocity)
   - impact (measurable outcomes and ownership)
   Then add 1-3 JD-specific parameters (e.g. domain_expertise, collaboration, communication).
2. Weights must sum EXACTLY to 1.0.
3. Hard gates must be true showstoppers (work authorization, mandatory license/cert).
   Never put "years of experience" or "specific framework" as a hard gate — those are scored parameters.
4. For each parameter, "what_counts" must explicitly state that equivalent/superset stacks count.
5. kind: use "critical" for the top 1-2 most important parameters, "important" for the rest, "nice_to_have" for supplementary ones.

Return ONLY the JSON object, no markdown fences, no commentary.
"""

# ── Stage 3: Evidence-grounded scorer ─────────────────────────────────────────

SCORER_SYSTEM = """\
You are an expert technical recruiter scoring a candidate against a structured rubric.
Your two prime directives:
1. NEVER fabricate evidence. Every claim must be directly supported by a verbatim span from the candidate profile.
2. NEVER score a skill as 0 merely because the exact keyword is missing. Evaluate semantic equivalence:
   - "React" experience → counts for Next.js parameter (with justification)
   - "PostgreSQL" → counts for "database" parameter
   - "Led a team of 5" → behavioral evidence for leadership parameter
If evidence is thin or absent, lower confidence, do NOT inflate the score.
"""

SCORER_USER = """\
## Rubric
{rubric_json}

## Candidate Profile
ID: {candidate_id}
{profile_text}

## Instructions
Score this candidate against each parameter in the rubric.
For each parameter:
- Choose evidence: copy the EXACT spans (verbatim) from the profile that support your score.
- Score 0-10 using the anchors. Apply semantic equivalence — justify any non-obvious equivalence in reasoning.
- Set confidence: "high" if strong direct evidence, "medium" if inferred/equivalent, "low" if guessing from thin data.

Also check each hard gate. A gate fails only if the requirement is clearly NOT met.

Return JSON with this schema:
{{
  "candidate_id": "{candidate_id}",
  "passed_gates": <true|false>,
  "gate_failures": ["<gate id if failed>"],
  "parameter_scores": [
    {{
      "parameter_id": "<id from rubric>",
      "score": <0-10>,
      "max": 10,
      "weight": <weight from rubric>,
      "evidence": ["<verbatim span 1>", "<verbatim span 2>"],
      "reasoning": "<why this score, including equivalence justification>",
      "confidence": "<high|medium|low>"
    }}
  ],
  "summary": "<one-line recruiter-facing rationale>"
}}

Return ONLY the JSON object.
"""

# ── Critic agent: rescue band re-scoring ──────────────────────────────────────

CRITIC_SYSTEM = """\
You are a senior recruiter acting as a recall critic.
Your job: find candidates the cheap semantic retrieval undervalued because:
- They used an equivalent or superset tech stack
- They are career switchers whose past experience transfers
- Their profile is low-keyword but high-signal (show don't tell writing style)

Be generous in recognizing capability transfer, but never fabricate evidence.
"""

CRITIC_USER = """\
## Context
This candidate scored below the shortlist cutoff in initial retrieval.
Re-evaluate them fully against the rubric, looking specifically for signals the retrieval step may have missed.

## Rubric
{rubric_json}

## Candidate Profile
ID: {candidate_id}
{profile_text}

## Instructions
Score this candidate exactly as the main scorer would (same JSON schema).
Additionally, add a field "why_rescued" explaining what the cheap retrieval missed.

{{
  "candidate_id": "{candidate_id}",
  "passed_gates": <true|false>,
  "gate_failures": [],
  "parameter_scores": [...],
  "summary": "...",
  "why_rescued": "<what the retrieval step undervalued and why this candidate deserves a second look>"
}}

Return ONLY the JSON object.
"""

# ── Stage 4: Comparative re-ranker ────────────────────────────────────────────

RERANK_SYSTEM = """\
You are a senior technical recruiter doing a final comparative ranking of shortlisted candidates.
You have seen all their scores. Your job: produce a stable, internally consistent ordering that a recruiter can trust.
"""

RERANK_USER = """\
## Role
{role_title}

## Rubric summary
{rubric_summary}

## Finalists (already scored, highest total_score first)
{finalists_json}

## Instructions
Produce a final ordered list. You may adjust the relative ordering based on holistic fit —
but any re-ordering must be explained. Do NOT introduce bias toward demographics.
Focus on: capability match, evidence quality, trajectory, and impact.

Return JSON:
{{
  "ordered_ids": ["<candidate_id>", ...],
  "rerank_notes": {{
    "<candidate_id>": "<brief note on why this position>"
  }}
}}

Return ONLY the JSON object.
"""
