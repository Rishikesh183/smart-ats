const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function req<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export type Mode = "normal" | "high" | "extra_high";

export interface ScoringAnchor { high: string; mid: string; low: string; }
export interface RubricParameter {
  id: string; label: string; weight: number;
  kind: string; what_counts: string; anchors: ScoringAnchor;
}
export interface HardGate { id: string; requirement: string; rationale: string; }
export interface Rubric {
  role_title: string; role_summary: string;
  parameters: RubricParameter[]; hard_gates: HardGate[];
}

export interface ParameterScore {
  parameter_id: string; score: number; max: number; weight: number;
  evidence: string[]; reasoning: string; confidence: "high" | "medium" | "low";
}
export interface CandidateScore {
  candidate_id: string; passed_gates: boolean; gate_failures: string[];
  total_score: number; parameter_scores: ParameterScore[];
  rescued: boolean; summary: string;
}
export interface RescuedEntry {
  candidate_id: string; retrieval_rank: string; new_total_score: number;
  why_rescued: string; evidence: string[];
}
export interface RankedShortlist {
  mode: Mode;
  shortlist: CandidateScore[];
  rescued_report: { rescued: RescuedEntry[] };
  rubric: Rubric;
  stats: Record<string, unknown>;
}
export interface EvalResult {
  mode: Mode; total_candidates: number; advanced: number;
  rescued_count: number; shortlisted: number;
  planted_recall: number; planted_ids: string[];
  recovered_planted: string[]; explainability_ok: boolean;
  explainability_failures: string[];
}

export const api = {
  buildRubric: (jd: string) =>
    req<Rubric>("/rubric", { method: "POST", body: JSON.stringify({ job_description: jd }) }),

  updateRubric: (rubric: Rubric) =>
    req<{ status: string; rubric: Rubric }>("/rubric", {
      method: "PUT", body: JSON.stringify(rubric),
    }),

  rank: (jd: string, mode: Mode, rubric?: Rubric) =>
    req<RankedShortlist>("/rank", {
      method: "POST",
      body: JSON.stringify({ job_description: jd, mode, rubric: rubric || null }),
    }),

  runEval: (modes: Mode[]) =>
    req<Record<Mode, EvalResult>>("/eval", {
      method: "POST", body: JSON.stringify({ modes }),
    }),

  getCandidates: () =>
    req<{ total: number; candidates: { id: string; name: string; title: string; skills: string[] }[] }>("/candidates"),
};
