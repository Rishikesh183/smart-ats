"use client";
import { useState } from "react";
import type { CandidateScore } from "@/lib/api";

const confidenceColor = {
  high: "bg-green-100 text-green-700",
  medium: "bg-yellow-100 text-yellow-700",
  low: "bg-red-100 text-red-700",
};

interface Props {
  score: CandidateScore;
  rank: number;
  candidateName?: string;
  candidateTitle?: string;
  rescued?: boolean;
}

export default function CandidateCard({ score, rank, candidateName, candidateTitle, rescued }: Props) {
  const [open, setOpen] = useState(false);

  const scoreColor =
    score.total_score >= 75 ? "text-green-600 bg-green-50" :
    score.total_score >= 55 ? "text-yellow-600 bg-yellow-50" :
    "text-red-600 bg-red-50";

  return (
    <div className={`border rounded-2xl overflow-hidden transition-shadow hover:shadow-md ${rescued ? "border-amber-300 bg-amber-50/30" : "border-slate-200 bg-white"}`}>
      {/* Header */}
      <div
        className="flex items-center gap-4 px-5 py-4 cursor-pointer"
        onClick={() => setOpen(o => !o)}
      >
        {/* Rank */}
        <div className="text-2xl font-bold text-slate-300 w-8 shrink-0">#{rank}</div>

        {/* Name + title */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-800 truncate">
              {candidateName || score.candidate_id}
            </span>
            {rescued && (
              <span className="shrink-0 text-xs font-medium bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full">
                🔍 Rescued
              </span>
            )}
          </div>
          <div className="text-sm text-slate-500 truncate">{candidateTitle || ""}</div>
          <div className="text-xs text-slate-400 mt-0.5 truncate">{score.summary}</div>
        </div>

        {/* Score */}
        <div className={`shrink-0 text-lg font-bold px-3 py-1 rounded-xl ${scoreColor}`}>
          {Math.round(score.total_score)}
        </div>

        {/* Expand arrow */}
        <div className="text-slate-400 text-lg">{open ? "▲" : "▼"}</div>
      </div>

      {/* Score bar */}
      <div className="px-5 pb-3">
        <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${score.total_score >= 75 ? "bg-green-400" : score.total_score >= 55 ? "bg-yellow-400" : "bg-red-400"}`}
            style={{ width: `${score.total_score}%` }}
          />
        </div>
      </div>

      {/* Expanded: parameter breakdown */}
      {open && (
        <div className="px-5 pb-5 border-t border-slate-100 pt-4 space-y-4">
          {score.parameter_scores.map(ps => (
            <div key={ps.parameter_id}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium text-slate-700 capitalize">
                  {ps.parameter_id.replace(/_/g, " ")}
                </span>
                <div className="flex items-center gap-2">
                  <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${confidenceColor[ps.confidence]}`}>
                    {ps.confidence}
                  </span>
                  <span className="text-sm font-semibold text-slate-700">
                    {ps.score}/{ps.max}
                  </span>
                </div>
              </div>

              {/* Mini bar */}
              <div className="h-1 bg-slate-100 rounded-full mb-2">
                <div
                  className="h-full bg-indigo-400 rounded-full"
                  style={{ width: `${(ps.score / ps.max) * 100}%` }}
                />
              </div>

              {/* Evidence */}
              {ps.evidence.filter(e => e !== "[No direct evidence found in profile]").length > 0 && (
                <div className="space-y-1">
                  {ps.evidence.filter(e => e !== "[No direct evidence found in profile]").slice(0, 3).map((ev, i) => (
                    <div key={i} className="text-xs text-slate-600 bg-slate-50 rounded-lg px-3 py-1.5 italic border-l-2 border-indigo-300">
                      "{ev}"
                    </div>
                  ))}
                </div>
              )}

              {/* Reasoning */}
              {ps.reasoning && (
                <p className="text-xs text-slate-500 mt-1.5">{ps.reasoning}</p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
