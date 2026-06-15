"use client";
import { useState, useEffect } from "react";
import { api, type Mode, type Rubric, type RankedShortlist, type RescuedEntry } from "@/lib/api";
import CandidateCard from "@/components/CandidateCard";
import RubricEditor from "@/components/RubricEditor";
import EvalPanel from "@/components/EvalPanel";

const SAMPLE_JD = `Senior Full-Stack Engineer

We are building a high-traffic SaaS platform and need a Senior Full-Stack Engineer who can own features end-to-end.

Requirements:
- 4+ years of professional software engineering experience
- Strong proficiency in React and Next.js (or equivalent SSR framework)
- TypeScript throughout
- Node.js or Python backend experience
- Experience with PostgreSQL or similar relational database
- Track record of shipping production systems to real users
- Must be authorized to work in the United States

Nice to have:
- AWS or cloud platform experience
- Open source contributions
- Experience mentoring junior engineers
- Measurable performance improvements (Core Web Vitals, query optimization, etc.)`;

const MODE_INFO: Record<Mode, { label: string; desc: string; color: string; bg: string }> = {
  normal: {
    label: "Normal",
    desc: "Fastest. Narrow rescue band, no critic agent.",
    color: "text-slate-700",
    bg: "bg-slate-50 border-slate-200",
  },
  high: {
    label: "High Recall",
    desc: "Wider rescue band + critic re-scores borderline candidates.",
    color: "text-blue-700",
    bg: "bg-blue-50 border-blue-200",
  },
  extra_high: {
    label: "Extra-High Recall",
    desc: "Maximum recall. Largest rescue band + critic flags borderline finalists.",
    color: "text-purple-700",
    bg: "bg-purple-50 border-purple-200",
  },
};

type Step = "jd" | "rubric" | "results";

export default function Home() {
  const [step, setStep] = useState<Step>("jd");
  const [jd, setJd] = useState(SAMPLE_JD);
  const [mode, setMode] = useState<Mode>("normal");
  const [rubric, setRubric] = useState<Rubric | null>(null);
  const [editingRubric, setEditingRubric] = useState(false);
  const [result, setResult] = useState<RankedShortlist | null>(null);
  const [candidateMap, setCandidateMap] = useState<Record<string, { name: string; title: string }>>({});
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [showEval, setShowEval] = useState(false);
  const [activeTab, setActiveTab] = useState<"shortlist" | "rescued">("shortlist");

  // Load candidate metadata for display
  useEffect(() => {
    api.getCandidates().then(data => {
      const map: Record<string, { name: string; title: string }> = {};
      data.candidates.forEach(c => { map[c.id] = { name: c.name, title: c.title }; });
      setCandidateMap(map);
    }).catch(() => {});
  }, []);

  const buildRubric = async () => {
    if (!jd.trim()) return;
    setLoading(true);
    setLoadingMsg("Analyzing job description…");
    setError(null);
    try {
      const r = await api.buildRubric(jd);
      setRubric(r);
      setStep("rubric");
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const runRanking = async () => {
    setLoading(true);
    setLoadingMsg("Ranking candidates — this takes 1-2 minutes…");
    setError(null);
    try {
      const r = await api.rank(jd, mode, rubric || undefined);
      setResult(r);
      setStep("results");
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const saveRubric = async (edited: Rubric) => {
    try {
      await api.updateRubric(edited);
      setRubric(edited);
      setEditingRubric(false);
    } catch (e) {
      setError(String(e));
    }
  };

  const rescuedInShortlist = new Set(
    result?.shortlist.filter(s => s.rescued).map(s => s.candidate_id) || []
  );

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-40">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-slate-800">Smart ATS</h1>
            <p className="text-xs text-slate-500">Semantic candidate ranking · capability over keywords</p>
          </div>
          <button
            onClick={() => setShowEval(e => !e)}
            className="text-sm px-3 py-1.5 border border-slate-200 rounded-lg hover:bg-slate-50 text-slate-600"
          >
            {showEval ? "Hide Eval" : "📊 Eval Harness"}
          </button>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8 space-y-6">
        {/* Step indicator */}
        <div className="flex items-center gap-2 text-sm">
          {(["jd", "rubric", "results"] as Step[]).map((s, i) => (
            <div key={s} className="flex items-center gap-2">
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold
                ${step === s ? "bg-indigo-600 text-white" :
                  (["jd","rubric","results"].indexOf(step) > i) ? "bg-green-500 text-white" :
                  "bg-slate-200 text-slate-500"}`}>
                {i + 1}
              </div>
              <span className={step === s ? "font-semibold text-indigo-700" : "text-slate-500"}>
                {s === "jd" ? "Job Description" : s === "rubric" ? "Review Rubric" : "Ranked Shortlist"}
              </span>
              {i < 2 && <span className="text-slate-300">→</span>}
            </div>
          ))}
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700 text-sm">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Loading overlay */}
        {loading && (
          <div className="bg-white border border-indigo-200 rounded-2xl p-8 text-center">
            <div className="text-3xl mb-3 animate-bounce">⚡</div>
            <p className="font-medium text-slate-700">{loadingMsg}</p>
            <p className="text-sm text-slate-400 mt-1">LLM pipeline running…</p>
          </div>
        )}

        {/* Step 1: JD input */}
        {!loading && step === "jd" && (
          <div className="bg-white border border-slate-200 rounded-2xl p-6">
            <h2 className="text-lg font-bold text-slate-800 mb-1">Paste Job Description</h2>
            <p className="text-sm text-slate-500 mb-4">
              The AI will analyze what the role actually needs — not just extract keywords.
            </p>
            <textarea
              className="w-full border border-slate-200 rounded-xl px-4 py-3 text-sm font-mono resize-none focus:outline-none focus:ring-2 focus:ring-indigo-300"
              rows={14}
              value={jd}
              onChange={e => setJd(e.target.value)}
              placeholder="Paste your job description here…"
            />
            <div className="flex justify-end mt-4">
              <button
                onClick={buildRubric}
                disabled={!jd.trim() || loading}
                className="px-6 py-2.5 bg-indigo-600 text-white font-medium rounded-xl hover:bg-indigo-700 disabled:opacity-40 transition-colors"
              >
                Analyze JD →
              </button>
            </div>
          </div>
        )}

        {/* Step 2: Rubric review */}
        {!loading && step === "rubric" && rubric && (
          <div className="space-y-4">
            <div className="bg-white border border-slate-200 rounded-2xl p-6">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h2 className="text-lg font-bold text-slate-800">{rubric.role_title}</h2>
                  <p className="text-sm text-slate-500 mt-1">{rubric.role_summary}</p>
                </div>
                <button
                  onClick={() => setEditingRubric(true)}
                  className="text-sm px-3 py-1.5 border border-slate-200 rounded-lg hover:bg-slate-50 text-slate-600 shrink-0 ml-4"
                >
                  ✏️ Edit Rubric
                </button>
              </div>

              {/* Parameters */}
              <h3 className="font-semibold text-slate-700 mb-3 text-sm uppercase tracking-wide">Scoring Parameters</h3>
              <div className="space-y-2 mb-5">
                {rubric.parameters.map(p => (
                  <div key={p.id} className="flex items-center gap-3 p-3 bg-slate-50 rounded-xl">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-slate-700">{p.label}</span>
                        <span className={`text-xs px-1.5 py-0.5 rounded font-medium
                          ${p.kind === "critical" ? "bg-red-100 text-red-600" :
                            p.kind === "important" ? "bg-orange-100 text-orange-600" :
                            "bg-slate-100 text-slate-500"}`}>
                          {p.kind}
                        </span>
                      </div>
                      <p className="text-xs text-slate-400 mt-0.5">{p.what_counts.slice(0, 100)}{p.what_counts.length > 100 ? "…" : ""}</p>
                    </div>
                    <div className="text-sm font-bold text-indigo-600 w-12 text-right">
                      {Math.round(p.weight * 100)}%
                    </div>
                  </div>
                ))}
              </div>

              {/* Hard gates */}
              {rubric.hard_gates.length > 0 && (
                <>
                  <h3 className="font-semibold text-slate-700 mb-3 text-sm uppercase tracking-wide">Hard Gates</h3>
                  {rubric.hard_gates.map(g => (
                    <div key={g.id} className="flex gap-3 p-3 bg-red-50 border border-red-100 rounded-xl mb-2">
                      <span className="text-red-500 text-lg">🚫</span>
                      <div>
                        <span className="text-sm font-medium text-red-700">{g.requirement}</span>
                        <p className="text-xs text-slate-500 mt-0.5">{g.rationale}</p>
                      </div>
                    </div>
                  ))}
                </>
              )}
            </div>

            {/* Mode selector + rank */}
            <div className="bg-white border border-slate-200 rounded-2xl p-6">
              <h3 className="font-semibold text-slate-800 mb-3">Recall Mode</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-5">
                {(Object.entries(MODE_INFO) as [Mode, typeof MODE_INFO[Mode]][]).map(([m, info]) => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    className={`border-2 rounded-xl p-3 text-left transition-all
                      ${mode === m ? "border-indigo-400 bg-indigo-50" : `${info.bg} hover:border-slate-300`}`}
                  >
                    <div className={`font-semibold text-sm ${mode === m ? "text-indigo-700" : info.color}`}>
                      {info.label}
                    </div>
                    <div className="text-xs text-slate-500 mt-1">{info.desc}</div>
                  </button>
                ))}
              </div>
              <div className="flex gap-3">
                <button
                  onClick={() => setStep("jd")}
                  className="px-4 py-2 border border-slate-200 text-slate-600 text-sm rounded-xl hover:bg-slate-50"
                >
                  ← Back
                </button>
                <button
                  onClick={runRanking}
                  className="flex-1 py-2.5 bg-indigo-600 text-white font-medium rounded-xl hover:bg-indigo-700 transition-colors"
                >
                  Rank Candidates ({mode === "normal" ? "Fast" : mode === "high" ? "High Recall" : "Max Recall"}) →
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Results */}
        {!loading && step === "results" && result && (
          <div className="space-y-5">
            {/* Stats bar */}
            <div className="bg-white border border-slate-200 rounded-2xl p-5">
              <div className="flex flex-wrap items-center gap-4">
                <div className="flex-1">
                  <h2 className="text-lg font-bold text-slate-800">
                    {result.rubric.role_title}
                  </h2>
                  <p className="text-sm text-slate-500">
                    Mode: <strong>{MODE_INFO[result.mode].label}</strong> ·{" "}
                    {(result.stats as Record<string, number>).total_candidates ?? "?"} total candidates →{" "}
                    {result.shortlist.length} shortlisted
                    {result.rescued_report.rescued.length > 0 &&
                      ` · ${result.rescued_report.rescued.length} rescued`}
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => setEditingRubric(true)}
                    className="text-sm px-3 py-1.5 border border-slate-200 rounded-lg hover:bg-slate-50 text-slate-600"
                  >
                    ✏️ Rubric
                  </button>
                  <button
                    onClick={() => { setStep("rubric"); setResult(null); }}
                    className="text-sm px-3 py-1.5 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
                  >
                    Re-rank →
                  </button>
                  <button
                    onClick={() => { setStep("jd"); setResult(null); setRubric(null); }}
                    className="text-sm px-3 py-1.5 border border-slate-200 rounded-lg hover:bg-slate-50 text-slate-600"
                  >
                    New JD
                  </button>
                </div>
              </div>

              {/* Stat chips */}
              <div className="flex flex-wrap gap-2 mt-3 text-xs">
                {Object.entries(result.stats as Record<string, unknown>).map(([k, v]) => {
                  if (typeof v !== "number" || k === "timings") return null;
                  return (
                    <span key={k} className="bg-slate-100 text-slate-600 px-2 py-1 rounded-lg">
                      {k.replace(/_/g, " ")}: <strong>{v}</strong>
                    </span>
                  );
                })}
              </div>
            </div>

            {/* Tabs */}
            <div className="flex gap-2">
              <button
                onClick={() => setActiveTab("shortlist")}
                className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors
                  ${activeTab === "shortlist" ? "bg-indigo-600 text-white" : "bg-white border border-slate-200 text-slate-600 hover:bg-slate-50"}`}
              >
                Shortlist ({result.shortlist.length})
              </button>
              {result.rescued_report.rescued.length > 0 && (
                <button
                  onClick={() => setActiveTab("rescued")}
                  className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors
                    ${activeTab === "rescued" ? "bg-amber-500 text-white" : "bg-white border border-amber-200 text-amber-600 hover:bg-amber-50"}`}
                >
                  🔍 Rescued ({result.rescued_report.rescued.length})
                </button>
              )}
            </div>

            {/* Shortlist */}
            {activeTab === "shortlist" && (
              <div className="space-y-3">
                {result.shortlist.length === 0 && (
                  <div className="bg-white border border-slate-200 rounded-2xl p-8 text-center text-slate-500">
                    No candidates cleared all gates and scoring thresholds.
                  </div>
                )}
                {result.shortlist.map((score, i) => (
                  <CandidateCard
                    key={score.candidate_id}
                    score={score}
                    rank={i + 1}
                    candidateName={candidateMap[score.candidate_id]?.name}
                    candidateTitle={candidateMap[score.candidate_id]?.title}
                    rescued={rescuedInShortlist.has(score.candidate_id)}
                  />
                ))}
              </div>
            )}

            {/* Rescued panel */}
            {activeTab === "rescued" && (
              <div className="space-y-3">
                <div className="bg-amber-50 border border-amber-200 rounded-2xl p-4 text-sm text-amber-800">
                  <strong>Rescued candidates</strong> — these were below the initial retrieval cutoff but the critic agent found them genuinely strong. They would have been lost in Normal mode.
                </div>
                {result.rescued_report.rescued.map((entry: RescuedEntry) => (
                  <div key={entry.candidate_id} className="bg-white border border-amber-200 rounded-2xl p-5">
                    <div className="flex items-center justify-between mb-2">
                      <div>
                        <span className="font-semibold text-slate-800">
                          {candidateMap[entry.candidate_id]?.name || entry.candidate_id}
                        </span>
                        <span className="ml-2 text-sm text-slate-500">
                          {candidateMap[entry.candidate_id]?.title}
                        </span>
                      </div>
                      <div className="text-lg font-bold text-amber-600">
                        {entry.new_total_score}/100
                      </div>
                    </div>
                    <div className="text-xs text-slate-500 mb-2">
                      Retrieval rank: {entry.retrieval_rank}
                    </div>
                    <div className="bg-amber-50 rounded-xl p-3 text-sm text-slate-700 mb-3">
                      <strong>Why rescued:</strong> {entry.why_rescued}
                    </div>
                    {entry.evidence.length > 0 && (
                      <div className="space-y-1">
                        {entry.evidence.slice(0, 3).map((ev, i) => (
                          <div key={i} className="text-xs text-slate-600 bg-slate-50 rounded-lg px-3 py-1.5 italic border-l-2 border-amber-300">
                            "{ev}"
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Eval panel */}
        {showEval && (
          <div className="mt-6">
            <EvalPanel />
          </div>
        )}
      </main>

      {/* Rubric editor modal */}
      {editingRubric && rubric && (
        <RubricEditor
          rubric={rubric}
          onSave={saveRubric}
          onClose={() => setEditingRubric(false)}
        />
      )}
    </div>
  );
}
