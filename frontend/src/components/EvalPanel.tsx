"use client";
import { useState } from "react";
import { api, type EvalResult, type Mode } from "@/lib/api";

export default function EvalPanel() {
  const [results, setResults] = useState<Record<string, EvalResult> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await api.runEval(["normal", "high", "extra_high"]);
      setResults(r as Record<string, EvalResult>);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const modeLabel: Record<string, string> = {
    normal: "Normal",
    high: "High Recall",
    extra_high: "Extra-High Recall",
  };
  const modeColor: Record<string, string> = {
    normal: "bg-slate-100 border-slate-200",
    high: "bg-blue-50 border-blue-200",
    extra_high: "bg-purple-50 border-purple-200",
  };

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="font-bold text-slate-800 text-lg">Eval Harness</h2>
          <p className="text-sm text-slate-500">Planted-candidate recall test across all modes</p>
        </div>
        <button
          onClick={run}
          disabled={loading}
          className="px-4 py-2 bg-purple-600 text-white text-sm font-medium rounded-lg hover:bg-purple-700 disabled:opacity-50"
        >
          {loading ? "Running…" : "Run Eval"}
        </button>
      </div>

      {error && <p className="text-red-500 text-sm mb-3">{error}</p>}

      {loading && (
        <div className="text-sm text-slate-500 animate-pulse py-8 text-center">
          Running pipeline in all three modes — this takes a few minutes…
        </div>
      )}

      {results && !loading && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {(["normal", "high", "extra_high"] as Mode[]).map(mode => {
            const r = results[mode];
            if (!r) return null;
            const pct = r.planted_recall != null ? Math.round(r.planted_recall * 100) : null;
            return (
              <div key={mode} className={`border rounded-xl p-4 ${modeColor[mode]}`}>
                <h3 className="font-semibold text-slate-700 mb-3">{modeLabel[mode]}</h3>
                <div className="text-4xl font-bold text-slate-800 mb-1">
                  {pct != null ? `${pct}%` : "—"}
                </div>
                <div className="text-xs text-slate-500 mb-3">planted recall</div>

                <div className="space-y-1 text-sm">
                  <Row label="Total candidates" value={r.total_candidates} />
                  <Row label="Advanced" value={r.advanced} />
                  <Row label="Rescued" value={r.rescued_count} />
                  <Row label="Shortlisted" value={r.shortlisted} />
                  <Row label="Explainability" value={r.explainability_ok ? "✓ OK" : "✗ FAIL"} />
                </div>

                {r.recovered_planted.length > 0 && (
                  <div className="mt-3">
                    <div className="text-xs text-slate-500 mb-1">Recovered planted:</div>
                    <div className="flex flex-wrap gap-1">
                      {r.recovered_planted.map(id => (
                        <span key={id} className="text-xs bg-green-100 text-green-700 px-1.5 py-0.5 rounded font-medium">{id}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between">
      <span className="text-slate-500">{label}</span>
      <span className="font-medium text-slate-700">{value}</span>
    </div>
  );
}
