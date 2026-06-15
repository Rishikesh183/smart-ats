"use client";
import { useState } from "react";
import type { Rubric, RubricParameter } from "@/lib/api";

interface Props {
  rubric: Rubric;
  onSave: (r: Rubric) => void;
  onClose: () => void;
}

export default function RubricEditor({ rubric, onSave, onClose }: Props) {
  const [draft, setDraft] = useState<Rubric>(JSON.parse(JSON.stringify(rubric)));

  const updateParam = (idx: number, field: keyof RubricParameter, val: string | number) => {
    setDraft(d => {
      const params = [...d.parameters];
      params[idx] = { ...params[idx], [field]: val };
      return { ...d, parameters: params };
    });
  };

  const weightTotal = draft.parameters.reduce((s, p) => s + Number(p.weight), 0);
  const weightOk = Math.abs(weightTotal - 1.0) < 0.01;

  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-start justify-center pt-10 overflow-auto">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-3xl p-6 mx-4 mb-10">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-slate-800">Edit Rubric</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600 text-2xl">×</button>
        </div>

        {/* Role summary */}
        <div className="mb-5">
          <label className="block text-sm font-medium text-slate-600 mb-1">Role Title</label>
          <input
            className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm"
            value={draft.role_title}
            onChange={e => setDraft(d => ({ ...d, role_title: e.target.value }))}
          />
          <label className="block text-sm font-medium text-slate-600 mb-1 mt-3">Summary</label>
          <textarea
            className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm resize-none"
            rows={2}
            value={draft.role_summary}
            onChange={e => setDraft(d => ({ ...d, role_summary: e.target.value }))}
          />
        </div>

        {/* Parameters */}
        <h3 className="font-semibold text-slate-700 mb-3">Scoring Parameters</h3>
        <div className="space-y-4 mb-4">
          {draft.parameters.map((p, i) => (
            <div key={p.id} className="border border-slate-200 rounded-xl p-4">
              <div className="flex gap-3 mb-2">
                <div className="flex-1">
                  <label className="block text-xs text-slate-500 mb-1">Label</label>
                  <input
                    className="w-full border border-slate-200 rounded px-2 py-1 text-sm"
                    value={p.label}
                    onChange={e => updateParam(i, "label", e.target.value)}
                  />
                </div>
                <div className="w-24">
                  <label className="block text-xs text-slate-500 mb-1">Weight</label>
                  <input
                    type="number" step="0.01" min="0" max="1"
                    className="w-full border border-slate-200 rounded px-2 py-1 text-sm"
                    value={p.weight}
                    onChange={e => updateParam(i, "weight", parseFloat(e.target.value) || 0)}
                  />
                </div>
                <div className="w-32">
                  <label className="block text-xs text-slate-500 mb-1">Kind</label>
                  <select
                    className="w-full border border-slate-200 rounded px-2 py-1 text-sm"
                    value={p.kind}
                    onChange={e => updateParam(i, "kind", e.target.value)}
                  >
                    <option value="critical">critical</option>
                    <option value="important">important</option>
                    <option value="nice_to_have">nice_to_have</option>
                  </select>
                </div>
              </div>
              <label className="block text-xs text-slate-500 mb-1">What counts</label>
              <textarea
                className="w-full border border-slate-200 rounded px-2 py-1 text-sm resize-none"
                rows={2}
                value={p.what_counts}
                onChange={e => updateParam(i, "what_counts", e.target.value)}
              />
            </div>
          ))}
        </div>

        {/* Weight check */}
        <div className={`text-sm mb-4 font-medium ${weightOk ? "text-green-600" : "text-red-500"}`}>
          Weight total: {weightTotal.toFixed(3)} {weightOk ? "✓" : "(must equal 1.0)"}
        </div>

        {/* Hard gates */}
        {draft.hard_gates.length > 0 && (
          <div className="mb-4">
            <h3 className="font-semibold text-slate-700 mb-2">Hard Gates</h3>
            {draft.hard_gates.map(g => (
              <div key={g.id} className="bg-red-50 border border-red-200 rounded-lg p-3 mb-2 text-sm">
                <span className="font-medium text-red-700">{g.id}:</span>{" "}
                <span className="text-red-600">{g.requirement}</span>
                <p className="text-xs text-slate-500 mt-1">{g.rationale}</p>
              </div>
            ))}
          </div>
        )}

        <div className="flex gap-3 justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg border border-slate-200 text-slate-600 text-sm hover:bg-slate-50"
          >
            Cancel
          </button>
          <button
            disabled={!weightOk}
            onClick={() => onSave(draft)}
            className="px-4 py-2 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 disabled:opacity-40"
          >
            Save & Use Rubric
          </button>
        </div>
      </div>
    </div>
  );
}
