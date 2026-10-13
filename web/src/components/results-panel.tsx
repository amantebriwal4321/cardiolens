"use client";

import { Badge } from "@/components/ui/badge";
import { formatMetric, formatPercent, likelihoodBand, riskCss } from "@/lib/risk";
import type { ModelCard, Prediction, TargetEstimate, VesselId } from "@/lib/types";

const TARGET_ORDER = ["cad", "lad", "lcx", "rca"] as const;

function DivergingBar({ value, max }: { value: number; max: number }) {
  const width = max > 0 ? (Math.abs(value) / max) * 50 : 0;
  const positive = value >= 0;
  return (
    <div className="relative h-2 rounded-full bg-white/5">
      <div className="absolute top-0 bottom-0 left-1/2 w-px bg-white/30" />
      <div
        className="absolute top-0 bottom-0 rounded-full"
        style={{
          left: positive ? "50%" : `${50 - width}%`,
          width: `${width}%`,
          background: positive ? "#d94e48" : "#2f9d84",
        }}
      />
    </div>
  );
}

function EstimateButton({
  id,
  estimate,
  active,
  onSelect,
}: {
  id: string;
  estimate: TargetEstimate;
  active: boolean;
  onSelect: () => void;
}) {
  const vessel = id === "lad" || id === "lcx" || id === "rca";
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={active}
      className={`w-full rounded-xl border px-3 py-3 text-left transition ${
        active ? "border-[#d7efe4]/50 bg-white/5" : "border-border bg-transparent hover:bg-white/5"
      }`}
    >
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-xs tracking-wide text-muted-foreground uppercase">{estimate.title}</p>
          <p className="mt-1 text-2xl font-medium tabular-nums" style={{ color: riskCss(estimate.probability) }}>
            {formatPercent(estimate.probability)}
          </p>
        </div>
        {vessel ? (
          <span
            className="size-3 rounded-full"
            style={{ background: riskCss(estimate.probability) }}
            aria-hidden
          />
        ) : (
          <Badge variant="outline">{estimate.predicted_label}</Badge>
        )}
      </div>
      <p className="mt-1 text-xs text-muted-foreground">{likelihoodBand(estimate.probability)}</p>
      <p className="mt-1 text-xs text-muted-foreground">
        Display label at 0.50: {estimate.predicted_label}. Not a diagnosis.
      </p>
    </button>
  );
}

export function ResultsPanel({
  prediction,
  model,
  activeTarget,
  onTarget,
  sampleLabels,
  labelsApply,
  inSampleNote,
}: {
  prediction: Prediction | null;
  model: ModelCard;
  activeTarget: "cad" | VesselId;
  onTarget: (target: "cad" | VesselId) => void;
  sampleLabels: Record<string, string> | null;
  labelsApply: boolean;
  inSampleNote: string;
}) {
  const estimate = prediction?.targets[activeTarget];
  const maxFeature = estimate
    ? Math.max(...estimate.contributions.map((item) => Math.abs(item.shap)), 0.001)
    : 1;
  const groupRows = prediction?.groups ?? [];
  const maxGroup = Math.max(...groupRows.map((group) => Math.abs(group.shap[activeTarget] ?? 0)), 0.001);
  const pushesUp = estimate?.contributions.filter((item) => item.shap > 0).slice(0, 3) ?? [];
  const pushesDown = estimate?.contributions.filter((item) => item.shap < 0).slice(0, 3) ?? [];

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-4">
        <div>
          <p className="text-[11px] font-medium tracking-[0.16em] text-[#8fb9aa] uppercase">Estimate</p>
          <h2 className="mt-1 font-[family-name:var(--font-display)] text-2xl">What the model sees</h2>
        </div>
        {prediction ? (
          <div className="grid gap-2">
            {TARGET_ORDER.map((id) => (
              <EstimateButton
                key={id}
                id={id}
                estimate={prediction.targets[id]}
                active={activeTarget === id}
                onSelect={() => onTarget(id)}
              />
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">Run an estimate to color the arteries and fill this panel.</p>
        )}

        {labelsApply && sampleLabels ? (
          <div className="rounded-xl border border-border px-3 py-3 text-sm">
            <p className="text-xs tracking-wide text-muted-foreground uppercase">Source angiography labels</p>
            <p className="mt-1">
              Cath {sampleLabels.cad}, LAD {sampleLabels.lad}, LCX {sampleLabels.lcx}, RCA {sampleLabels.rca}.
            </p>
            <p className="mt-2 text-xs leading-5 text-muted-foreground">{inSampleNote}</p>
          </div>
        ) : null}
        {!labelsApply && sampleLabels ? (
          <p className="text-xs leading-5 text-muted-foreground">
            Inputs no longer match the source row, so the angiography labels are hidden.
          </p>
        ) : null}

        {estimate ? (
          <section className="space-y-3">
            <div>
              <h3 className="text-sm font-medium">Why the {estimate.title} estimate moved</h3>
              <p className="mt-1 text-xs leading-5 text-muted-foreground">
                {estimate.space === "probability"
                  ? `SHAP values are in probability units. They start at ${formatPercent(estimate.base_value)} and sum to ${formatPercent(estimate.probability)}.`
                  : `SHAP values are in log-odds. Positive values push toward ${estimate.positive_label}.`}
              </p>
            </div>
            <ul className="space-y-2 text-sm">
              {pushesUp.map((item) => (
                <li key={item.feature}>
                  {item.label} ({item.display}) pushes toward {estimate.positive_label}.
                </li>
              ))}
              {pushesDown.map((item) => (
                <li key={item.feature}>
                  {item.label} ({item.display}) pushes toward {estimate.negative_label}.
                </li>
              ))}
            </ul>
            <div className="space-y-3">
              <h3 className="text-sm font-medium">Physiological groups</h3>
              {groupRows.map((group) => {
                const members = estimate.contributions.filter((item) => item.group === group.id);
                const total = group.shap[activeTarget] ?? 0;
                return (
                  <details key={`${group.id}-${activeTarget}`} className="rounded-xl border border-border px-3 py-2">
                    <summary className="cursor-pointer list-none">
                      <div className="flex items-baseline justify-between gap-3">
                        <span className="text-sm font-medium">{group.id}</span>
                        <span className="text-xs tabular-nums text-muted-foreground">
                          {total >= 0 ? "+" : ""}
                          {total.toFixed(3)}
                        </span>
                      </div>
                      <p className="mt-0.5 text-[11px] text-muted-foreground">{group.description}</p>
                      <div className="mt-2">
                        <DivergingBar value={total} max={maxGroup} />
                      </div>
                    </summary>
                    <ul className="mt-3 space-y-2">
                      {members.slice(0, 8).map((item) => (
                        <li key={item.feature} className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 text-xs">
                          <span>{item.label}</span>
                          <span className="tabular-nums text-muted-foreground">{item.display}</span>
                          <div className="col-span-2">
                            <DivergingBar value={item.shap} max={maxFeature} />
                          </div>
                        </li>
                      ))}
                    </ul>
                  </details>
                );
              })}
              <p className="text-[11px] leading-4 text-muted-foreground">
                Crimson bars push toward disease. Teal bars push toward the normal label. The sign is the model&apos;s, not a clinical rule.
              </p>
            </div>
          </section>
        ) : null}

        <ModelCardView model={model} />
      </div>
    </div>
  );
}

function ModelCardView({ model }: { model: ModelCard }) {
  return (
    <details className="rounded-xl border border-border px-3 py-3">
      <summary className="cursor-pointer text-sm font-medium">Cross-validation card</summary>
      <div className="mt-3 space-y-3 text-xs leading-5 text-muted-foreground">
        <p>
          Deployed model: {model.name}. {model.validation.final_fit} Metrics below are stratified {model.validation.n_splits}-fold
          out-of-fold scores on all 303 rows, threshold {model.validation.threshold.toFixed(2)}.
        </p>
        <p>{model.selection_rule}</p>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="text-[#d7efe4]">
                <th className="py-1 pr-2 font-medium">Target</th>
                <th className="py-1 pr-2 font-medium">Acc</th>
                <th className="py-1 pr-2 font-medium">F1</th>
                <th className="py-1 pr-2 font-medium">AUC</th>
                <th className="py-1 font-medium">Majority acc</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(model.targets).map(([id, target]) => (
                <tr key={id} className="border-t border-border">
                  <td className="py-1 pr-2 text-foreground">{target.title}</td>
                  <td className="py-1 pr-2 tabular-nums">{formatMetric(target.oof_pooled.accuracy)}</td>
                  <td className="py-1 pr-2 tabular-nums">{formatMetric(target.oof_pooled.f1)}</td>
                  <td className="py-1 pr-2 tabular-nums">{formatMetric(target.oof_pooled.roc_auc)}</td>
                  <td className="py-1 tabular-nums">{formatMetric(target.majority_class_accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>
          Mean ROC-AUC across targets:{" "}
          {model.ranking.map((row) => `${row.model} ${row.mean_roc_auc.toFixed(3)}`).join(", ")}.
        </p>
        <p>
          Label audit: {model.label_audit.mismatch_count} row where Cath does not match the OR of the three vessel labels.{" "}
          {model.label_audit.note}
        </p>
      </div>
    </details>
  );
}
