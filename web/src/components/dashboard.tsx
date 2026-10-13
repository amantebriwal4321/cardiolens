"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";

import { PatientForm } from "@/components/patient-form";
import { ResultsPanel } from "@/components/results-panel";
import { Button } from "@/components/ui/button";
import { formatPercent, likelihoodBand, riskCss } from "@/lib/risk";
import type { Bootstrap, FeatureMap, InputField, Prediction, Sample, VesselId } from "@/lib/types";

const HeartView = dynamic(() => import("@/components/heart-view"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
      Preparing the 3D view…
    </div>
  ),
});

const VESSELS: { id: VesselId; name: string }[] = [
  { id: "lad", name: "LAD" },
  { id: "lcx", name: "LCX" },
  { id: "rca", name: "RCA" },
];

function presentFeatures(features: FeatureMap, inputs: InputField[]): FeatureMap {
  const next: FeatureMap = { ...features };
  for (const field of inputs) {
    if (field.kind !== "number") continue;
    const number = Number(next[field.key]);
    if (!Number.isFinite(number)) continue;
    const step = field.step ?? 1;
    next[field.key] = step >= 1 ? Math.round(number) : Math.round(number * 100) / 100;
  }
  return next;
}

function sameFeatures(left: FeatureMap, right: FeatureMap): boolean {
  const keys = new Set([...Object.keys(left), ...Object.keys(right)]);
  for (const key of keys) {
    if (String(left[key]) !== String(right[key])) return false;
  }
  return true;
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    if (typeof payload?.detail === "string") return payload.detail;
  } catch {
    /* body was not JSON */
  }
  return `Request failed (${response.status}).`;
}

export function Dashboard() {
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);
  const [features, setFeatures] = useState<FeatureMap>({});
  const [baseline, setBaseline] = useState<FeatureMap>({});
  const [sample, setSample] = useState<Sample | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [activeTarget, setActiveTarget] = useState<"cad" | VesselId>("cad");
  const [resetToken, setResetToken] = useState(0);

  const run = useCallback(async (next: FeatureMap) => {
    setPredicting(true);
    setPredictError(null);
    try {
      const response = await fetch("/api/backend/predict", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ features: next }),
      });
      if (!response.ok) throw new Error(await readError(response));
      const payload = (await response.json()) as Prediction;
      setPrediction(payload);
    } catch (error) {
      setPredictError(error instanceof Error ? error.message : "Could not reach the model.");
    } finally {
      setPredicting(false);
    }
  }, []);

  useEffect(() => {
    let cancel = false;
    fetch("/api/backend/bootstrap")
      .then(async (response) => {
        if (!response.ok) throw new Error(await readError(response));
        return response.json() as Promise<Bootstrap>;
      })
      .then((payload) => {
        if (cancel) return;
        const first = payload.schema.samples[0];
        const presented = presentFeatures(first.features, payload.schema.inputs);
        setBootstrap(payload);
        setSample(first);
        setFeatures(presented);
        setBaseline(presented);
        void run(presented);
      })
      .catch((error: Error) => {
        if (!cancel) setBootError(error.message);
      });
    return () => {
      cancel = true;
    };
  }, [run]);

  const dirty = useMemo(() => !sameFeatures(features, baseline), [features, baseline]);

  const probabilities = useMemo(() => {
    if (!prediction) return {};
    return {
      lad: prediction.targets.lad.probability,
      lcx: prediction.targets.lcx.probability,
      rca: prediction.targets.rca.probability,
    };
  }, [prediction]);

  function updateFeature(key: string, value: string | number) {
    setFeatures((current) => {
      const next = { ...current, [key]: value };
      if (key === "Weight" || key === "Length") {
        const weight = Number(next.Weight);
        const heightM = Number(next.Length) / 100;
        if (weight > 0 && heightM > 0) {
          next.BMI = Math.round((weight / (heightM * heightM)) * 100) / 100;
        }
      }
      return next;
    });
  }

  function chooseSample(next: Sample) {
    if (!bootstrap) return;
    const presented = presentFeatures(next.features, bootstrap.schema.inputs);
    setSample(next);
    setFeatures(presented);
    setBaseline(presented);
    setActiveTarget("cad");
    void run(presented);
  }

  function chooseTarget(target: "cad" | VesselId) {
    setActiveTarget(target);
  }

  if (bootError) {
    return (
      <div className="mx-auto max-w-lg px-6 py-16">
        <h1 className="font-[family-name:var(--font-display)] text-3xl">CardioLens</h1>
        <p className="mt-3 text-sm leading-6 text-[#e7b6b2]">{bootError}</p>
      </div>
    );
  }

  if (!bootstrap) {
    return (
      <div className="flex min-h-screen items-center justify-center text-sm text-muted-foreground">
        Loading the Z-Alizadeh cohort schema…
      </div>
    );
  }

  const selectedVessel = activeTarget === "cad" ? null : activeTarget;
  const focus = selectedVessel ? bootstrap.schema.targets[selectedVessel] : null;
  const focusEstimate = selectedVessel ? prediction?.targets[selectedVessel] : null;

  return (
    <div className="flex min-h-screen flex-col lg:h-dvh lg:overflow-hidden">
      <header className="border-b border-border px-4 py-3 lg:px-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="text-[11px] font-medium tracking-[0.18em] text-[#8fb9aa] uppercase">
              Track A · educational viewer
            </p>
            <h1 className="font-[family-name:var(--font-display)] text-3xl leading-none">CardioLens</h1>
          </div>
          <p className="max-w-xl text-xs leading-5 text-muted-foreground">
            Random forest estimates for overall CAD and LAD, LCX, and RCA stenosis. Built by Aman Tebriwal.
          </p>
        </div>
      </header>
      <div
        role="note"
        className="border-b border-[#e6b15a]/50 bg-[#3a2a12] px-4 py-2.5 text-sm leading-5 text-[#f6e7c1] lg:px-6"
      >
        {bootstrap.schema.disclaimer} Color on an artery is the model probability of stenosis at or above 50% diameter
        narrowing. It is not a measured narrowing and not a cath report.
      </div>
      {predictError ? (
        <div className="bg-[#3a1c1c] px-4 py-2 text-sm text-[#f0c7c4] lg:px-6">{predictError}</div>
      ) : null}
      <main className="grid flex-1 lg:min-h-0 lg:grid-cols-[minmax(300px,380px)_minmax(0,1fr)_minmax(320px,420px)] lg:overflow-hidden">
        <section className="cardiolens-form order-3 min-h-[70vh] border-b border-border lg:order-1 lg:min-h-0 lg:overflow-hidden lg:border-r lg:border-b-0">
          <PatientForm
            schema={bootstrap.schema}
            features={features}
            activeSample={sample}
            dirty={dirty}
            predicting={predicting}
            onChange={updateFeature}
            onSample={chooseSample}
            onSubmit={() => void run(features)}
          />
        </section>
        <section className="cardiolens-heart relative order-1 h-[560px] border-b border-border lg:order-2 lg:h-full lg:min-h-0 lg:overflow-hidden lg:border-b-0">
          <div className="absolute inset-0">
          <HeartView
            probabilities={probabilities}
            selected={selectedVessel}
            onSelect={chooseTarget}
            resetToken={resetToken}
          />
          </div>
          <div className="pointer-events-none absolute top-3 right-3 left-3 flex items-start justify-between gap-3">
            <div className="pointer-events-auto rounded-xl border border-white/10 bg-[#101614]/80 px-3 py-2 text-[11px] leading-4 text-[#d5e4dc] backdrop-blur">
              <p>Drag to rotate. Scroll or pinch to zoom. Click a vessel.</p>
              <div className="mt-2 flex items-center gap-2">
                <span className="h-1.5 w-16 rounded-full" style={{ background: "linear-gradient(90deg, #2f9d84, #e6b15a, #d94e48)" }} />
                <span>0 to 100% stenosis probability</span>
              </div>
            </div>
            <Button type="button" size="sm" variant="outline" className="pointer-events-auto" onClick={() => setResetToken((value) => value + 1)}>
              Front view
            </Button>
          </div>
          <div className="pointer-events-none absolute right-3 bottom-3 left-3">
            <div className="pointer-events-auto rounded-xl border border-white/10 bg-[#101614]/85 px-3 py-3 backdrop-blur">
              {focus && focusEstimate ? (
                <>
                  <p className="text-xs tracking-wide text-[#8fb9aa] uppercase">{focus.short}</p>
                  <p className="mt-1 text-lg" style={{ color: riskCss(focusEstimate.probability) }}>
                    {focus.title}: {formatPercent(focusEstimate.probability)}
                  </p>
                  <p className="text-xs text-muted-foreground">{likelihoodBand(focusEstimate.probability)}</p>
                  <p className="mt-1 text-sm text-[#d5e4dc]">{vesselBlurb(focus.short)}</p>
                </>
              ) : (
                <div className="flex flex-wrap gap-3">
                  {VESSELS.map((vessel) => {
                    const probability = probabilities[vessel.id];
                    return (
                      <button
                        key={vessel.id}
                        type="button"
                        className="text-left"
                        onClick={() => chooseTarget(vessel.id)}
                      >
                        <span className="text-xs text-muted-foreground">{vessel.name}</span>
                        <span className="mt-0.5 block text-lg tabular-nums" style={{ color: typeof probability === "number" ? riskCss(probability) : "#7f948c" }}>
                          {typeof probability === "number" ? formatPercent(probability) : "n/a"}
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
              <p className="mt-2 text-[11px] leading-4 text-muted-foreground">{bootstrap.schema.anatomy_note}</p>
            </div>
          </div>
        </section>
        <section className="cardiolens-results order-2 min-h-[70vh] lg:order-3 lg:min-h-0 lg:overflow-hidden">
          <ResultsPanel
            prediction={prediction}
            model={bootstrap.model}
            activeTarget={activeTarget}
            onTarget={chooseTarget}
            sampleLabels={sample?.dataset_labels ?? null}
            labelsApply={!dirty && Boolean(sample)}
            inSampleNote={bootstrap.schema.in_sample_note}
          />
        </section>
      </main>
    </div>
  );
}

function vesselBlurb(short: string): string {
  if (short === "LAD") {
    return "Anterior interventricular groove, down the front of the heart toward the apex.";
  }
  if (short === "LCX") {
    return "Left atrioventricular groove, toward the lateral wall and the back.";
  }
  return "Right atrioventricular groove, along the right side and toward the underside.";
}
