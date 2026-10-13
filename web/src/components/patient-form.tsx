"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { FeatureMap, InputField, Sample, Schema } from "@/lib/types";

function FieldControl({
  id,
  field,
  value,
  onChange,
}: {
  id: string;
  field: InputField;
  value: string | number | undefined;
  onChange: (value: string | number) => void;
}) {
  if (field.kind === "binary") {
    const yes = field.true_value ?? 1;
    const no = field.false_value ?? 0;
    return (
      <div className="grid grid-cols-2 gap-1">
        <Button
          type="button"
          size="sm"
          variant={value === yes ? "default" : "outline"}
          aria-pressed={value === yes}
          onClick={() => onChange(yes)}
        >
          Yes
        </Button>
        <Button
          type="button"
          size="sm"
          variant={value === no ? "default" : "outline"}
          aria-pressed={value === no}
          onClick={() => onChange(no)}
        >
          No
        </Button>
      </div>
    );
  }

  if (field.kind === "select") {
    const current = value === undefined ? "" : String(value);
    return (
      <Select
        value={current}
        onValueChange={(next) => {
          if (next == null) return;
          onChange(field.key === "Function Class" || field.key === "Region RWMA" ? Number(next) : next);
        }}
      >
        <SelectTrigger id={id} className="w-full" aria-label={field.label}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {(field.options ?? []).map((option) => (
            <SelectItem key={option} value={option}>
              {option === "N" ? "None" : option}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    );
  }

  const outside =
    typeof field.train_min === "number" &&
    typeof field.train_max === "number" &&
    value !== undefined &&
    value !== "" &&
    (Number(value) < field.train_min || Number(value) > field.train_max);

  return (
    <div className="space-y-1">
      <Input
        type="number"
        inputMode="decimal"
        step={field.step ?? 1}
        id={id}
        aria-label={field.label}
        value={value === undefined || value === "" ? "" : String(value)}
        onChange={(event) => {
          const next = event.target.value;
          onChange(next === "" ? "" : Number(next));
        }}
      />
      {outside ? (
        <p className="text-[11px] text-[#e6b15a]">
          Outside the training range ({field.train_min} to {field.train_max}).
        </p>
      ) : null}
    </div>
  );
}

export function PatientForm({
  schema,
  features,
  activeSample,
  dirty,
  predicting,
  onChange,
  onSample,
  onSubmit,
}: {
  schema: Schema;
  features: FeatureMap;
  activeSample: Sample | null;
  dirty: boolean;
  predicting: boolean;
  onChange: (key: string, value: string | number) => void;
  onSample: (sample: Sample) => void;
  onSubmit: () => void;
}) {
  const grouped = schema.groups.map((group) => ({
    ...group,
    fields: schema.inputs.filter((field) => field.group === group.id),
  }));

  return (
    <form
      className="flex h-full flex-col"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <div className="border-b border-border px-4 py-3">
        <p className="text-[11px] font-medium tracking-[0.16em] text-[#8fb9aa] uppercase">
          Cohort rows
        </p>
        <div className="mt-2 flex gap-2 overflow-x-auto pb-1">
          {schema.samples.map((sample) => (
            <Button
              key={sample.id}
              type="button"
              size="sm"
              variant={activeSample?.id === sample.id && !dirty ? "default" : "outline"}
              className="shrink-0"
              onClick={() => onSample(sample)}
            >
              {sample.id} · {sample.title}
            </Button>
          ))}
        </div>
        {activeSample ? (
          <p className="mt-2 text-sm leading-5 text-muted-foreground">{activeSample.vignette}</p>
        ) : null}
      </div>
      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-4">
        {grouped.map((group) => (
          <section key={group.id} aria-labelledby={`group-${group.id}`}>
            <div className="mb-2">
              <h3 id={`group-${group.id}`} className="text-sm font-medium">
                {group.id}
              </h3>
              <p className="text-xs text-muted-foreground">{group.description}</p>
            </div>
            <div className="space-y-3">
              {group.fields.map((field) => (
                <div key={field.key} className="grid gap-1.5">
                  <Label htmlFor={field.key}>
                    {field.label}
                    {field.unit ? (
                      <span className="font-normal text-muted-foreground">{field.unit}</span>
                    ) : null}
                    {field.derived ? (
                      <span className="font-normal text-muted-foreground">from height and weight</span>
                    ) : null}
                  </Label>
                  <FieldControl
                    id={field.key}
                    field={field}
                    value={features[field.key]}
                    onChange={(value) => onChange(field.key, value)}
                  />
                  {field.help && (field.key === "Length" || field.key === "BP" || field.key === "Region RWMA" || field.key === "BBB" || field.key === "VHD") ? (
                    <p className="text-[11px] leading-4 text-muted-foreground">{field.help}</p>
                  ) : null}
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>
      <div className="border-t border-border px-4 py-3">
        <Button type="submit" className="w-full" disabled={predicting}>
          {predicting ? "Updating estimate…" : dirty ? "Update estimate" : "Run estimate again"}
        </Button>
        <p className="mt-2 text-[11px] leading-4 text-muted-foreground">
          LAD, LCX, RCA, and Cath are not inputs. The service rejects them.
        </p>
      </div>
    </form>
  );
}
