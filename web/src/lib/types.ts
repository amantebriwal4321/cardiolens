export type InputKind = "number" | "binary" | "select";

export type InputField = {
  key: string;
  label: string;
  group: string;
  kind: InputKind;
  unit?: string;
  step?: number;
  help?: string;
  derived?: boolean;
  options?: string[];
  true_value?: string | number;
  false_value?: string | number;
  train_min?: number;
  train_max?: number;
  train_median?: number;
};

export type FeatureMap = Record<string, string | number>;

export type Sample = {
  id: string;
  source_row: number;
  pattern: string;
  title: string;
  vignette: string;
  features: FeatureMap;
  dataset_labels: Record<string, string>;
};

export type Schema = {
  disclaimer: string;
  anatomy_note: string;
  in_sample_note: string;
  selected_model: string;
  threshold: number;
  groups: { id: string; description: string }[];
  inputs: InputField[];
  samples: Sample[];
  targets: Record<
    string,
    { title: string; short: string; question: string; positive: string; negative: string }
  >;
};

export type Contribution = {
  feature: string;
  label: string;
  group: string;
  display: string;
  shap: number;
};

export type TargetEstimate = {
  title: string;
  short: string;
  probability: number;
  predicted_label: string;
  positive_label: string;
  negative_label: string;
  threshold: number;
  base_value: number;
  shap_sum: number;
  space: string;
  reconstruction_error: number;
  contributions: Contribution[];
};

export type Prediction = {
  features: FeatureMap;
  targets: Record<string, TargetEstimate>;
  groups: { id: string; description: string; shap: Record<string, number> }[];
  disclaimer: string;
};

export type ModelCard = {
  name: string;
  validation: {
    method: string;
    n_splits: number;
    threshold: number;
    final_fit: string;
  };
  selection_rule: string;
  ranking: { model: string; mean_roc_auc: number; mean_f1: number }[];
  label_audit: { mismatch_count: number; note: string };
  targets: Record<
    string,
    {
      title: string;
      oof_pooled: {
        accuracy: number;
        precision: number;
        recall: number;
        f1: number;
        roc_auc: number;
        brier: number;
      };
      cv_mean: Record<string, { mean: number; std: number }>;
      positive_rate: number;
      majority_class_accuracy: number;
      shap_space: string;
      top_shap: { label: string; group: string; mean_abs_shap: number }[];
    }
  >;
};

export type Bootstrap = {
  schema: Schema;
  model: ModelCard;
};

export type VesselId = "lad" | "lcx" | "rca";
