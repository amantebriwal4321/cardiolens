"""Feature schema and encoders for the Z-Alizadeh Sani extension.

LAD, LCX, RCA, and Cath are targets. They are never encoded as inputs.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

LEAKAGE_COLUMNS = ("LAD", "LCX", "RCA", "Cath")
DROPPED_CONSTANT_COLUMNS = ("Exertional CP",)

TARGET_SPECS: dict[str, dict[str, str]] = {
    "cad": {
        "column": "Cath",
        "positive": "CAD",
        "negative": "Normal",
        "title": "Overall CAD",
        "short": "CAD",
        "question": "Probability the source angiography label would be CAD",
    },
    "lad": {
        "column": "LAD",
        "positive": "Stenotic",
        "negative": "Normal",
        "title": "LAD stenosis",
        "short": "LAD",
        "question": "Probability of significant LAD stenosis",
    },
    "lcx": {
        "column": "LCX",
        "positive": "Stenotic",
        "negative": "Normal",
        "title": "LCX stenosis",
        "short": "LCX",
        "question": "Probability of significant LCX stenosis",
    },
    "rca": {
        "column": "RCA",
        "positive": "Stenotic",
        "negative": "Normal",
        "title": "RCA stenosis",
        "short": "RCA",
        "question": "Probability of significant RCA stenosis",
    },
}

VHD_SCORE = {"N": 0, "mild": 1, "Moderate": 2, "Severe": 3}
BBB_VALUES = ("N", "LBBB", "RBBB")

# kind: number | binary | select
# Binary numeric fields use 0/1. Binary yes/no fields use Y/N.
INPUTS: list[dict[str, Any]] = [
    {"key": "Age", "label": "Age", "group": "Demographics", "kind": "number", "unit": "years", "step": 1, "help": "Age in years."},
    {"key": "Sex", "label": "Sex", "group": "Demographics", "kind": "select", "options": ["Female", "Male"], "help": "The source file spells female as Fmale. CardioLens stores Female."},
    {"key": "Weight", "label": "Weight", "group": "Demographics", "kind": "number", "unit": "kg", "step": 1, "help": "Body weight."},
    {"key": "Length", "label": "Height", "group": "Demographics", "kind": "number", "unit": "cm", "step": 1, "help": "The source column is named Length. It is height in centimeters."},
    {"key": "BMI", "label": "BMI", "group": "Demographics", "kind": "number", "unit": "kg/m2", "step": 0.1, "derived": True, "help": "Body mass index. The form recalculates it from height and weight when either changes."},
    {"key": "DM", "label": "Diabetes mellitus", "group": "History", "kind": "binary", "true_value": 1, "false_value": 0, "help": "Recorded history of diabetes."},
    {"key": "HTN", "label": "Hypertension", "group": "History", "kind": "binary", "true_value": 1, "false_value": 0, "help": "Recorded history of hypertension."},
    {"key": "Current Smoker", "label": "Current smoker", "group": "History", "kind": "binary", "true_value": 1, "false_value": 0, "help": "Smokes at the time of the record."},
    {"key": "EX-Smoker", "label": "Former smoker", "group": "History", "kind": "binary", "true_value": 1, "false_value": 0, "help": "Source column EX-Smoker."},
    {"key": "FH", "label": "Family history", "group": "History", "kind": "binary", "true_value": 1, "false_value": 0, "help": "Family history of coronary disease, as recorded."},
    {"key": "Obesity", "label": "Obesity", "group": "History", "kind": "binary", "true_value": "Y", "false_value": "N", "help": "Obesity flag from the source record."},
    {"key": "CRF", "label": "Chronic renal failure", "group": "History", "kind": "binary", "true_value": "Y", "false_value": "N", "help": "Chronic renal failure."},
    {"key": "CVA", "label": "Prior CVA", "group": "History", "kind": "binary", "true_value": "Y", "false_value": "N", "help": "Prior cerebrovascular accident."},
    {"key": "Airway disease", "label": "Airway disease", "group": "History", "kind": "binary", "true_value": "Y", "false_value": "N", "help": "Recorded airway disease."},
    {"key": "Thyroid Disease", "label": "Thyroid disease", "group": "History", "kind": "binary", "true_value": "Y", "false_value": "N", "help": "Recorded thyroid disease."},
    {"key": "CHF", "label": "Heart failure", "group": "History", "kind": "binary", "true_value": "Y", "false_value": "N", "help": "Congestive heart failure."},
    {"key": "DLP", "label": "Dyslipidemia", "group": "History", "kind": "binary", "true_value": "Y", "false_value": "N", "help": "Dyslipidemia flag."},
    {"key": "Typical Chest Pain", "label": "Typical chest pain", "group": "Symptoms", "kind": "binary", "true_value": 1, "false_value": 0, "help": "Typical anginal chest pain."},
    {"key": "Dyspnea", "label": "Dyspnea", "group": "Symptoms", "kind": "binary", "true_value": "Y", "false_value": "N", "help": "Shortness of breath."},
    {"key": "Function Class", "label": "Function class", "group": "Symptoms", "kind": "select", "options": ["0", "1", "2", "3"], "help": "Functional class as coded in the source file, 0 through 3."},
    {"key": "Atypical", "label": "Atypical chest pain", "group": "Symptoms", "kind": "binary", "true_value": "Y", "false_value": "N", "help": "Atypical chest pain."},
    {"key": "Nonanginal", "label": "Nonanginal pain", "group": "Symptoms", "kind": "binary", "true_value": "Y", "false_value": "N", "help": "Nonanginal chest pain."},
    {"key": "LowTH Ang", "label": "Low-threshold angina", "group": "Symptoms", "kind": "binary", "true_value": "Y", "false_value": "N", "help": "Source column LowTH Ang."},
    {"key": "BP", "label": "Blood pressure", "group": "Exam", "kind": "number", "unit": "mmHg", "step": 1, "help": "Single blood pressure value from the source file. The recorded range matches systolic readings."},
    {"key": "PR", "label": "Pulse rate", "group": "Exam", "kind": "number", "unit": "bpm", "step": 1, "help": "Pulse rate."},
    {"key": "Edema", "label": "Edema", "group": "Exam", "kind": "binary", "true_value": 1, "false_value": 0, "help": "Edema on exam."},
    {"key": "Weak Peripheral Pulse", "label": "Weak peripheral pulse", "group": "Exam", "kind": "binary", "true_value": "Y", "false_value": "N", "help": "Weak peripheral pulse."},
    {"key": "Lung rales", "label": "Lung rales", "group": "Exam", "kind": "binary", "true_value": "Y", "false_value": "N", "help": "Rales on lung exam."},
    {"key": "Systolic Murmur", "label": "Systolic murmur", "group": "Exam", "kind": "binary", "true_value": "Y", "false_value": "N", "help": "Systolic murmur."},
    {"key": "Diastolic Murmur", "label": "Diastolic murmur", "group": "Exam", "kind": "binary", "true_value": "Y", "false_value": "N", "help": "Diastolic murmur."},
    {"key": "Q Wave", "label": "Q wave", "group": "ECG", "kind": "binary", "true_value": 1, "false_value": 0, "help": "Pathologic Q wave flag."},
    {"key": "St Elevation", "label": "ST elevation", "group": "ECG", "kind": "binary", "true_value": 1, "false_value": 0, "help": "ST elevation flag."},
    {"key": "St Depression", "label": "ST depression", "group": "ECG", "kind": "binary", "true_value": 1, "false_value": 0, "help": "ST depression flag."},
    {"key": "Tinversion", "label": "T-wave inversion", "group": "ECG", "kind": "binary", "true_value": 1, "false_value": 0, "help": "Source column Tinversion."},
    {"key": "LVH", "label": "LVH", "group": "ECG", "kind": "binary", "true_value": "Y", "false_value": "N", "help": "Left ventricular hypertrophy on ECG."},
    {"key": "Poor R Progression", "label": "Poor R progression", "group": "ECG", "kind": "binary", "true_value": "Y", "false_value": "N", "help": "Poor R-wave progression."},
    {"key": "BBB", "label": "Bundle branch block", "group": "ECG", "kind": "select", "options": ["N", "LBBB", "RBBB"], "help": "None, left bundle branch block, or right bundle branch block."},
    {"key": "FBS", "label": "Fasting blood sugar", "group": "Laboratory", "kind": "number", "unit": "mg/dL", "step": 1, "help": "Fasting blood sugar."},
    {"key": "CR", "label": "Creatinine", "group": "Laboratory", "kind": "number", "unit": "mg/dL", "step": 0.1, "help": "Serum creatinine."},
    {"key": "TG", "label": "Triglycerides", "group": "Laboratory", "kind": "number", "unit": "mg/dL", "step": 1, "help": "Triglycerides."},
    {"key": "LDL", "label": "LDL", "group": "Laboratory", "kind": "number", "unit": "mg/dL", "step": 1, "help": "LDL cholesterol."},
    {"key": "HDL", "label": "HDL", "group": "Laboratory", "kind": "number", "unit": "mg/dL", "step": 0.1, "help": "HDL cholesterol."},
    {"key": "BUN", "label": "BUN", "group": "Laboratory", "kind": "number", "unit": "mg/dL", "step": 1, "help": "Blood urea nitrogen."},
    {"key": "ESR", "label": "ESR", "group": "Laboratory", "kind": "number", "unit": "mm/h", "step": 1, "help": "Erythrocyte sedimentation rate."},
    {"key": "HB", "label": "Hemoglobin", "group": "Laboratory", "kind": "number", "unit": "g/dL", "step": 0.1, "help": "Hemoglobin."},
    {"key": "K", "label": "Potassium", "group": "Laboratory", "kind": "number", "unit": "mEq/L", "step": 0.1, "help": "Serum potassium."},
    {"key": "Na", "label": "Sodium", "group": "Laboratory", "kind": "number", "unit": "mEq/L", "step": 1, "help": "Serum sodium."},
    {"key": "WBC", "label": "WBC", "group": "Laboratory", "kind": "number", "unit": "per uL", "step": 100, "help": "White blood cell count."},
    {"key": "Lymph", "label": "Lymphocytes", "group": "Laboratory", "kind": "number", "unit": "%", "step": 1, "help": "Lymphocyte percent."},
    {"key": "Neut", "label": "Neutrophils", "group": "Laboratory", "kind": "number", "unit": "%", "step": 1, "help": "Neutrophil percent."},
    {"key": "PLT", "label": "Platelets", "group": "Laboratory", "kind": "number", "unit": "10^3/uL", "step": 1, "help": "Platelet count. Source values are in the thousands per microliter."},
    {"key": "EF-TTE", "label": "Ejection fraction", "group": "Echo", "kind": "number", "unit": "%", "step": 1, "help": "Left ventricular ejection fraction by transthoracic echo."},
    {"key": "Region RWMA", "label": "Region with RWMA", "group": "Echo", "kind": "select", "options": ["0", "1", "2", "3", "4"], "help": "Regional wall motion abnormality code from echo. This is a clinical feature, not a 3D lesion map."},
    {"key": "VHD", "label": "Valve disease", "group": "Echo", "kind": "select", "options": ["N", "mild", "Moderate", "Severe"], "help": "Valvular heart disease grade as coded in the source file."},
]

GROUP_ORDER = [
    "Demographics",
    "History",
    "Symptoms",
    "Exam",
    "ECG",
    "Laboratory",
    "Echo",
]

GROUP_COPY = {
    "Demographics": "Age, sex, and body size.",
    "History": "Diagnoses and risk factors already on the chart.",
    "Symptoms": "Chest pain pattern and functional class.",
    "Exam": "Bedside cardiovascular findings.",
    "ECG": "Electrical findings that can travel with ischemia.",
    "Laboratory": "Sugar, lipids, kidney labs, and blood counts.",
    "Echo": "Pump function, wall motion, and valves.",
}

INPUT_BY_KEY = {item["key"]: item for item in INPUTS}


def encoded_feature_table() -> list[dict[str, str]]:
    """Model matrix columns, in order, with display metadata."""
    rows: list[dict[str, str]] = []
    for item in INPUTS:
        key = item["key"]
        if key == "Sex":
            rows.append({
                "key": "Sex_Male",
                "label": "Sex is male",
                "group": "Demographics",
                "source": "Sex",
            })
        elif key == "BBB":
            rows.append({
                "key": "BBB_LBBB",
                "label": "Left bundle branch block",
                "group": "ECG",
                "source": "BBB",
            })
            rows.append({
                "key": "BBB_RBBB",
                "label": "Right bundle branch block",
                "group": "ECG",
                "source": "BBB",
            })
        elif key == "VHD":
            rows.append({
                "key": "VHD_score",
                "label": "Valve disease grade",
                "group": "Echo",
                "source": "VHD",
            })
        else:
            rows.append({
                "key": key,
                "label": item["label"],
                "group": item["group"],
                "source": key,
            })
    return rows


ENCODED_FEATURES = [row["key"] for row in encoded_feature_table()]
ENCODED_META = {row["key"]: row for row in encoded_feature_table()}


def assert_no_leakage(columns: list[str]) -> None:
    banned = {name.lower() for name in LEAKAGE_COLUMNS}
    found = [col for col in columns if col.lower() in banned or col.lower().replace("_", "") in {"lad", "lcx", "rca", "cath"}]
    if found:
        raise RuntimeError(f"Target leakage columns reached the model matrix: {found}")


def normalize_sex(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"male", "m"}:
        return "Male"
    if text in {"female", "fmale", "f"}:
        return "Female"
    raise ValueError("Sex must be Female or Male.")


def normalize_record(raw: dict[str, Any], *, compute_bmi: bool = False) -> dict[str, Any]:
    """Coerce one patient record into the training input schema."""
    leaked = [col for col in LEAKAGE_COLUMNS if col in raw]
    if leaked:
        raise ValueError(
            "LAD, LCX, RCA, and Cath are labels, not inputs. Remove: " + ", ".join(leaked)
        )
    unknown = sorted(set(raw) - set(INPUT_BY_KEY) - set(DROPPED_CONSTANT_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown input fields: {', '.join(unknown)}")

    missing = [item["key"] for item in INPUTS if item["key"] not in raw and not (compute_bmi and item["key"] == "BMI")]
    if missing:
        raise ValueError(f"Missing input fields: {', '.join(missing)}")

    clean: dict[str, Any] = {}
    for item in INPUTS:
        key = item["key"]
        if key not in raw:
            continue
        value = raw[key]
        kind = item["kind"]
        if key == "Sex":
            clean[key] = normalize_sex(value)
            continue
        if kind == "number":
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be a number.") from exc
            if pd.isna(number):
                raise ValueError(f"{key} must be a number.")
            clean[key] = number
            continue
        if kind == "binary":
            true_value = item["true_value"]
            false_value = item["false_value"]
            if isinstance(true_value, str):
                text = str(value).strip().upper()
                if text in {"Y", "YES", "1", "TRUE"}:
                    clean[key] = true_value
                elif text in {"N", "NO", "0", "FALSE"}:
                    clean[key] = false_value
                else:
                    raise ValueError(f"{key} must be Y or N.")
            else:
                try:
                    number = int(float(value))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{key} must be 0 or 1.") from exc
                if number not in (0, 1):
                    raise ValueError(f"{key} must be 0 or 1.")
                clean[key] = number
            continue
        text = str(value).strip()
        options = [str(option) for option in item["options"]]
        if text not in options:
            raise ValueError(f"{key} must be one of: {', '.join(options)}.")
        if key == "Function Class" or key == "Region RWMA":
            clean[key] = int(text)
        else:
            clean[key] = text

    if compute_bmi or "BMI" not in clean:
        height_m = float(clean["Length"]) / 100.0
        if height_m <= 0:
            raise ValueError("Height must be positive to compute BMI.")
        clean["BMI"] = float(clean["Weight"]) / (height_m * height_m)
    return clean


def encode_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        sex_male = 1.0 if record["Sex"] == "Male" else 0.0
        bbb = record["BBB"]
        row: dict[str, float] = {}
        for item in INPUTS:
            key = item["key"]
            if key in {"Sex", "BBB", "VHD"}:
                continue
            value = record[key]
            if item["kind"] == "binary" and isinstance(item["true_value"], str):
                row[key] = 1.0 if value == "Y" else 0.0
            else:
                row[key] = float(value)
        row["Sex_Male"] = sex_male
        row["VHD_score"] = float(VHD_SCORE[record["VHD"]])
        row["BBB_LBBB"] = 1.0 if bbb == "LBBB" else 0.0
        row["BBB_RBBB"] = 1.0 if bbb == "RBBB" else 0.0
        rows.append(row)
    frame = pd.DataFrame(rows, columns=ENCODED_FEATURES)
    assert_no_leakage(list(frame.columns))
    return frame


def load_raw_table(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Sex" in frame.columns:
        frame["Sex"] = frame["Sex"].replace({"Fmale": "Female", "female": "Female", "male": "Male"})
    return frame


def records_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for _, series in frame.iterrows():
        raw = {}
        for item in INPUTS:
            key = item["key"]
            value = series[key]
            if item["kind"] == "number":
                number = float(value)
                raw[key] = int(number) if number.is_integer() and item.get("step") == 1 else number
            elif item["kind"] == "binary" and not isinstance(item["true_value"], str):
                raw[key] = int(value)
            elif key in {"Function Class", "Region RWMA"}:
                raw[key] = str(int(value))
            else:
                raw[key] = str(value) if key in {"Sex", "BBB", "VHD"} or isinstance(item.get("true_value"), str) else value
                if key == "Sex":
                    raw[key] = normalize_sex(value)
                elif isinstance(item.get("true_value"), str):
                    raw[key] = "Y" if str(value).strip().upper() in {"Y", "YES", "1"} else "N"
        records.append(normalize_record(raw))
    return records


def display_input(key: str, value: Any) -> str:
    item = INPUT_BY_KEY[key]
    if key == "Sex":
        return str(value)
    if item["kind"] == "binary":
        return "Yes" if value == item["true_value"] else "No"
    if key == "BBB":
        return {"N": "None", "LBBB": "LBBB", "RBBB": "RBBB"}[str(value)]
    if key == "VHD":
        return {"N": "None", "mild": "Mild", "Moderate": "Moderate", "Severe": "Severe"}[str(value)]
    if key in {"Function Class", "Region RWMA"}:
        return str(int(value))
    number = float(value)
    if item.get("step") == 1 and number.is_integer():
        text = str(int(number))
    else:
        text = f"{number:.1f}" if abs(number) >= 10 or number.is_integer() else f"{number:.2f}"
        if number.is_integer() and item.get("step", 1) >= 1:
            text = str(int(number))
    unit = item.get("unit")
    return f"{text} {unit}" if unit else text


def display_encoded(feature: str, record: dict[str, Any]) -> str:
    meta = ENCODED_META[feature]
    source = meta["source"]
    if feature == "Sex_Male":
        return record["Sex"]
    if feature == "BBB_LBBB":
        return "Yes" if record["BBB"] == "LBBB" else "No"
    if feature == "BBB_RBBB":
        return "Yes" if record["BBB"] == "RBBB" else "No"
    if feature == "VHD_score":
        return display_input("VHD", record["VHD"])
    return display_input(source, record[source])
