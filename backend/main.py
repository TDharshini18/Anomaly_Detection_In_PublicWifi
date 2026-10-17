"""Wi-Fi anomaly detection API.

This is a prototype for analysing uploaded feature tables. It does not sniff live Wi-Fi traffic.
"""
from __future__ import annotations

import io
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="Wi-Fi Anomaly Detection API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURES = ["signal_strength", "packet_size", "frequency", "connection_duration", "data_rate", "retry_count", "encryption_type", "channel"]
NUMERIC_FEATURES = [x for x in FEATURES if x != "encryption_type"]
ALIASES = {
    "signal_strength": ["signal_strength", "signal", "strength", "rssi", "dbm", "power"],
    "packet_size": ["packet_size", "packet", "bytes", "length", "payload"],
    "frequency": ["frequency", "freq", "band", "ghz", "channel_freq"],
    "connection_duration": ["connection_duration", "duration", "session_duration", "elapsed"],
    "data_rate": ["data_rate", "rate", "speed", "bandwidth", "throughput", "mbps"],
    "retry_count": ["retry_count", "retry", "retries", "attempts", "failures"],
    "encryption_type": ["encryption_type", "encryption", "security", "auth", "protocol", "cipher"],
    "channel": ["channel", "ch", "wifi_channel"],
}
LABEL_ALIASES = {"is_suspicious", "is_anomaly", "anomaly", "label", "target", "class"}
CATEGORIES = ["WPA2", "WPA3", "WEP", "OPEN", "UNKNOWN"]

class PredictionRequest(BaseModel):
    file_id: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    column_mapping: Dict[str, str] = Field(default_factory=dict)

class PredictionResponse(BaseModel):
    prediction_id: str
    anomaly_scores: List[float]
    is_anomaly: List[bool]
    reconstruction_errors: List[float]
    overall_anomaly: bool
    avg_anomaly_score: float
    anomaly_percentage: float
    total_records: int
    timestamp: str
    status: str

uploaded_datasets: Dict[str, dict] = {}
prediction_results: Dict[str, dict] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_columns(df: pd.DataFrame, mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Map user columns to canonical features and clean types, preserving row count."""
    source = df.copy()
    source.columns = [str(c).strip() for c in source.columns]
    if mapping:
        mapped = {}
        for target, src in mapping.items():
            if src in source.columns:
                mapped[str(target).strip().lower()] = source[src]
        if mapped:
            source = pd.DataFrame(mapped, index=source.index)
    lower_lookup = {str(c).strip().lower(): c for c in source.columns}
    output = pd.DataFrame(index=source.index)
    for feature, aliases in ALIASES.items():
        found = None
        for alias in aliases:
            if alias in lower_lookup:
                found = lower_lookup[alias]
                break
        if found is None:
            continue
        output[feature] = source[found]
    if not output.columns.tolist():
        raise ValueError("No recognised Wi-Fi feature columns were found. Map at least one supported feature.")
    for col in NUMERIC_FEATURES:
        if col in output:
            output[col] = pd.to_numeric(output[col], errors="coerce")
    if "encryption_type" in output:
        output["encryption_type"] = output["encryption_type"].astype("string").fillna("UNKNOWN").str.strip().str.upper()
        output["encryption_type"] = output["encryption_type"].replace({"OPEN NETWORK": "OPEN", "NONE": "OPEN", "": "UNKNOWN"})
        output.loc[~output["encryption_type"].isin(CATEGORIES), "encryption_type"] = "UNKNOWN"
    # Drop rows that have no usable feature values; fill partial numeric gaps with column medians.
    output = output.replace([np.inf, -np.inf], np.nan)
    output = output.dropna(how="all")
    for col in NUMERIC_FEATURES:
        if col in output:
            med = output[col].median()
            output[col] = output[col].fillna(0 if pd.isna(med) else med)
    if output.empty:
        raise ValueError("The uploaded data has no usable records after cleaning.")
    return output


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stable one-hot encoding with the same columns at training and inference."""
    pieces = []
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            pieces.append(df[[col]].astype(float).rename(columns={col: f"num__{col}"}))
        else:
            pieces.append(pd.DataFrame({f"num__{col}": np.zeros(len(df))}, index=df.index))
    if "encryption_type" in df.columns:
        cat = df["encryption_type"].astype(str).str.upper()
    else:
        cat = pd.Series(["UNKNOWN"] * len(df), index=df.index)
    cat = pd.Categorical(cat, categories=CATEGORIES)
    dummies = pd.get_dummies(cat, prefix="encryption", dtype=float)
    dummies.index = df.index
    # Explicitly ensure all expected categories exist, regardless of the input.
    for category in CATEGORIES:
        name = f"encryption_{category}"
        if name not in dummies:
            dummies[name] = 0.0
    pieces.append(dummies[[f"encryption_{c}" for c in CATEGORIES]])
    return pd.concat(pieces, axis=1).astype(float)

class WiFiAnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model: Optional[IsolationForest] = None
        self.threshold: Optional[float] = None
        self.is_trained = False
        self.feature_columns: List[str] = []
        self.training_source = "not trained"

    def create_synthetic_data(self, n_samples: int = 3000) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n = n_samples
        return pd.DataFrame({
            "signal_strength": np.clip(rng.normal(-52, 9, n), -100, -5),
            "packet_size": np.clip(rng.normal(1000, 250, n), 40, 9000),
            "frequency": rng.choice([2.4, 5.0], n, p=[0.6, 0.4]),
            "connection_duration": np.clip(rng.exponential(300, n), 1, 7200),
            "data_rate": np.clip(rng.normal(54, 15, n), 1, 1000),
            "retry_count": rng.poisson(2, n),
            "encryption_type": rng.choice(["WPA2", "WPA3", "OPEN"], n, p=[0.7, 0.2, 0.1]),
            "channel": rng.choice(list(range(1, 14)) + [36, 40, 44, 48, 149, 153, 157, 161], n),
        })

    def train_model(self, df: Optional[pd.DataFrame] = None):
        source = "synthetic demo data" if df is None else "uploaded data"
        if df is None:
            normal = self.create_synthetic_data()
        else:
            raw = df.copy()
            labels_col = next((c for c in raw.columns if str(c).strip().lower() in LABEL_ALIASES), None)
            if labels_col:
                labels = pd.to_numeric(raw[labels_col], errors="coerce")
                # 0 means normal, 1 means suspicious. Only train on normal rows if available.
                normal_raw = raw.loc[labels == 0].drop(columns=[labels_col], errors="ignore")
                if len(normal_raw) >= 30:
                    raw = normal_raw
            normal = normalize_columns(raw)
        encoded = encode_features(normal)
        self.feature_columns = encoded.columns.tolist()
        scaled = self.scaler.fit_transform(encoded)
        self.model = IsolationForest(n_estimators=250, contamination="auto", random_state=42, n_jobs=-1)
        self.model.fit(scaled)
        # Lower decision_function values are more anomalous. Use a conservative training-tail threshold.
        training_scores = self.model.decision_function(scaled)
        self.threshold = float(np.quantile(training_scores, 0.02))
        self.is_trained = True
        self.training_source = source
        return {"training_records": int(len(normal)), "threshold": self.threshold, "source": source}

    def predict_anomaly(self, df: pd.DataFrame):
        if not self.is_trained or self.model is None or self.threshold is None:
            self.train_model()
        cleaned = normalize_columns(df)
        encoded = encode_features(cleaned).reindex(columns=self.feature_columns, fill_value=0.0)
        scaled = self.scaler.transform(encoded)
        decision = self.model.decision_function(scaled)
        errors = np.maximum(self.threshold - decision, 0.0) + np.maximum(-decision, 0.0)
        # A positive score is above the calibrated anomaly boundary; avoid division by zero.
        scores = np.maximum(self.threshold - decision, 0.0) / max(abs(self.threshold), 1e-6)
        anomalies = decision < self.threshold
        return scores, anomalies, errors

detector = WiFiAnomalyDetector()

@app.get("/")
def root():
    return {"message": "Wi-Fi Anomaly Detection API", "status": "running", "model_trained": detector.is_trained, "training_source": detector.training_source, "note": "Uploaded-table analysis prototype; not a live Wi-Fi packet monitor."}

@app.get("/health")
def health():
    return {"status": "ok", "model_trained": detector.is_trained}

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    filename = (file.filename or "upload.csv").lower()
    try:
        content = await file.read()
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File exceeds the 25 MB upload limit.")
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith(".json"):
            data = json.loads(content.decode("utf-8-sig"))
            df = pd.DataFrame(data if isinstance(data, list) else [data])
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Upload CSV, JSON, or Excel.")
        df = df.dropna(how="all").loc[:, df.notna().any()]
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        file_id = f"upload_{uuid.uuid4().hex[:12]}"
        uploaded_datasets[file_id] = {"dataframe": df, "filename": file.filename, "upload_time": now_iso(), "processed": False}
        return {"file_id": file_id, "filename": file.filename, "columns": list(df.columns), "shape": list(df.shape), "sample_data": df.head(10).replace({np.nan: None}).to_dict("records"), "total_records": len(df), "upload_time": uploaded_datasets[file_id]["upload_time"], "status": "uploaded_successfully"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {exc}") from exc

@app.post("/auto-detect-columns")
def auto_detect_columns(data: Dict[str, List[str]]):
    columns = data.get("columns", [])
    mapping = {}
    normalized = {str(c).strip().lower(): c for c in columns}
    for feature, aliases in ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                mapping[feature] = normalized[alias]
                break
        if feature not in mapping:
            for col in columns:
                low = str(col).strip().lower()
                if any(alias in low for alias in aliases):
                    mapping[feature] = col
                    break
    return {"column_mapping": mapping, "mapped_columns": len(mapping), "total_columns": len(columns)}

@app.post("/train-model")
def train_model(file_id: Optional[str] = None):
    try:
        if file_id:
            if file_id not in uploaded_datasets:
                raise HTTPException(status_code=404, detail="Uploaded file ID was not found. Upload the file again.")
            training = detector.train_model(uploaded_datasets[file_id]["dataframe"])
        else:
            training = detector.train_model()
        return {"message": f"Model trained using {training['source']}", "threshold": training["threshold"], "model_trained": True, "training_completed": True, "training_records": training["training_records"]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Training failed: {exc}") from exc

@app.post("/predict", response_model=PredictionResponse)
def predict_anomaly(request: PredictionRequest):
    try:
        if request.file_id:
            if request.file_id not in uploaded_datasets:
                raise HTTPException(status_code=404, detail="Uploaded file ID was not found. Upload the file again.")
            original = uploaded_datasets[request.file_id]["dataframe"].copy()
        elif request.data:
            original = pd.DataFrame(request.data)
        else:
            raise HTTPException(status_code=400, detail="Provide a file_id or non-empty data array.")
        if not detector.is_trained:
            detector.train_model()
        mapped = normalize_columns(original, request.column_mapping)
        scores, anomalies, errors = detector.predict_anomaly(mapped)
        pct = float(np.mean(anomalies) * 100)
        pred_id = f"pred_{uuid.uuid4().hex[:12]}"
        timestamp = now_iso()
        prediction_results[pred_id] = {"original_data": original.loc[mapped.index].copy(), "anomaly_scores": scores, "is_anomaly": anomalies, "reconstruction_errors": errors, "timestamp": timestamp}
        if request.file_id:
            uploaded_datasets[request.file_id]["processed"] = True
        return PredictionResponse(prediction_id=pred_id, anomaly_scores=scores.astype(float).tolist(), is_anomaly=anomalies.astype(bool).tolist(), reconstruction_errors=errors.astype(float).tolist(), overall_anomaly=pct >= 5.0, avg_anomaly_score=float(np.mean(scores)), anomaly_percentage=pct, total_records=len(mapped), timestamp=timestamp, status="prediction_completed")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

@app.get("/download-results/{prediction_id}")
def download_results(prediction_id: str):
    if prediction_id not in prediction_results:
        raise HTTPException(status_code=404, detail="Prediction not found. Run an analysis first.")
    result = prediction_results[prediction_id]
    out_df = result["original_data"].copy()
    out_df["anomaly_score"] = result["anomaly_scores"]
    out_df["is_anomaly"] = result["is_anomaly"]
    out_df["decision_margin"] = result["reconstruction_errors"]
    out_df["anomaly_status"] = np.where(result["is_anomaly"], "ANOMALOUS", "NORMAL")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="Anomaly_Results", index=False)
        count = int(np.sum(result["is_anomaly"]))
        pd.DataFrame({"Metric": ["Total Records", "Anomalous Records", "Normal Records", "Anomaly Percentage", "Average Anomaly Score"], "Value": [len(out_df), count, len(out_df)-count, f"{count/len(out_df)*100:.2f}%", f"{np.mean(result['anomaly_scores']):.4f}"]}).to_excel(writer, sheet_name="Summary", index=False)
    output.seek(0)
    return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename=anomaly_results_{prediction_id}.xlsx"})

@app.post("/reset-session")
def reset_session():
    uploaded_datasets.clear()
    prediction_results.clear()
    return {"message": "Session reset successfully", "status": "ready_for_new_data"}

@app.get("/session-status")
def session_status():
    return {"uploaded_files": len(uploaded_datasets), "predictions": len(prediction_results), "model_trained": detector.is_trained, "files": [{"file_id": fid, "filename": item["filename"], "records": len(item["dataframe"]), "processed": item["processed"], "upload_time": item["upload_time"]} for fid, item in uploaded_datasets.items()]}

@app.get("/demo-data/{data_type}")
def demo_data(data_type: str):
    if data_type not in {"normal", "suspicious"}:
        raise HTTPException(status_code=400, detail="data_type must be 'normal' or 'suspicious'.")
    rng = np.random.default_rng(2026 if data_type == "normal" else 2027)
    suspicious = data_type == "suspicious"
    rows = []
    for _ in range(100):
        rows.append({"signal_strength": float(rng.normal(-25 if suspicious else -52, 5 if suspicious else 8)), "packet_size": int(max(40, rng.normal(2048 if suspicious else 1024, 300 if suspicious else 150))), "frequency": float(rng.choice([2.4, 5.0])), "connection_duration": float(rng.exponential(60 if suspicious else 300)), "data_rate": float(rng.normal(25 if suspicious else 54, 15 if suspicious else 10)), "retry_count": int(rng.poisson(12 if suspicious else 2)), "encryption_type": "OPEN" if suspicious else str(rng.choice(["WPA2", "WPA3"])), "channel": int(rng.choice([1, 6, 11]) if suspicious else rng.integers(1, 12))})
    return {"data": rows, "total_records": len(rows)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
