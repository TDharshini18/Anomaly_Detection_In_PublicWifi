# Public Wi-Fi Anomaly Detection

A local web application for analysing **tabular Wi-Fi/network feature data** and flagging records that differ from a learned baseline. It includes a FastAPI backend, React frontend, sample dataset, demo data, and downloadable Excel results.

> **Scope and limitations:** this is an educational anomaly-analysis prototype. It does not scan nearby networks, capture packets, identify a real attacker, or provide a validated production IDS. Results are anomaly flags, not proof of malicious activity. The default baseline is synthetic; performance on real networks must be evaluated with representative labelled data.

## Architecture

- **Frontend:** React 18, Ant Design, Recharts/Plotly
- **API:** FastAPI + Uvicorn
- **Detection:** scikit-learn Isolation Forest, trained on baseline records; stable numeric scaling and categorical encoding
- **Input:** CSV, JSON, XLSX/XLS
- **Output:** per-record anomaly flags/scores and downloadable Excel summary

## Requirements

- Python 3.10–3.12 (recommended: Python 3.11)
- Node.js 18 or 20 LTS and npm
- Windows, macOS, or Linux

TensorFlow is no longer required. This avoids the prior TensorFlow/Python-version incompatibility and makes local installation lighter.

## Run on Windows

Open PowerShell in this repository directory.

### 1. Start the backend

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

If Python 3.11 is not installed, install it from python.org or use another supported Python version and replace `py -3.11` accordingly.

Verify `http://127.0.0.1:8000/health` returns `{"status":"ok",...}`. API documentation is at `http://127.0.0.1:8000/docs`.

### 2. Start the frontend (new terminal)

```powershell
cd frontend
npm ci
npm start
```

Open `http://localhost:3000`. Keep the backend terminal running.

## Run tests

With the backend virtual environment activated:

```powershell
cd backend
pytest -q
```

## Dataset

`wifinetwork_dataset.csv` has 20,040 rows and columns including `signal_strength`, `packet_size`, `frequency`, `connection_duration`, `data_rate`, `retry_count`, `encryption_type`, `channel`, `network_name`, and `is_suspicious`.

The API recognises `is_suspicious` and `is_anomaly` as label columns when choosing normal records for training. Labels are not used as model input features. For a defensible evaluation, reserve labelled records that were not used for training and report precision, recall, F1, and confusion matrix. The current app does not claim benchmark accuracy.

## API endpoints

- `GET /health` — health check
- `POST /upload-file` — upload CSV, JSON, or Excel
- `POST /auto-detect-columns` — suggest feature mappings
- `POST /train-model` — train on synthetic baseline or an uploaded file ID
- `POST /predict` — analyse a dataset or uploaded file
- `GET /download-results/{prediction_id}` — download Excel results
- `GET /demo-data/normal` and `/demo-data/suspicious` — demo examples
- `GET /session-status` — current in-memory session status
- `POST /reset-session` — clear in-memory uploaded data and results

Uploaded data and prediction results are held in memory and disappear when the API process restarts. Do not upload sensitive or personal data to an untrusted deployment.

## Model notes

Isolation Forest is an unsupervised baseline. The model is calibrated against a low-tail score threshold from its training records. A record flagged as anomalous is simply unusual relative to that baseline; it is not automatically a malicious Wi-Fi event. Real-world use requires domain-specific features, robust validation, drift monitoring, and security review.
