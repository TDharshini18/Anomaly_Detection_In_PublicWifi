import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from fastapi.testclient import TestClient
from main import app, detector

client = TestClient(app)


def sample_rows(n=40, suspicious=False):
    return [{
        "signal_strength": -25 if suspicious else -52,
        "packet_size": 2048 if suspicious else 1024,
        "frequency": 2.4,
        "connection_duration": 30 if suspicious else 300,
        "data_rate": 20 if suspicious else 54,
        "retry_count": 12 if suspicious else 2,
        "encryption_type": "OPEN" if suspicious else "WPA2",
        "channel": 1 if suspicious else 6,
    } for _ in range(n)]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_demo_data_and_prediction():
    detector.is_trained = False
    demo = client.get("/demo-data/normal")
    assert demo.status_code == 200
    payload = {"data": demo.json()["data"], "column_mapping": {}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_records"] == 100
    assert len(body["is_anomaly"]) == 100
    assert body["status"] == "prediction_completed"


def test_upload_actual_dataset_schema_and_predict():
    df = pd.DataFrame(sample_rows(40))
    df["network_name"] = ["CafeWiFi"] * len(df)
    df["is_suspicious"] = [0] * len(df)
    content = df.to_csv(index=False).encode()
    uploaded = client.post("/upload-file", files={"file": ("test_wifi.csv", content, "text/csv")})
    assert uploaded.status_code == 200, uploaded.text
    file_id = uploaded.json()["file_id"]
    predicted = client.post("/predict", json={"file_id": file_id, "column_mapping": {"signal_strength": "signal_strength", "packet_size": "packet_size", "frequency": "frequency", "connection_duration": "connection_duration", "data_rate": "data_rate", "retry_count": "retry_count", "encryption_type": "encryption_type", "channel": "channel"}})
    assert predicted.status_code == 200, predicted.text
    assert predicted.json()["total_records"] == 40


def test_unknown_mapping_returns_helpful_error():
    response = client.post("/predict", json={"data": [{"random_col": 1}], "column_mapping": {}})
    assert response.status_code == 400
    assert "recognised Wi-Fi feature" in response.json()["detail"]
