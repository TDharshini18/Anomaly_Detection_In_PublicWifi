from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import io
from typing import Dict, List, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import keras
from keras import layers
from datetime import datetime

app = FastAPI(title="Wi-Fi Anomaly Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    file_id: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    column_mapping: Dict[str, str]

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

# Storage for datasets and predictions
uploaded_datasets = {}
prediction_results = {}

class WiFiAnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.threshold = None
        self.is_trained = False
        
    def create_synthetic_data(self, n_samples=2000):
        """Create synthetic Wi-Fi traffic data for training"""
        np.random.seed(42)
        
        # Normal traffic patterns (70%)
        normal_size = int(n_samples * 0.7)
        normal_data = {
            'signal_strength': np.random.normal(-50, 10, normal_size),
            'packet_size': np.random.normal(1024, 200, normal_size),
            'frequency': np.random.choice([2.4, 5.0], normal_size, p=[0.6, 0.4]),
            'connection_duration': np.random.exponential(300, normal_size),
            'data_rate': np.random.normal(54, 15, normal_size),
            'retry_count': np.random.poisson(2, normal_size),
            'encryption_type': np.random.choice(['WPA2', 'WPA3', 'Open'], normal_size, p=[0.7, 0.2, 0.1]),
            'channel': np.random.randint(1, 13, normal_size),
            'is_anomaly': [0] * normal_size
        }
        
        # Anomalous traffic patterns (30%)
        anomaly_size = n_samples - normal_size
        anomaly_data = {
            'signal_strength': np.random.normal(-30, 5, anomaly_size),
            'packet_size': np.random.normal(2048, 500, anomaly_size),
            'frequency': np.random.choice([2.4, 5.0], anomaly_size, p=[0.8, 0.2]),
            'connection_duration': np.random.exponential(60, anomaly_size),
            'data_rate': np.random.normal(30, 20, anomaly_size),
            'retry_count': np.random.poisson(8, anomaly_size),
            'encryption_type': np.random.choice(['WPA2', 'WPA3', 'Open'], anomaly_size, p=[0.3, 0.1, 0.6]),
            'channel': np.random.randint(1, 13, anomaly_size),
            'is_anomaly': [1] * anomaly_size
        }
        
        # Combine data
        data = {}
        for key in normal_data.keys():
            data[key] = np.concatenate([normal_data[key], anomaly_data[key]])
        
        # Shuffle the data
        df = pd.DataFrame(data)
        return df.sample(frac=1).reset_index(drop=True)
    
    def preprocess_data(self, df):
        """Preprocess the data for training/prediction"""
        df_processed = df.copy()
        
        # Handle categorical variables
        categorical_columns = ['encryption_type']
        for col in categorical_columns:
            if col in df_processed.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
                else:
                    # Handle unseen categories
                    unique_values = set(self.label_encoders[col].classes_)
                    df_processed[col] = df_processed[col].astype(str).apply(
                        lambda x: x if x in unique_values else 'unknown'
                    )
                    if 'unknown' not in unique_values:
                        self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'unknown')
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        # Remove target column if present
        feature_columns = [col for col in df_processed.columns if col != 'is_anomaly']
        X = df_processed[feature_columns]
        
        return X
    
    def build_autoencoder(self, input_dim):
        """Build autoencoder model"""
        encoder_input = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(128, activation='relu')(encoder_input)
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.Dense(32, activation='relu')(encoded)
        encoded = layers.Dense(16, activation='relu')(encoded)
        
        decoded = layers.Dense(32, activation='relu')(encoded)
        decoded = layers.Dense(64, activation='relu')(decoded)
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        autoencoder = keras.Model(encoder_input, decoded)
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return autoencoder
    
    def train_model(self, df=None):
        """Train the autoencoder model"""
        if df is None:
            df = self.create_synthetic_data(3000)
            print("Training on synthetic data...")
        else:
            print(f"Training on provided data with {len(df)} records...")
        
        # Use only normal data for training
        if 'is_anomaly' in df.columns:
            normal_data = df[df['is_anomaly'] == 0]
        else:
            normal_data = df
        
        X_normal = self.preprocess_data(normal_data)
        X_normal_scaled = self.scaler.fit_transform(X_normal)
        
        self.model = self.build_autoencoder(X_normal_scaled.shape[1])
        
        history = self.model.fit(
            X_normal_scaled, X_normal_scaled,
            epochs=100,
            batch_size=64,
            validation_split=0.2,
            verbose=1
        )
        
        # Calculate threshold
        reconstructions = self.model.predict(X_normal_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_normal_scaled - reconstructions), axis=1)
        self.threshold = np.percentile(reconstruction_errors, 95)
        self.is_trained = True
        
        print(f"Model trained successfully. Threshold: {self.threshold:.4f}")
        return history
    
    def predict_anomaly(self, df):
        """Predict if the input data contains anomalies"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet!")
        
        X = self.preprocess_data(df)
        X_scaled = self.scaler.transform(X)
        
        reconstructions = self.model.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        anomaly_scores = reconstruction_errors / self.threshold
        is_anomaly = reconstruction_errors > self.threshold
        
        return anomaly_scores, is_anomaly, reconstruction_errors

# Global detector instance
detector = WiFiAnomalyDetector()

@app.get("/")
async def root():
    return {
        "message": "Wi-Fi Anomaly Detection API",
        "status": "running",
        "model_trained": detector.is_trained
    }

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process file - store full dataset"""
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Clean data
        df = df.dropna(how='all').loc[:, df.notna().any()]
        
        # Generate unique file ID
        file_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store full dataset
        uploaded_datasets[file_id] = {
            'dataframe': df,
            'filename': file.filename,
            'upload_time': datetime.now().isoformat(),
            'processed': False
        }
        
        print(f"File uploaded: {file.filename}, Records: {len(df)}")
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "columns": list(df.columns),
            "shape": df.shape,
            "sample_data": df.head(10).to_dict('records'),
            "total_records": len(df),
            "upload_time": uploaded_datasets[file_id]['upload_time'],
            "status": "uploaded_successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.post("/auto-detect-columns")
async def auto_detect_columns(data: Dict[str, List[str]]):
    """Auto-detect column mappings"""
    columns = data.get("columns", [])
    
    column_patterns = {
        'signal_strength': ['signal', 'strength', 'rssi', 'dbm', 'power'],
        'packet_size': ['packet', 'size', 'bytes', 'length', 'payload'],
        'frequency': ['frequency', 'freq', 'band', 'ghz', 'channel_freq'],
        'connection_duration': ['duration', 'time', 'connection', 'session', 'elapsed'],
        'data_rate': ['rate', 'speed', 'bandwidth', 'mbps', 'throughput'],
        'retry_count': ['retry', 'retries', 'attempts', 'count', 'failures'],
        'encryption_type': ['encryption', 'security', 'auth', 'protocol', 'cipher'],
        'channel': ['channel', 'ch', 'wifi_channel', 'frequency_channel']
    }
    
    column_mapping = {}
    columns_lower = [col.lower() for col in columns]
    
    for target_col, keywords in column_patterns.items():
        best_match = None
        best_score = 0
        
        for i, col in enumerate(columns_lower):
            score = 0
            for keyword in keywords:
                if keyword in col:
                    score += 1
                    if keyword == col:
                        score += 2
            
            if score > best_score:
                best_score = score
                best_match = columns[i]
        
        if best_match and best_score > 0:
            column_mapping[target_col] = best_match
    
    return {
        "column_mapping": column_mapping,
        "mapped_columns": len(column_mapping),
        "total_columns": len(columns)
    }

@app.post("/train-model")
async def train_model(file_id: Optional[str] = None):
    """Train the anomaly detection model"""
    try:
        if file_id and file_id in uploaded_datasets:
            df = uploaded_datasets[file_id]['dataframe']
            history = detector.train_model(df)
            message = f"Model trained on uploaded data ({len(df)} records)"
        else:
            history = detector.train_model()
            message = "Model trained on synthetic data"
        
        return {
            "message": message,
            "threshold": float(detector.threshold),
            "model_trained": detector.is_trained,
            "training_completed": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_anomaly(request: PredictionRequest):
    """Predict anomalies in the provided data"""
    try:
        # Auto-train if not trained
        if not detector.is_trained:
            print("Model not trained, training now...")
            detector.train_model()
        
        # Get data
        if request.file_id and request.file_id in uploaded_datasets:
            df = uploaded_datasets[request.file_id]['dataframe'].copy()
            print(f"Processing full dataset: {len(df)} records")
        elif request.data:
            df = pd.DataFrame(request.data)
            print(f"Processing sample data: {len(df)} records")
        else:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Apply column mapping
        if request.column_mapping:
            mapped_df = pd.DataFrame()
            for target_col, source_col in request.column_mapping.items():
                if source_col and source_col in df.columns:
                    mapped_df[target_col] = df[source_col]
            
            if mapped_df.empty:
                raise HTTPException(status_code=400, detail="No valid column mappings found")
            
            # Store original data for results
            original_df = df.copy()
            df = mapped_df
        
        # Make predictions
        anomaly_scores, is_anomaly, reconstruction_errors = detector.predict_anomaly(df)
        
        # Calculate statistics
        overall_anomaly = np.mean(is_anomaly) > 0.3
        avg_anomaly_score = float(np.mean(anomaly_scores))
        anomaly_percentage = float(np.mean(is_anomaly) * 100)
        
        # Generate prediction ID
        prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store results with original data
        prediction_results[prediction_id] = {
            'original_data': original_df if 'original_df' in locals() else df,
            'mapped_data': df,
            'anomaly_scores': anomaly_scores,
            'is_anomaly': is_anomaly,
            'reconstruction_errors': reconstruction_errors,
            'column_mapping': request.column_mapping,
            'timestamp': datetime.now().isoformat(),
            'file_id': request.file_id
        }
        
        # Mark file as processed
        if request.file_id and request.file_id in uploaded_datasets:
            uploaded_datasets[request.file_id]['processed'] = True
        
        print(f"Prediction complete: {len(df)} records, {np.sum(is_anomaly)} anomalies found")
        
        return PredictionResponse(
            prediction_id=prediction_id,
            anomaly_scores=anomaly_scores.tolist(),
            is_anomaly=is_anomaly.tolist(),
            reconstruction_errors=reconstruction_errors.tolist(),
            overall_anomaly=overall_anomaly,
            avg_anomaly_score=avg_anomaly_score,
            anomaly_percentage=anomaly_percentage,
            total_records=len(df),
            timestamp=datetime.now().isoformat(),
            status="prediction_completed"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")

@app.get("/download-results/{prediction_id}")
async def download_results(prediction_id: str):
    """Download prediction results as Excel file"""
    try:
        if prediction_id not in prediction_results:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        result = prediction_results[prediction_id]
        
        # Prepare results dataframe
        df_results = result['original_data'].copy()
        df_results['anomaly_score'] = result['anomaly_scores']
        df_results['is_anomaly'] = result['is_anomaly']
        df_results['reconstruction_error'] = result['reconstruction_errors']
        df_results['anomaly_status'] = ['ANOMALOUS' if x else 'NORMAL' for x in result['is_anomaly']]
        
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='Anomaly_Results', index=False)
            
            # Add summary sheet
            summary_data = {
                'Metric': ['Total Records', 'Anomalous Records', 'Normal Records', 
                          'Anomaly Percentage', 'Average Anomaly Score'],
                'Value': [
                    len(result['is_anomaly']),
                    np.sum(result['is_anomaly']),
                    len(result['is_anomaly']) - np.sum(result['is_anomaly']),
                    f"{np.mean(result['is_anomaly']) * 100:.2f}%",
                    f"{np.mean(result['anomaly_scores']):.4f}"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={"Content-Disposition": f"attachment; filename=anomaly_results_{prediction_id}.xlsx"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading results: {str(e)}")

@app.post("/reset-session")
async def reset_session():
    """Reset current session - clear all data"""
    global uploaded_datasets, prediction_results
    
    uploaded_datasets.clear()
    prediction_results.clear()
    
    return {
        "message": "Session reset successfully",
        "status": "ready_for_new_data"
    }

@app.get("/session-status")
async def get_session_status():
    """Get current session status"""
    return {
        "uploaded_files": len(uploaded_datasets),
        "predictions": len(prediction_results),
        "model_trained": detector.is_trained,
        "files": [
            {
                "file_id": fid,
                "filename": data['filename'],
                "records": len(data['dataframe']),
                "processed": data['processed'],
                "upload_time": data['upload_time']
            }
            for fid, data in uploaded_datasets.items()
        ]
    }

@app.get("/demo-data/{data_type}")
async def get_demo_data(data_type: str):
    """Get demo data for testing"""
    try:
        if data_type == "normal":
            demo_data = []
            for i in range(100):
                demo_data.append({
                    'signal_strength': float(np.random.normal(-52, 8)),
                    'packet_size': int(np.random.normal(1024, 150)),
                    'frequency': float(np.random.choice([2.4, 5.0])),
                    'connection_duration': float(np.random.exponential(300)),
                    'data_rate': float(np.random.normal(54, 10)),
                    'retry_count': int(np.random.poisson(2)),
                    'encryption_type': np.random.choice(['WPA2', 'WPA3']),
                    'channel': int(np.random.randint(1, 13))
                })
        elif data_type == "suspicious":
            demo_data = []
            for i in range(100):
                demo_data.append({
                    'signal_strength': float(np.random.normal(-25, 5)),
                    'packet_size': int(np.random.normal(2048, 300)),
                    'frequency': 2.4,
                    'connection_duration': float(np.random.exponential(60)),
                    'data_rate': float(np.random.normal(25, 15)),
                    'retry_count': int(np.random.poisson(12)),
                    'encryption_type': 'Open',
                    'channel': 1
                })
        else:
            raise HTTPException(status_code=400, detail="Invalid data type")
        
        return {"data": demo_data, "total_records": len(demo_data)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating demo data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)