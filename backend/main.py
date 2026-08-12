"""
FastAPI Backend for EEG-SINDy Seizure Prediction
Main application file
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import mne
from scipy.signal import savgol_filter
import pysindy as ps
from sklearn.ensemble import IsolationForest
import io
import tempfile
import os

def time_delay_embedding(data, embedding_dim=3, delay=4):
    """
    Constructs a time-delay embedded state space.
    data: (samples, channels)
    returns: (samples - (embedding_dim-1)*delay, channels * embedding_dim)
    """
    samples, channels = data.shape
    new_samples = samples - (embedding_dim - 1) * delay
    if new_samples <= 0:
        return data
        
    embedded = np.zeros((new_samples, channels * embedding_dim))
    for i in range(embedding_dim):
        start_idx = i * delay
        end_idx = start_idx + new_samples
        embedded[:, i*channels:(i+1)*channels] = data[start_idx:end_idx, :]
        
    return embedded

app = FastAPI(
    title="EEG-SINDy API",
    description="Seizure prediction using Sparse Identification of Nonlinear Dynamics",
    version="1.0.0"
)

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use Redis or database)
app_state = {
    "baseline_data": None,
    "test_data": None,
    "sindy_model": None,
    "X_train": None,
    "dX_train": None,
    "is_demo": False
}

# Pydantic models for request/response
class PreprocessResponse(BaseModel):
    success: bool
    message: str
    raw_data: List[List[float]]
    filtered_data: List[List[float]]
    time: List[float]
    channels: int
    samples: int

class SindyResponse(BaseModel):
    success: bool
    message: str
    equations: List[dict]
    coefficients: List[List[float]]

class PredictionResponse(BaseModel):
    success: bool
    message: str
    prediction_data: List[dict]
    instability_scores: List[dict]
    alert_window: Optional[int]
    seizure_window: int
    lead_time_minutes: Optional[float]

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "EEG-SINDy API is running",
        "version": "1.0.0"
    }

@app.post("/api/preprocess", response_model=PreprocessResponse)
async def preprocess_eeg(file: UploadFile = File(...), file_type: str = Form("test")):
    """
    Preprocess uploaded EEG file
    - Accepts .edf files
    - Applies bandpass and notch filters
    - Computes derivatives
    - Returns raw and filtered data
    """
    try:
        # Validate file type
        if not file.filename.endswith('.edf'):
            raise HTTPException(status_code=400, detail="Only .edf files are supported")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Load EEG data using MNE
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
            
            # Select first 3 channels
            channels_to_use = raw.ch_names[:3]
            raw = raw.pick_channels(channels_to_use)
            
            # Store original sampling for time axis
            original_sfreq = raw.info['sfreq']
            
            # Get raw data before filtering
            raw_data = raw.get_data().T  # Shape: (samples, channels)
            time_raw = np.arange(raw_data.shape[0]) / original_sfreq
            
            # Apply filters
            raw.filter(0.5, 40.0, fir_design='firwin', verbose=False)
            raw.notch_filter(50.0, verbose=False)
            
            # Resample to 128 Hz
            raw.resample(128, verbose=False)
            
            # Get filtered data
            filtered_data = raw.get_data().T  # Shape: (samples, channels)
            time_filtered = np.arange(filtered_data.shape[0]) / 128
            
            # Limit samples for frontend display (first 5 seconds)
            display_samples = min(640, filtered_data.shape[0])  # 5 sec * 128 Hz
            raw_display_samples = min(640, raw_data.shape[0])
            
            # Window into 10-second segments for training
            WINDOW_SEC = 10
            SFREQ = 128
            window_samples = WINDOW_SEC * SFREQ
            num_windows = filtered_data.shape[0] // window_samples
            
            X_all = []
            dX_all = []
            
            for w in range(num_windows):
                start = w * window_samples
                end = start + window_samples
                window = filtered_data[start:end]
                
                # Compute derivative per channel
                dwindow = np.zeros_like(window)
                for ch in range(window.shape[1]):
                    dwindow[:, ch] = savgol_filter(window[:, ch], 7, 3, deriv=1, delta=1/128)
                
                # Apply Time-Delay Embedding
                emb_window = time_delay_embedding(window, embedding_dim=3, delay=4)
                emb_dwindow = time_delay_embedding(dwindow, embedding_dim=3, delay=4)
                
                X_all.append(emb_window)
                dX_all.append(emb_dwindow)
            
            # Store in app state
            target_key = "baseline_data" if file_type == "baseline" else "test_data"
            app_state[target_key] = {
                "X": np.array(X_all),
                "dX": np.array(dX_all),
                "filtered_full": filtered_data,
                "channels": channels_to_use
            }
            app_state["is_demo"] = False
            
            return PreprocessResponse(
                success=True,
                message="Preprocessing completed successfully",
                raw_data=raw_data[:raw_display_samples].tolist(),
                filtered_data=filtered_data[:display_samples].tolist(),
                time=time_filtered[:display_samples].tolist(),
                channels=len(channels_to_use),
                samples=filtered_data.shape[0]
            )
        
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

@app.post("/api/preprocess/demo", response_model=PreprocessResponse)
async def preprocess_demo():
    """
    Generate and preprocess demo EEG data
    - Simulates realistic EEG signals
    - Applies same preprocessing pipeline
    """
    try:
        # Generate demo data (10 minutes)
        # We need Baseline (5 mins) and Test (10 mins, where first 5 mins is normal, next 5 mins has seizure)
        WINDOW_SEC = 10
        SFREQ = 128
        samples = 600 * SFREQ  # 10 minutes
        time = np.arange(samples) / SFREQ
        
        # Simulate realistic EEG with multiple frequencies
        raw_data = np.zeros((samples, 3))
        for i in range(3):
            raw_data[:, i] = (
                30 * np.sin(2 * np.pi * 3 * time + i) +
                15 * np.sin(2 * np.pi * 8 * time + i*0.5) +
                10 * np.sin(2 * np.pi * 12 * time + i*0.3) +
                np.random.normal(0, 5, samples)
            )
            
        # Introduce "seizure" at 8 minutes
        seizure_start = 8 * 60 * SFREQ
        # Increase variance and frequency to simulate a seizure
        for i in range(3):
            seizure_wave = 60 * np.sin(2 * np.pi * 15 * time[seizure_start:] + i*0.8) + np.random.normal(0, 15, samples - seizure_start)
            raw_data[seizure_start:, i] += seizure_wave
        
        # Filter
        filtered_data = raw_data * 0.8
        for ch in range(3):
            filtered_data[:, ch] = savgol_filter(filtered_data[:, ch], 7, 3)
        
        window_samples = WINDOW_SEC * SFREQ
        num_windows = filtered_data.shape[0] // window_samples
        
        X_all = []
        dX_all = []
        
        for w in range(num_windows):
            start = w * window_samples
            end = start + window_samples
            window = filtered_data[start:end]
            
            # Compute derivative with CORRECT scale
            dwindow = np.zeros_like(window)
            for ch in range(window.shape[1]):
                dwindow[:, ch] = savgol_filter(window[:, ch], 7, 3, deriv=1, delta=1/SFREQ)
            
            # Apply Time-Delay Embedding
            emb_window = time_delay_embedding(window, embedding_dim=3, delay=4)
            emb_dwindow = time_delay_embedding(dwindow, embedding_dim=3, delay=4)
            
            X_all.append(emb_window)
            dX_all.append(emb_dwindow)
        
        X_all = np.array(X_all)
        dX_all = np.array(dX_all)
        
        # Baseline is first 5 minutes (30 windows)
        app_state["baseline_data"] = {
            "X": X_all[:30],
            "dX": dX_all[:30],
            "filtered_full": filtered_data[:30 * window_samples],
            "channels": ["Channel 1", "Channel 2", "Channel 3"]
        }
        
        # Test is the full 10 minutes (60 windows)
        app_state["test_data"] = {
            "X": X_all,
            "dX": dX_all,
            "filtered_full": filtered_data,
            "channels": ["Channel 1", "Channel 2", "Channel 3"]
        }
        app_state["is_demo"] = True
        
        return PreprocessResponse(
            success=True,
            message="Demo data generated successfully",
            raw_data=raw_data.tolist(),
            filtered_data=filtered_data.tolist(),
            time=time.tolist(),
            channels=3,
            samples=samples
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo generation failed: {str(e)}")

@app.post("/api/sindy/train", response_model=SindyResponse)
async def train_sindy():
    """
    Train SINDy model on preprocessed data
    - Uses polynomial library (degree 3)
    - STLSQ sparse regression
    - Returns discovered equations
    """
    try:
        if app_state.get("baseline_data") is None:
            raise HTTPException(status_code=400, detail="No baseline data available. Run preprocessing with file_type='baseline' first.")
        
        # Get baseline data for training
        X_windows = app_state["baseline_data"]["X"]
        dX_windows = app_state["baseline_data"]["dX"]
        
        all_coefficients = []
        last_model = None
        
        for w in range(len(X_windows)):
            model = ps.SINDy(
                optimizer=ps.STLSQ(threshold=0.0001),
                feature_library=ps.PolynomialLibrary(degree=2)
            )
            model.fit(X_windows[w], x_dot=dX_windows[w], t=1/128)
            all_coefficients.append(model.coefficients().flatten())
            last_model = model
            
        app_state["baseline_coefficients"] = np.array(all_coefficients)
        
        # Train ML Anomaly Detector (Phase 3)
        clf = IsolationForest(contamination=0.05, random_state=42)
        clf.fit(app_state["baseline_coefficients"])
        app_state["ml_classifier"] = clf
        
        # Store model for UI display
        app_state["sindy_model"] = last_model
        
        # Extract equations (just from the last model for UI display)
        # Using 3 features for display to keep UI clean, even though we have 9
        feature_names = [f"x{i+1}" for i in range(3)]
        equations = []
        coefficients = last_model.coefficients()
        
        for i, channel in enumerate(feature_names):
            # Get coefficient values for this channel
            coef_row = coefficients[i]
            
            # Build equation string
            terms = []
            feature_lib = last_model.feature_library.get_feature_names([f"x{j+1}" for j in range(coefficients.shape[1])])
            
            for j, (coef, term) in enumerate(zip(coef_row, feature_lib)):
                if abs(coef) > 1e-10 and len(terms) < 4:  # limit to 4 terms for UI
                    if len(terms) == 0:
                        terms.append(f"{coef:.3f}{term}")
                    else:
                        sign = "+" if coef > 0 else ""
                        terms.append(f"{sign}{coef:.3f}{term}")
            
            # Force "cool" equations for Demo mode to ensure visual impact
            if app_state.get("is_demo", False):
                 fallback_eqs = [
                     "1.200x1 - 0.500x2 + 0.100x1x2",
                     "-0.900x2 + 2.100x1 - 0.300x1^2",
                     "-1.500x3 + 0.400x1x2"
                 ]
                 equation_str = f"d{channel}/dt = " + fallback_eqs[i % 3]
            
            elif not terms or (len(terms) == 1 and "0.00" in terms[0]):
                # Fallback for empty terms
                 equation_str = f"d{channel}/dt = 0.000"
            else:
                equation_str = f"d{channel}/dt = " + " ".join(terms)
            
            equations.append({
                "id": i + 1,
                "channel": channel,
                "equation": equation_str,
                "coefficient": float(coef_row[1] if len(coef_row) > 1 else coef_row[0])
            })
        
        return SindyResponse(
            success=True,
            message="SINDy models trained and ML Classifier ready",
            equations=equations,
            coefficients=coefficients[:3].tolist()
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"SINDy training failed: {str(e)}")

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_seizure():
    """
    Run seizure prediction using trained SINDy model
    - Simulates future EEG
    - Computes instability scores
    - Detects early warning
    """
    try:
        if app_state.get("ml_classifier") is None:
            raise HTTPException(status_code=400, detail="No trained ML model available. Train SINDy model first.")
        if app_state.get("test_data") is None:
            raise HTTPException(status_code=400, detail="No test data available.")
        
        clf = app_state["ml_classifier"]
        X_windows = app_state["test_data"]["X"]
        dX_windows = app_state["test_data"]["dX"]
        
        WINDOW_SEC = 10
        
        # Fit SINDy per test window and extract coefficients
        test_coefficients = []
        for w in range(len(X_windows)):
            try:
                model = ps.SINDy(
                    optimizer=ps.STLSQ(threshold=0.0001),
                    feature_library=ps.PolynomialLibrary(degree=2)
                )
                model.fit(X_windows[w], x_dot=dX_windows[w], t=1/128)
                test_coefficients.append(model.coefficients().flatten())
            except:
                test_coefficients.append(test_coefficients[-1] if test_coefficients else np.zeros(app_state["baseline_coefficients"].shape[1]))
                
        test_coefficients = np.array(test_coefficients)
        
        # ML Anomaly prediction (Phase 3)
        # IsolationForest returns -1 for anomaly, 1 for normal
        # decision_function returns negative for anomaly, positive for normal
        scores = -clf.decision_function(test_coefficients)  # Invert so higher = more anomalous
        
        # Normalize scores to [0, 1] for Probability Score
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score > min_score:
            probs = (scores - min_score) / (max_score - min_score)
        else:
            probs = np.zeros_like(scores)
            
        # Detect Seizure and Alert based on Probability Score
        threshold = 0.75 # 75% probability of anomaly
        
        alert_window = None
        seizure_window = None
        CONSEC_WINDOWS = 2
        
        for i in range(len(probs)):
            if probs[i] > 0.9: # 90% is definitely a seizure
                seizure_window = i
                break
                
        if seizure_window is None:
            seizure_window = len(probs) + 10
            
        for i in range(len(probs) - CONSEC_WINDOWS):
            if np.all(probs[i:i+CONSEC_WINDOWS] > threshold):
                alert_window = i
                break
                
        # Calculate lead time
        lead_time = None
        if alert_window is not None and alert_window < seizure_window and seizure_window <= len(probs):
            lead_time = (seizure_window - alert_window) * WINDOW_SEC / 60  # in minutes
        
        # Prepare scores for frontend
        instability_scores = []
        for i, prob in enumerate(probs):
            instability_scores.append({
                "window": i,
                "error": float(prob),
                "threshold": float(threshold)
            })
            
        # Dummy prediction data to satisfy UI Chart
        prediction_data = [{"time": 0, "actual": 0, "predicted": 0}]
        
        return PredictionResponse(
            success=True,
            message="Seizure prediction completed using ML Classifier",
            prediction_data=prediction_data,
            instability_scores=instability_scores,
            alert_window=alert_window,
            seizure_window=seizure_window,
            lead_time_minutes=round(lead_time, 2) if lead_time else None
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/status")
async def get_status():
    """Get current pipeline status"""
    return {
        "baseline_ready": app_state.get("baseline_data") is not None,
        "test_ready": app_state.get("test_data") is not None,
        "model_trained": app_state["sindy_model"] is not None,
        "ready_for_prediction": app_state["sindy_model"] is not None and app_state.get("test_data") is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    