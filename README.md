# 🧠 EEG Seizure Detection Using SINDy & Machine Learning
# 🧠 EEG Seizure Detection Using SINDy & Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SINDy-orange)
![Domain](https://img.shields.io/badge/Domain-EEG%20Signal%20Processing-purple)
![NLP/ML](https://img.shields.io/badge/Model-Sparse%20Identification-red)
![Status](https://img.shields.io/badge/Status-Research%20Project-green)
![License](https://img.shields.io/badge/License-Research%20&%20Academic-green)

---

> [!CAUTION]
> **MEDICAL DISCLAIMER:** This software is for **academic and research purposes only**. It is NOT a medical device and has not been evaluated by the FDA or any other regulatory agency. This software should **never** be used to diagnose, treat, cure, or prevent any disease, nor should it be used for clinical decision-making or patient monitoring. The developers assume no responsibility for any misuse.

> [!WARNING]
> **AI DISCLOSURE & ACCURACY:** The architecture and codebase of this project were developed with heavy assistance from Artificial Intelligence (AI) coding tools. The accuracy, reliability, and mathematical correctness of the algorithms have not been clinically or rigorously verified. The outputs of this system **cannot be trusted** for real-world scenarios and the system exists purely to be studied, analyzed, and evaluated.

## 📌 Overview
This project implements a **seizure detection system** using Electroencephalogram (EEG) data and modern data-driven modeling techniques.  
It applies **Sparse Identification of Nonlinear Dynamics (SINDy)** along with Machine Learning anomaly detection (IsolationForest) to detect epileptic seizure events from EEG recordings.

### 🏗️ Architecture Pipeline
1. **Time-Delay Coordinate Embedding:** Reconstructs the underlying dynamic state space from raw EEG data, creating a multi-dimensional delayed manifold.
2. **SINDy Coefficient Tracking:** Extracts sparse dynamical equations ($\Xi$) for 10-second sliding windows, capturing the governing dynamics instead of attempting unstable forward integration.
3. **Machine Learning Anomaly Detection:** An IsolationForest classifier learns the normal (healthy) baseline dynamics and outputs a real-time Probability Score indicating deviation toward a seizure (bifurcation).

---

## 🎯 Motivation
EEG-based seizure detection is important for:
- Real-time patient monitoring.
- Automated healthcare systems.
- Reducing manual review workload.
- Enhancing diagnostic accuracy.

This system demonstrates how machine learning and dynamical system modeling can be applied to real-world biomedical signal analysis.

---

## 📂 Project Structure
EEG-Seizure-Detection/
│
├── data/ # EEG dataset files (if included or referenced)
├── preprocessing.py # EEG data cleaning and filtering
├── feature_extraction.py # Feature extraction module
├── sindy_model.py # SINDy model implementation
├── classifier.py # Classification & detection logic
├── visualization.py # Plots and result visualization
│
├── README.md
├── requirements.txt
└── .gitignore

---

## 🧠 Key Modules

### `preprocessing.py`
- Loads raw EEG signals
- Filters noise and artifacts
- Normalizes time series

### `feature_extraction.py`
- Extracts meaningful EEG features
- Frequency domain and time domain characteristics

### `sindy_model.py`
- Builds a SINDy model for underlying EEG dynamics
- Identifies sparse governing equations

### `classifier.py`
- Uses feature and dynamic behavior for seizure classification
- Metrics evaluation

### `visualization.py`
- Plots EEG signals and detection results
- Helps in analysis and interpretation

## ⚙️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/Sindy-Seizure-Predictor.git
   cd Sindy-Seizure-Predictor
   ```

2. **Set up a Virtual Environment:**
   ```bash
   python -m venv backend/venv
   backend\venv\Scripts\activate
   ```

3. **Install Dependencies:**
   Ensure you install the exact versions listed in `requirements.txt` to avoid dependency conflicts:
   ```bash
   pip install pysindy==2.1.0 scikit-learn==1.9.0 fastapi uvicorn mne
   ```

4. **Run the Backend:**
   ```bash
   python -m uvicorn backend.main:app --reload
   ```

5. **Run the UI:**
   Open `frontend/index.html` in your browser.

---

## ⚖️ License
This project is licensed under the custom **Research and Academic Use License**. 

You are free to copy, modify, and distribute this software for educational and non-commercial research purposes, provided that you strictly adhere to the conditions prohibiting clinical use. The developers assume zero liability for any use or misuse of this software. 

See the [LICENSE](LICENSE) file for the full legal text.
