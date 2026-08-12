# Project Details: SINDy Seizure Prediction
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SINDy-orange)
![Domain](https://img.shields.io/badge/Domain-EEG%20Signal%20Processing-purple)
![NLP/ML](https://img.shields.io/badge/Model-Sparse%20Identification-red)
![Status](https://img.shields.io/badge/Status-Research%20Project-green)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

## 1. Project Overview
**Title:** EEG-SINDy Seizure Prediction Scope

This project is a sophisticated web application designed to detect and predict epileptic seizures using **Sparse Identification of Nonlinear Dynamics (SINDy)**. By applying data-driven discovery of physical dynamics to Electroencephalogram (EEG) signals, the system moves beyond traditional black-box AI to offer an interpretable, robust, and mathematically grounded approach to seizure forecasting.

The application allows users (doctors, researchers, or patients) to upload raw EEG data (`.edf` format), processes the signals, learns the underlying governing equations of the brain's electrical activity, and identifies "instability" precursors that warn of an impending seizure.

---

## 2. Problem Statement
**The Challenge:**
Epilepsy affects millions worldwide, and the unpredictability of seizures poses a significant risk to patient safety and quality of life.
*   **Unpredictability:** Seizures often strike without warning.
*   **Black-Box Limitations:** Traditional Deep Learning models (CNNs/RNNs) can predict seizures but lack interpretability—clinicians cannot understand *why* a prediction was made.
*   **Complexity:** EEG data is high-dimensional, noisy, and chaotic, making it difficult to model with standard linear tools.

**Our Goal:** To build a transparent, interpretable early-warning system that can predict seizures minutes in advance, allowing for timely intervention.

---

## 3. Our Novelty & Methodology
The core innovation of this project lies in its **White-Box AI** approach.

### A. Sparse Identification of Nonlinear Dynamics (SINDy)
Instead of training a neural network to classify "Normal" vs. "Seizure," we use **PySINDy** to discover the **differential equations** that govern the brain's electrical dynamics.
*   **Equation Discovery:** The system identifies a sparse set of terms (e.g., $ \dot{x} = -0.5x + 0.1x^3 $) that best describe the evolution of the EEG signal over time.
*   **Physics-Informed:** We treat the brain as a dynamical system, extracting the "laws of motion" for the EEG signal.

### B. Instability as a Biomarker
We posit that a healthy brain state follows specific stable dynamics, while a pre-seizure state exhibits localized instability or deviation from these dynamics.
*   **Mechanism:** We train a SINDy model on a window of EEG data. We then use this model to simulate the future trajectory of the signal.
*   **Detection:** We compare the model's simulation with the actual incoming signal. A high **Mean Squared Error (MSE)** indicates that the brain's dynamics have shifted (bifurcation), signalling an imminent seizure.

---

## 4. Technology Stack

### Frontend (User Interface)
*   **Framework:** Vanilla HTML5 & JavaScript (No heavy frameworks required for speed/simplicity).
*   **Styling:** Custom CSS3 with **Glassmorphism** design system, animations, and gradients.
*   **Visualization:** `Chart.js` for real-time rendering of EEG signals and instability metrics.
*   **Design:** Modern, clean UI with the "Outfit" font family.

### Backend (Core Logic)
*   **Server:** **FastAPI** (Python) - High-performance async web framework.
*   **Server Runner:** Uvicorn.
*   **API Design:** RESTful endpoints for preprocessing, training, and prediction.

### Data Science & Machine Learning
*   **Core Algorithm:** **PySINDy** (Python package for SINDy).
*   **EEG Processing:** **MNE-Python** (Gold standard for neurophysiological data).
*   **Signal Processing:** `SciPy` (Savgol filters), `NumPy`.
*   **Data Handling:** `Pandas`, `pyEDFlib` (for reading .edf files).

---

## 5. Key Features

1.  **Drag & Drop EEG Upload:** Seamlessly handles standard `.edf` medical files.
2.  **Automated Signal Pipeline:**
    *   **Bandpass Filter:** 0.5Hz - 40Hz (removes drift and high-freq noise).
    *   **Notch Filter:** 50Hz (removes power line interference).
    *   **Resampling:** Downsamples to 128Hz for computational efficiency.
3.  **Real-Time Math Discovery:** Displays the actual coefficients and equations discovered from the patient's brain waves.
4.  **Lead Time Estimation:** Calculates and displays the estimated time (in minutes) before a seizure event based on instability thresholds.
5.  **Interactive Dashboard:**
    *   Instability Score Chart (tracking risk over time).
    *   Signal Preview (visualizing the filtered EEG).
    *   Risk Badges (High/Low Risk).
6.  **Demo Mode:** capable of generating synthetic multi-frequency EEG data to demonstrate the pipeline without needing a real file.

---

## 6. Project Architecture
1.  **User** uploads `.edf` file via Frontend.
2.  **FastAPI** receives file and uses `MNE` to filter and window the data.
3.  **PySINDy** trains on the windowed data to find the best-fit polynomial equations.
4.  **Simulation Engine** uses the learned equations to predict the next time-steps.
5.  **Instability Checker** compares Simulation vs. Reality. Large divergence counts as an "Error".
6.  **Frontend** polls for results and visualizes the Error trajectory to warn the user.
