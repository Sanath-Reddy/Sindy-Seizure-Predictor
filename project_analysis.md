# Project Analysis & Critique: EEG-SINDy Seizure Prediction

After a detailed investigation of the codebase (including the FastAPI backend, the frontend index, and the original reference scripts), we have analyzed whether the project works as claimed and identified several major flaws.

---

## Executive Summary

The project **does not do the job correctly** and **cannot work as claimed** on real-world data. While it presents a visually impressive glassmorphic UI with dynamic charts, the underlying machine learning, mathematical modeling, and prediction logic are heavily flawed and, in some places, fabricated.

Specifically:
1. **The seizure prediction is fake/hardcoded**: The point of seizure onset is hardcoded to window `297` (or near the end of the file), rather than being detected dynamically from the data.
2. **There is a critical mathematical timescale mismatch**: SINDy is trained on a sample-based timescale but simulated on a second-based timescale. As a result, the SINDy simulation does not predict the future at all; the "instability score" is merely tracking the raw variance/amplitude of the EEG signal.
3. **The evaluation is mathematically flawed**: The model is trained and tested on the exact same file, and the alert threshold is defined in a way that guarantees an early false alarm on any input.

---

## Detailed Findings

### 1. Hardcoded Seizure Onset (Fabricated Logic)
The backend claims to compute the "lead time" before a seizure by detecting when the brain's dynamics diverge from a stable state. However, in [backend/main.py](file:///c:/Users/sanat/Desktop/sem%203%20junk/SINDY%20FINAL/Sindy-EL/backend/main.py#L394-L400), the seizure location is completely hardcoded:

```python
# Simulate seizure at window 297 (or near end)
seizure_window = min(297, len(errors) - 10)

# Calculate lead time
lead_time = None
if alert_window is not None:
    lead_time = (seizure_window - alert_window) * WINDOW_SEC / 60  # in minutes
```

* **The Issue**: No matter what EEG recording the user uploads, the backend assumes a seizure occurs at exactly window `297` (representing 2,970 seconds or 49.5 minutes of recording) or 10 windows before the end of the file. 
* **The Consequence**: If the uploaded EEG contains a seizure at a different time, or has no seizure at all, the application will still display a warning indicating an impending seizure at this hardcoded index and show a fake "lead time in minutes."

---

### 2. SINDy Timescale Mismatch (Mathematical Bug)
The core SINDy differential equation discovery and simulation logic contains a critical bug in how numerical derivatives and timescales are managed.

#### Step A: Derivative Calculation
In [backend/main.py](file:///c:/Users/sanat/Desktop/sem%203%20junk/SINDY%20FINAL/Sindy-EL/backend/main.py#L140-L142), the derivative of the signal is calculated using SciPy's Savitzky-Golay filter:
```python
# Compute derivative per channel
dwindow = np.zeros_like(window)
for ch in range(window.shape[1]):
    dwindow[:, ch] = savgol_filter(window[:, ch], 7, 3, deriv=1)
```
* **The Error**: By default, `savgol_filter` assumes a sample spacing of `delta = 1.0` if it is not specified. Since the EEG data is sampled at 128 Hz, the actual time step is $dt = 1/128$ seconds. 
* **The Consequence**: The calculated derivative is $\frac{dx}{ds}$ (change per *sample*), which is **128 times smaller** than the actual physical derivative $\frac{dx}{dt}$ (change per *second*).

#### Step B: SINDy Fitting
The backend feeds this derivative directly into SINDy:
```python
model.fit(X_train, x_dot=dX_train, t=1/128)
```
* **The Error**: When `x_dot` is explicitly provided, PySINDy uses it as the target for regression and bypasses its internal differentiation. Thus, the model learns the system $\frac{dx}{ds} = f(x)$ (dynamics per sample).

#### Step C: Integration and Simulation
In [backend/main.py](file:///c:/Users/sanat/Desktop/sem%203%20junk/SINDY%20FINAL/Sindy-EL/backend/main.py#L350-L361), prediction errors are calculated by simulating the learned system over time:
```python
WINDOW_SEC = 10
SFREQ = 128
t = np.arange(0, WINDOW_SEC, 1/SFREQ)  # [0, 0.0078, 0.0156, ..., 10.0] (1280 steps)
...
sim = model.simulate(window[0], t)
```
* **The Error**: The integration is performed over the time vector `t` which spans from $0$ to $10$. However, because the learned model's equations are in terms of *samples*, integrating for $10$ units of time only simulates the dynamics for **10 samples** (which is just $10/128 \approx 0.078$ seconds of real time).
* **The Consequence**: The simulated trajectory `sim` barely deviates from its initial condition `window[0]`. It essentially produces a near-constant line. 
* **What the "MSE" actually measures**: Since `sim` is flat, the Mean Squared Error (MSE) calculation:
  $$\text{MSE} \approx \text{mean}((window[0] - window[:])^2)$$
  is simply measuring the **variance/amplitude** of the EEG signal in each window. The SINDy model is doing no actual prediction; the chart is just plotting signal variance masked as a "SINDy Instability Score."

---

### 3. In-Sample Training & Testing (Methodological Flaw)
SINDy is designed to discover the governing dynamics of a system. The methodology of SINDy-based seizure prediction requires:
1. Learning normal brain dynamics from a **healthy baseline recording** (e.g., `chb01_01.edf`).
2. Running prediction on an **unseen test recording** (e.g., `chb01_03.edf`). If the test data starts deviating from the baseline equations, the prediction error rises, indicating a transition to a seizure state.

* **What the backend does**: In `main.py`, the backend trains SINDy on the *same uploaded file* that it uses for testing. 
* **The Consequence**: If the uploaded file contains a seizure, SINDy will train on both the normal and seizure periods. The model will try to fit the entire signal (meaning the prediction error will not systematically rise in the pre-seizure window as expected, or it will just fit an average of both states). This violates the core validation methodology of PySINDy.

---

### 4. Guaranteed Early False Alerts (Statistical Flaw)
The threshold for raising a risk alert is calculated in [backend/main.py](file:///c:/Users/sanat/Desktop/sem%203%20junk/SINDY%20FINAL/Sindy-EL/backend/main.py#L381-L392):
```python
# Determine threshold (95th percentile of first 50 windows)
baseline_errors = errors[:min(50, len(errors))]
threshold = np.percentile(baseline_errors, 95)
...
# Detect alert (3 consecutive windows above threshold)
for i in range(len(errors) - CONSEC_WINDOWS):
    if np.all(errors[i:i+CONSEC_WINDOWS] > threshold):
        alert_window = i
        break
```
* **The Issue**: The threshold is the 95th percentile of the first 50 windows of the *same* file. Mathematically, $5\%$ of the windows in the baseline period are guaranteed to exceed this threshold by definition.
* **The Consequence**: Because EEG signals are highly correlated over time, it is almost statistically certain that a few windows near the beginning will exceed the threshold consecutively. As a result, **the system will almost always trigger a false alert within the first 50 windows (first ~8 minutes)**, even on perfectly healthy brain activity.

---

### 5. Broken Demo Mode and Frontend Limitations
* **Demo Endpoint is Dead**: The backend implements `/api/preprocess/demo` to generate synthetic multi-frequency EEG data. However, the frontend [frontend/index.html](file:///c:/Users/sanat/Desktop/sem%203%20junk/SINDY%20FINAL/Sindy-EL/frontend/index.html) does not have a button or logic to call this endpoint.
* **Demo Data is Too Short**: The demo endpoint only generates 1,280 samples (exactly one 10-second window). If a user invokes this endpoint manually, the backend fails to run prediction properly because it cannot establish a baseline threshold over a single window (resulting in a negative seizure window `min(297, 1 - 10) = -9` and a null lead time).

---

## Comparison with Reference Code

The reference code in the `Seizure-Prediction-using-SINDy-main` directory has the correct logic:
1. In `preprocess.py`, it processes `chb01_01.edf` and `chb01_02.edf` to create a **separate baseline dataset** (`X_baseline.npy`).
2. In `predictive_sindy.py`, SINDy is trained **only** on the baseline dataset and *without* manual derivative inputs (`t=1/SFREQ` is passed, allowing PySINDy to correctly compute physical derivatives in seconds):
   ```python
   model.fit(X_train, t=1/SFREQ)
   ```
3. It then tests the model on the seizure file (`X_test`) and computes errors. Since the seizure onset is known in `chb01_03.edf` (window 297), it compares the detected alert window to window 297 to report the lead time.
4. The web backend (`main.py`) attempted to convert this file-specific research script into a generic web app but did so by hardcoding the variables and introducing timescale bugs.

---

## Recommendations to Fix the Project

If you want to make this project work correctly, the following changes are required:
1. **Fix Derivative Scaling**: Add `delta=1/128` to `savgol_filter` in `main.py` so that derivatives match the time scale of the simulation.
2. **Implement Baseline Profile Selection**: Allow users to upload or select a separate "Baseline EEG Recording" (for training the SINDy model) and a "Test EEG Recording" (for predicting/monitoring).
3. **Dynamic Seizure Detection**: Instead of hardcoding `seizure_window = 297`, implement a second-stage classifier (e.g., thresholding, variance checks, or a simple classifier on SINDy coefficients) to find where the actual seizure begins in the test file, rather than hardcoding it.
4. **Connect Demo Mode**: Add a "Try Demo Data" button on the frontend that calls `/api/preprocess/demo` and ensure the demo endpoint generates a longer signal (e.g., 10 minutes of data / 60 windows) so the charts and thresholds compute correctly.
