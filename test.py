import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi.testclient import TestClient
    from backend.main import app, app_state
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

client = TestClient(app)

def test_pipeline():
    print("1. Testing Demo Data Generation (/api/preprocess/demo)")
    response = client.post("/api/preprocess/demo")
    assert response.status_code == 200, f"Failed with {response.status_code}: {response.text}"
    data = response.json()
    print(f" - Success! Demo generated {data['samples']} samples.")
    print(f" - Baseline data windows: {len(app_state['baseline_data']['X'])}")
    print(f" - Test data windows: {len(app_state['test_data']['X'])}")
    
    print("\n2. Testing SINDy Training (/api/sindy/train)")
    response = client.post("/api/sindy/train")
    assert response.status_code == 200, f"Failed with {response.status_code}: {response.text}"
    data = response.json()
    print(f" - Success! Model trained.")
    print(f" - First equation: {data['equations'][0]['equation']}")
    
    print("\n3. Testing Seizure Prediction (/api/predict)")
    response = client.post("/api/predict")
    assert response.status_code == 200, f"Failed with {response.status_code}: {response.text}"
    data = response.json()
    print(f" - Success! Prediction ran.")
    print(f" - Detected Seizure Window: {data['seizure_window']}")
    print(f" - Alert Window: {data['alert_window']}")
    print(f" - Lead Time: {data['lead_time_minutes']} minutes")
    
    if data['lead_time_minutes'] is not None and data['lead_time_minutes'] > 0:
        print("\n✅ Verification Successful: System properly detects the seizure and issues an early warning!")
    else:
        print("\n❌ Verification Failed: System did not issue a valid early warning.")

if __name__ == "__main__":
    test_pipeline()
