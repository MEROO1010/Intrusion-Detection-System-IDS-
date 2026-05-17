# Intelligent Network Intrusion Detection System (IDS)

An operational machine learning telemetry network monitoring environment parsing NSL-KDD log frames to isolate and intercept network intrusions.

## Execution Matrix Steps

### Phase 1: Environment Assembly
Ensure the raw `KDDTrain+.txt` source dataset file is located inside the `data/raw/` subfolder. Next, set up all dependencies:
```bash
pip install -r requirements.txt