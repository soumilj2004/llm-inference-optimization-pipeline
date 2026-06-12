# Agentic LLM Compression Lab

Current phase: stable FP32 vs magnitude-pruned experiment workflow.

The active experiment loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, compares the
FP32 reference model against an unstructured magnitude-pruned variant, evaluates
quality against FP32 references, and saves metrics/report files in `results/`.

Active methods:

- `fp32`
- `pruned`

Measured metrics:

- Parameter count
- RAM usage before model load
- RAM usage after model load
- Inference latency
- Generated token count
- Tokens per second

## Setup

```powershell
py -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
venv\Scripts\python.exe agents\experiment_manager.py
```

## Dashboard

```powershell
streamlit run dashboard/app.py
```
