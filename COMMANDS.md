# Commands

Reference for running every part of the project and where outputs are saved.
All commands run from the repo root and use **uv** (no manual virtualenv needed).

## 0. Setup (once)

```bash
uv sync          # creates .venv and installs the exact pinned dependencies
```

## 1. Generate the synthetic dataset

```bash
uv run python src/generate_data.py
```

| | |
|---|---|
| What it does | Generates 5000 synthetic compliance findings (seed=42, reproducible). |
| Output | `data/compliance_findings.csv` (git-ignored — only the generator is versioned). |

## 2. Train the model + log to MLflow

```bash
uv run python src/train.py
```

| | |
|---|---|
| What it does | Trains an XGBoost classifier, logs params/metrics, registers the model. |
| Output (metadata) | `mlflow.db` — SQLite backend: experiments, runs, params, metrics, **model registry**. |
| Output (artifacts) | `mlruns/<exp>/models/...` — the serialized model files. |
| Registered model | `compliance-risk-model` (a new **version** is created on every run). |

> Both `mlflow.db` and `mlruns/` are git-ignored (local artifacts, not versioned).

## 3. Explore results in the MLflow UI

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open <http://localhost:5000>.

> ⚠️ The `--backend-store-uri sqlite:///mlflow.db` flag is **required**: runs are
> stored in `mlflow.db`, so a plain `mlflow ui` would point at the empty file store
> and show nothing. (`train.py` pins the same URI so everything stays consistent.)

In the UI you can see:
- The `compliance-risk` **experiment** and each run (params + metrics).
- The **Models** tab with `compliance-risk-model` and its versions.

## 4. Serve the model as a REST API (BentoML)

```bash
# 4.1 Import the registered MLflow model into the BentoML model store (once per new version)
uv run python src/import_model.py

# 4.2 Start the API server (port 3000 is taken by Docker here, so we use 7777)
uv run bentoml serve service:ComplianceRiskService --port 7777
```

Test it with `curl`:

```bash
curl -X POST http://localhost:7777/predict \
  -H "Content-Type: application/json" \
  -d '{"finding": {"severity":5,"days_open":120,"control_failures":4,
       "affected_systems":6,"is_repeat_finding":1,"has_remediation_plan":0,
       "framework":"SOC2","department":"IT"}}'
# -> {"risk_high": 1, "risk_label": "HIGH"}
```

| | |
|---|---|
| What it does | Loads the model and exposes `POST /predict` (raw finding in, risk out). |
| Model source | BentoML model store (`compliance_risk:latest`), imported from MLflow. |
| Interactive docs | Open <http://localhost:7777> for the auto-generated Swagger UI. |

## Where things live (summary)

| Path | Content | Versioned in git? |
|---|---|---|
| `src/` | Source code (generator, training) | ✅ yes |
| `data/compliance_findings.csv` | Generated dataset | ❌ ignored |
| `mlflow.db` | MLflow backend (runs + registry) | ❌ ignored |
| `mlruns/` | MLflow model artifacts | ❌ ignored |
| `docs/` | Architecture & diagrams | ✅ yes |
