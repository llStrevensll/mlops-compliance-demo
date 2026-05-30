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

## Where things live (summary)

| Path | Content | Versioned in git? |
|---|---|---|
| `src/` | Source code (generator, training) | ✅ yes |
| `data/compliance_findings.csv` | Generated dataset | ❌ ignored |
| `mlflow.db` | MLflow backend (runs + registry) | ❌ ignored |
| `mlruns/` | MLflow model artifacts | ❌ ignored |
| `docs/` | Architecture & diagrams | ✅ yes |
