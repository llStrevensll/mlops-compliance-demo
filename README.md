# ComplianceML — End-to-End MLOps Demo

A hands-on MLOps project that takes a machine-learning model **from data to production**:
training with experiment tracking, model registry, containerized serving on Kubernetes,
automated retraining via CI/CD, and drift monitoring with observability dashboards.

> **Domain:** synthetic *compliance-risk scoring* (classify audit findings as high/low risk).
> All data is **synthetic** — generated for this demo, no real or proprietary data.

## Architecture

```
GitHub Actions (CI/CD) ── train → evaluate → quality gate → register
                                      │
   data ──► train.py (XGBoost) ──► MLflow (Tracking + Model Registry)
                                      │ pull "Production" model
                                      ▼
                 BentoML (REST API) ──► Kubernetes (kind) serving
                                      │ predictions + new data
                                      ▼
              Evidently (drift) ──► Prometheus ──► Grafana dashboards
```

## Tech Stack

| Area | Tool |
|---|---|
| Experiment tracking & registry | MLflow |
| Model | scikit-learn / XGBoost |
| Model serving | BentoML on Kubernetes (kind) |
| CI/CD for ML | GitHub Actions |
| Drift monitoring | Evidently |
| Observability | Prometheus + Grafana |
| Infrastructure as Code | Terraform |
| Cloud variant (optional) | AWS SageMaker |

## Status

🚧 Work in progress — built incrementally, commit by commit.

- [x] Project skeleton (uv, gitignore, README)
- [ ] Synthetic dataset
- [ ] Training + MLflow tracking & registry
- [ ] BentoML serving
- [ ] Kubernetes deployment (kind)
- [ ] CI/CD pipeline
- [ ] Drift monitoring + dashboards

## Getting Started

```bash
uv sync          # install dependencies
uv run mlflow ui # open MLflow at http://localhost:5000
```
