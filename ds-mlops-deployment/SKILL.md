---
name: ds-mlops-deployment
description: |
  Deploys trained ML models to production and keeps them healthy over time. Use this skill whenever a model needs to go from a notebook or .joblib file into something that actually runs in the real world — as an API, a batch job, a scheduled pipeline, or an embedded service. Also use it for model monitoring, drift detection, retraining triggers, experiment tracking with MLflow, model registries, and CI/CD for ML workflows. Trigger when the user says "deploy my model", "put this in production", "build a model API", "my model is drifting", "set up monitoring", "automate retraining", "containerize my model", "how do I version my models", "MLflow", "model registry", "serving infrastructure", "shadow deployment", "A/B test in production", or anything about taking a working model and making it operational. This skill picks up where ds-ml-pipeline leaves off — once you have a serialized pipeline, this skill handles what happens next.
---

# MLOps & Deployment Skill

This skill covers the full journey from a trained, serialized model to a production system that's deployed, monitored, and maintained over time. The pipeline skill gets you to `joblib.dump(pipeline, 'model.joblib')` — this skill takes over from there.

## Why Deployment is Its Own Discipline

Training a model is 20% of the work. The other 80% is everything that happens after: packaging it so it runs reliably anywhere, exposing it to consumers, watching it behave in the real world, and keeping it accurate as the world changes. A model that lives only in a Jupyter notebook has zero business value. The goal of this skill is to close that gap.

---

## The MLOps Lifecycle

```
┌──────────────────────────────────────────────────────────────────┐
│                    MLOps Lifecycle                                │
│                                                                  │
│  1. EXPERIMENT      2. REGISTER        3. SERVE                  │
│  Track runs &  →    Version &      →   API / Batch /             │
│  metrics            validate           Embedded                  │
│                                                                  │
│  4. MONITOR         5. RETRAIN         6. GOVERN                 │
│  Drift &       →    Auto/manual   →    Audit, docs,              │
│  performance        trigger            rollback                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Experiment Tracking with MLflow

Before deployment, you need a paper trail of what you trained and why you chose it. MLflow is the standard open-source tool for this.

### Setting Up MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

mlflow.set_experiment("churn_prediction_v1")

with mlflow.start_run(run_name="gbm_tuned_v3"):
    # Log parameters
    params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6}
    mlflow.log_params(params)

    # Train
    model = GradientBoostingClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Log metrics
    mlflow.log_metrics({
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        "test_accuracy": model.score(X_test, y_test)
    })

    # Log the model itself
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="churn_predictor"
    )
    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

Key things to always log: the parameters you trained with, the metrics that matter for the business case, the training data version (even just a hash or date), and the model artifact itself. Without this, you'll spend hours later trying to reproduce "the model that worked in March."

### Promoting a Model to Production

MLflow's model registry has four stages: `None` → `Staging` → `Production` → `Archived`.

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Promote best run to Staging
client.transition_model_version_stage(
    name="churn_predictor",
    version=3,
    stage="Staging",
    archive_existing_versions=False
)

# After validation, promote to Production
client.transition_model_version_stage(
    name="churn_predictor",
    version=3,
    stage="Production",
    archive_existing_versions=True  # archive previous prod version
)
```

Never promote directly to Production from a notebook. Always go through Staging first and validate on a held-out or recent time window.

---

## Phase 2: Serving — Choosing the Right Pattern

| Pattern | When to use | Latency | Throughput | Complexity |
|---|---|---|---|---|
| **REST API (FastAPI)** | Real-time, single predictions on demand | Low (ms) | Medium | Low |
| **Batch job** | Scheduled scoring of large datasets | High (minutes) | Very high | Low |
| **Streaming** | Event-driven, continuous scoring | Low | High | High |
| **Embedded** | Edge devices, mobile, no-network | Very low | N/A | Medium |

For most DS projects, start with a REST API or batch job. Only add streaming complexity when you actually need real-time event processing.

### Building a FastAPI Model Endpoint

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Churn Prediction API", version="1.0.0")

# Load model once at startup (not per-request)
MODEL_URI = "models:/churn_predictor/Production"
model = mlflow.pyfunc.load_model(MODEL_URI)

class PredictionRequest(BaseModel):
    customer_id: str
    tenure_months: float
    monthly_charges: float
    contract_type: str
    # ... other features

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    model_version: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Build feature row
        features = pd.DataFrame([{
            "tenure_months": request.tenure_months,
            "monthly_charges": request.monthly_charges,
            "contract_type": request.contract_type,
        }])

        # Score
        prob = model.predict(features)[0]
        logger.info(f"Predicted churn_prob={prob:.3f} for customer={request.customer_id}")

        return PredictionResponse(
            customer_id=request.customer_id,
            churn_probability=float(prob),
            churn_prediction=bool(prob > 0.5),
            model_version="churn_predictor/3"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

Run it: `uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4`

**Critical API design decisions:**
- Always validate input with Pydantic. Missing or wrong-type features cause silent failures without it.
- Log every prediction with its inputs and outputs. You'll need this for debugging and for retraining.
- Return a prediction alongside a model version identifier. When something goes wrong in production, you need to know exactly which model made which prediction.
- Use a `/health` endpoint — load balancers and Kubernetes need this.

### Batch Scoring Pattern

For cases where you score large datasets on a schedule (nightly, hourly), a batch job is simpler, cheaper, and more auditable than an API.

```python
# batch_score.py
import pandas as pd
import joblib
import mlflow.pyfunc
from datetime import datetime

def run_batch_scoring(
    input_path: str,
    output_path: str,
    model_name: str = "churn_predictor",
    stage: str = "Production"
):
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")

    df = pd.read_parquet(input_path)
    df['churn_probability'] = model.predict(df[FEATURE_COLS])
    df['scored_at'] = datetime.utcnow()
    df['model_version'] = f"{model_name}/{stage}"

    df.to_parquet(output_path, index=False)
    print(f"Scored {len(df):,} rows → {output_path}")

if __name__ == "__main__":
    run_batch_scoring(
        input_path="data/customers_to_score.parquet",
        output_path="data/churn_scores.parquet"
    )
```

---

## Phase 3: Containerization with Docker

Docker ensures your model runs identically everywhere — your laptop, staging, and production.

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY model_artifacts/ ./model_artifacts/

# Non-root user for security
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```
# requirements.txt
fastapi==0.111.0
uvicorn[standard]==0.29.0
pandas==2.2.0
scikit-learn==1.4.0
mlflow==2.13.0
pydantic==2.7.0
```

Build and run:
```bash
docker build -t churn-predictor:v1 .
docker run -p 8000:8000 churn-predictor:v1
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"customer_id": "C123", "tenure_months": 24, ...}'
```

---

## Phase 4: Monitoring — The Part Everyone Skips

Deploying a model without monitoring is like flying without instruments. Models degrade silently.

### What to Monitor

**Model performance metrics** — Requires ground truth (labels). Often delayed.
```python
# After ground truth arrives (e.g., 30 days after churn prediction)
from sklearn.metrics import f1_score
import pandas as pd

def calculate_production_metrics(scored_df, ground_truth_df):
    merged = scored_df.merge(ground_truth_df, on='customer_id')
    metrics = {
        "f1": f1_score(merged['actual_churn'], merged['churn_prediction']),
        "date": pd.Timestamp.now().date().isoformat(),
        "n_predictions": len(merged)
    }
    return metrics
```

**Data drift** — Detects when input features shift from the training distribution. This is often the *first* signal that performance is degrading, before you have labels.

```python
from scipy.stats import ks_2samp
import numpy as np

def detect_feature_drift(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    numeric_features: list,
    p_value_threshold: float = 0.05
) -> dict:
    """
    Kolmogorov-Smirnov test for each numeric feature.
    Small p-value = significant drift detected.
    """
    drift_report = {}
    for feature in numeric_features:
        stat, p_value = ks_2samp(
            reference_df[feature].dropna(),
            production_df[feature].dropna()
        )
        drift_report[feature] = {
            "ks_statistic": round(stat, 4),
            "p_value": round(p_value, 4),
            "drift_detected": p_value < p_value_threshold
        }
    drifted = [f for f, v in drift_report.items() if v['drift_detected']]
    print(f"Drift detected in {len(drifted)}/{len(numeric_features)} features: {drifted}")
    return drift_report
```

**Prediction distribution** — If the fraction of predicted positives suddenly changes, something is wrong even before you have labels.

```python
def check_prediction_distribution(current_scores: pd.Series, baseline_positive_rate: float):
    current_rate = current_scores.mean()
    ratio = current_rate / baseline_positive_rate
    alert = ratio < 0.5 or ratio > 2.0  # >2x or <0.5x is suspicious
    print(f"Baseline: {baseline_positive_rate:.2%}  Current: {current_rate:.2%}  Ratio: {ratio:.2f}")
    if alert:
        print("⚠️  ALERT: Prediction distribution has shifted significantly")
    return current_rate, alert
```

### Monitoring Cadence

| Signal | Check frequency | Action threshold |
|---|---|---|
| Prediction distribution | Every batch / hourly | >2x shift → investigate |
| Feature drift (KS test) | Daily | p < 0.05 on key features → investigate |
| Model performance (F1, RMSE) | As labels arrive | >10% degradation → retrain |
| API latency / error rate | Continuously | p99 latency >500ms → scale |

---

## Phase 5: Retraining Strategy

Models stale. The question isn't whether to retrain, but when and how.

### Retraining Triggers

**Schedule-based** — Simplest. Retrain weekly/monthly regardless of performance. Works well when data distribution is stable but the model needs recent data.

**Performance-based** — Retrain when a metric drops below a threshold. Requires labels to be available quickly enough to detect degradation.

**Drift-based** — Retrain when input distribution drifts significantly. Can act before performance degrades.

```python
# Automated retraining decision
def should_retrain(
    current_f1: float,
    baseline_f1: float,
    drift_score: float,
    days_since_last_train: int,
    config: dict
) -> tuple[bool, str]:
    if current_f1 < config['min_f1']:
        return True, f"F1 {current_f1:.3f} below threshold {config['min_f1']}"
    if drift_score > config['max_drift']:
        return True, f"Drift score {drift_score:.3f} exceeds threshold {config['max_drift']}"
    if days_since_last_train > config['max_days']:
        return True, f"Model is {days_since_last_train} days old"
    return False, "No retraining needed"
```

### Safe Retraining with Shadow Deployment

Never replace a production model with a new one without validating first. Use shadow deployment:

1. **Shadow mode**: New model receives the same traffic but predictions are not served to users — just logged.
2. **Compare**: Compare new model's predictions and metrics to production model's on the same inputs.
3. **Canary**: Route 5–10% of real traffic to the new model. Monitor closely for a set period.
4. **Full rollout**: Promote new model to 100% of traffic. Archive old version.

```python
# Canary routing example (using feature flags or load balancer weights)
import random

def get_model(customer_id: str, canary_fraction: float = 0.1):
    """Route a fraction of requests to the challenger model."""
    use_challenger = (hash(customer_id) % 100) < (canary_fraction * 100)
    if use_challenger:
        return challenger_model, "challenger_v4"
    return production_model, "production_v3"
```

---

## Phase 6: CI/CD for ML

Automate the path from model code to production to avoid manual error-prone deployments.

### Minimal ML CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/model_deploy.yml
name: Train and Deploy Model

on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2am
  workflow_dispatch:       # Allow manual trigger

jobs:
  train-and-validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Train model
        run: python train.py --output model_artifacts/
      - name: Validate model
        run: python validate.py --min-f1 0.75
      - name: Build and push Docker image
        run: |
          docker build -t $IMAGE_TAG .
          docker push $IMAGE_TAG
      - name: Deploy to staging
        run: kubectl set image deployment/churn-api app=$IMAGE_TAG
```

The validation step is the safety gate. If the new model doesn't meet the minimum performance threshold, the deploy never happens.

---

## Deployment Decision Framework

Use this to choose your serving strategy for a new project:

```
Is prediction needed synchronously, while a user is waiting?
├─ YES → REST API (FastAPI + Docker)
│        Consider: latency budget, input validation, auth
└─ NO → Batch job
         How fresh do scores need to be?
         ├─ Hours → Scheduled job (cron / Airflow)
         └─ Minutes → Near-real-time (streaming, Kafka consumer)
```

```
How much operational overhead can you absorb?
├─ Low (solo DS, no MLOps team) → FastAPI + Docker + basic monitoring
├─ Medium → Add MLflow, basic CI/CD, drift detection
└─ High → Full MLOps platform (Kubeflow, Vertex AI, SageMaker)
```

---

## Output Checklist

Every deployment should include:
1. A serialized pipeline artifact (from `ds-ml-pipeline`)
2. Experiment tracked in MLflow (or equivalent)
3. A serving layer — API or batch script
4. A `/health` check endpoint (for APIs)
5. Input validation and error handling
6. Prediction logging
7. At least one monitoring check (prediction distribution or drift)
8. A documented rollback procedure

---

## Common Mistakes

**Leaking data in production** — Fitting preprocessing on production data instead of re-using the training-time pipeline. Use `pipeline.predict(raw_df)`, never re-fit transformers.

**Not logging predictions** — Without a prediction log, you can't audit model behavior, diagnose failures, or create retraining datasets.

**No rollback plan** — Before deploying, know exactly how to get back to the previous version. MLflow model registry makes this one command.

**Deploying without a baseline** — Your new model should be compared to the current production model and to a simple heuristic. A model that's worse than "always predict the most common class" should never ship.

**Monitoring only accuracy** — Accuracy can look fine while the model is systematically wrong for a specific subgroup. Monitor sliced metrics if fairness matters for your use case.
