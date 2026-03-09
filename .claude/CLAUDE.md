# Data Science Hub

This repository is a **Data Science Hub** — a skills library and project workspace for end-to-end data science work. It contains 12 specialized DS skills that cover the complete DS lifecycle, plus 44 general-purpose skills for research, writing, planning, and code quality.

## Starting a New DS Project

**Always start with `ds-project-manager`.** It acts as the command center: it assesses your situation, frames the problem, and dispatches the right specialist skills in the right order.

Trigger it by saying things like:
- "I want to start a new data science project"
- "I have a dataset and want to build a model"
- "What should I do next on my project?"
- "I'm stuck mid-project — help me figure out the next step"

## The 12 DS Skills

| Skill | When to Use |
|-------|-------------|
| `ds-project-manager` | **Entry point for all DS work.** New projects, "what next?", stuck mid-project |
| `ds-eda-process` | Explore a new dataset; data profiling; data quality audit before modeling |
| `ds-data-engineering` | ETL/ELT pipelines, Airflow/Prefect/dbt, data warehouses, pipeline failures |
| `ds-feature-engineering` | Transform raw data into model-ready features; encoding; feature selection |
| `ds-supervised-modeling` | Build regression/classification models; algorithm selection; hyperparameter tuning |
| `ds-unsupervised-learning` | Clustering; customer segmentation; PCA/t-SNE/UMAP; anomaly detection |
| `ds-time-series` | Forecasting; ARIMA/Prophet; seasonal patterns; temporal cross-validation |
| `ds-nlp-cv-pipeline` | Text classification; NER; sentiment; image classification; Hugging Face; spaCy |
| `ds-ml-pipeline` | Reproducible sklearn pipelines; ColumnTransformer; preventing data leakage |
| `ds-causal-inference` | A/B tests; did it work?; treatment effects; uplift modeling; incrementality |
| `ds-model-explainability` | SHAP; LIME; fairness audit; GDPR explainability; model cards |
| `ds-mlops-deployment` | Deploy models; model APIs; MLflow; drift detection; retraining; containerization |

## DS Lifecycle Flow

```
ds-project-manager (orchestrates)
    │
    ├─► ds-eda-process          (understand the data)
    ├─► ds-data-engineering     (build/fix data pipelines)
    ├─► ds-feature-engineering  (transform raw → signals)
    │
    ├─► ds-supervised-modeling  │
    ├─► ds-unsupervised-learning ├─ (model the problem)
    ├─► ds-time-series          │
    ├─► ds-nlp-cv-pipeline      │
    │
    ├─► ds-ml-pipeline          (make it reproducible)
    ├─► ds-causal-inference     (measure what worked)
    ├─► ds-model-explainability (explain & audit)
    └─► ds-mlops-deployment     (ship & monitor)
```

## Project Workspace

Actual DS projects live in `projects/`. Each project gets its own subdirectory:

```
projects/
├── _template/           # Copy this to start a new project
│   ├── data/
│   │   ├── raw/         # Source data — never modify these files
│   │   ├── processed/   # Cleaned/transformed data
│   │   └── outputs/     # Model artifacts, predictions, exports
│   ├── notebooks/       # Jupyter notebooks for exploration
│   ├── src/             # Reusable Python modules
│   └── docs/
│       └── plans/       # Implementation plans (YYYY-MM-DD-feature.md)
│
├── my-churn-model/      # Example project
├── sales-forecast/      # Example project
└── ...
```

To start a new project:
```bash
cp -r projects/_template projects/my-project-name
```

## Supporting Skills (Use Alongside DS Work)

- `python-expert` — Python questions, debugging, optimization
- `visualization-expert` — Charts, dashboards, ggplot/matplotlib/plotly
- `deep-research` — Literature review, SOTA methods research
- `academic-researcher` — Reading papers, citations
- `strategy-advisor` — Business framing, stakeholder communication
- `brainstorming` — Explore approaches before committing to one
- `writing-plans` — Write implementation plans to `docs/plans/`
- `executing-plans` — Execute a plan task-by-task
- `technical-writer` — Document models, write reports
- `code-reviewer` — Review DS code (security → performance → correctness)
- `debugger` / `systematic-debugging` — Debug pipelines and model errors
