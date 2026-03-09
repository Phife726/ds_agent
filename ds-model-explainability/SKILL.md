---
name: ds-model-explainability
description: >
  Deep expertise in ML model interpretability and explainability — SHAP, LIME, PDP/ICE plots,
  counterfactuals, bias auditing, and model cards. Use whenever the user needs to explain model
  predictions (globally or per-instance), audit for fairness, document a model for deployment or
  compliance, or satisfy regulatory requirements. Trigger on: "explain my model", "why did the model
  predict X", "SHAP", "LIME", "partial dependence", "model card", "bias audit", "fairness",
  "disparate impact", "Fairlearn", "AI Fairness 360", "GDPR explainability", "credit model compliance",
  "ICE plot", "counterfactual", "regulated model", "document my model's limitations", or when a
  compliance/legal team asks to understand a model. Companion to ds-supervised-modeling; use it
  whenever interpretability or fairness goes beyond a quick feature importance plot.
---

# Model Interpretability & Explainability

Explainability isn't just a nice-to-have — it's the difference between a model you can trust and one
you can't. This skill covers the full toolkit: understanding what a model learned globally, explaining
individual predictions locally, auditing for discriminatory patterns, and documenting model behavior
for stakeholders and regulators.

## Framework: Picking the Right Technique

The question "how do I explain my model?" doesn't have one answer. The right technique depends on
three things: **scope** (global vs. local), **model type** (glass-box vs. black-box), and
**audience** (technical, business, or regulatory).

```
SCOPE          GLASS-BOX MODELS          BLACK-BOX MODELS
─────────────────────────────────────────────────────────
Global         Coefficients, feature     SHAP summary plots,
(whole model)  importances (built-in)    PDP/ICE plots, permutation
                                         importance

Local          Read the tree path        SHAP waterfall/force plots,
(one prediction) or coefficients         LIME, counterfactuals
```

**Quick selection guide:**
- Need a single number per feature for a slide deck → permutation importance or mean |SHAP|
- Need to explain one specific prediction to a customer or regulator → SHAP waterfall or LIME
- Need to show how a feature affects predictions across its range → PDP + ICE plots
- Need to answer "what would change this prediction?" → counterfactual explanations
- Need to audit for discriminatory outcomes → Fairlearn or AIF360
- Need to document everything for deployment → Model Card

---

## SHAP Values

SHAP (SHapley Additive exPlanations) is the gold standard for post-hoc explainability. It has a
solid mathematical foundation (game theory), works with any model, and decomposes predictions
into per-feature contributions that always sum to the actual prediction. Use it by default unless
the model or dataset is too large to make it practical.

### Choosing the Right Explainer

```python
import shap

# Tree-based models (XGBoost, LightGBM, Random Forest, sklearn trees)
# → Use TreeExplainer — fast, exact
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)  # Returns a shap.Explanation object

# Linear models (LogisticRegression, LinearRegression, Ridge, Lasso)
# → Use LinearExplainer — fast, exact
masker = shap.maskers.Independent(X_train)
explainer = shap.LinearExplainer(model, masker)
shap_values = explainer(X_test)

# Any other model (SVM, neural net, pipeline wrapping a black box)
# → Use KernelExplainer — slow but universal. Sample background data.
background = shap.sample(X_train, 100)  # keep background small
explainer = shap.KernelExplainer(model.predict_proba, background)
shap_values = explainer(X_test[:50])  # limit test size too
```

### Global Interpretation

```python
# Summary plot — shows feature importance + effect direction in one view.
# This is the single most informative SHAP visualization.
shap.summary_plot(shap_values, X_test)

# Bar chart version — cleaner for presentations (mean absolute SHAP)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Dependence plot — shows how feature X interacts with feature Z
shap.dependence_plot("loan_amount", shap_values.values, X_test,
                     interaction_index="credit_score")
```

### Local Interpretation (single prediction)

```python
# Waterfall plot — shows how each feature pushed the prediction
# above or below the baseline. Best for explaining one decision.
idx = 42  # the prediction you want to explain
shap.waterfall_plot(shap_values[idx])

# Force plot — more compact, good for embedding in reports
shap.force_plot(explainer.expected_value, shap_values[idx].values,
                X_test.iloc[idx])

# When to prefer which: waterfall for written reports, force plot
# for dashboards and interactive tools.
```

### SHAP for Multi-class Classification

```python
# shap_values will be a list of arrays, one per class
explainer = shap.TreeExplainer(multiclass_model)
shap_values = explainer(X_test)

# To plot for a specific class (e.g., class 2):
shap.summary_plot(shap_values[:, :, 2], X_test)
```

**Common pitfalls:**
- KernelExplainer on large datasets will be very slow. Always sample.
- For classification, check whether to use `predict_proba` vs `predict` — SHAP values mean different things for each.
- SHAP values can be correlated with each other when features are correlated. Don't interpret them as independent causes.

---

## LIME

LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions by fitting
a simple surrogate model (linear regression) in a neighborhood around the instance being explained.
It's more intuitive for non-technical audiences than SHAP and handles text/images naturally.

Use LIME when: SHAP is too slow (large KernelExplainer cases), you're working with text or image
models, or your audience finds the SHAP math explanation confusing.

```python
from lime import lime_tabular

# Train the explainer once on training data
explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['No Default', 'Default'],
    mode='classification'
)

# Explain a single instance
instance = X_test.iloc[42]
explanation = explainer.explain_instance(
    data_row=instance.values,
    predict_fn=model.predict_proba,
    num_features=10  # top 10 features in the local explanation
)

# Show in notebook
explanation.show_in_notebook(show_table=True)

# Or save as HTML
explanation.save_to_file('lime_explanation.html')
```

**SHAP vs. LIME in practice:** SHAP has stronger theoretical guarantees (consistency, local
accuracy) and scales to global explanations. LIME's local approximation can be unstable across
runs on the same instance. Default to SHAP; use LIME for text/image models or when explaining
to non-technical stakeholders who find the "contribution" framing clearer.

---

## Partial Dependence Plots (PDP) and ICE Plots

PDPs show the marginal effect of one or two features on predicted outcomes, averaged across
all other features. ICE (Individual Conditional Expectation) plots show the same thing but for
each individual row, revealing heterogeneous effects that averages hide.

Always show ICE alongside PDP — if ICE lines cross, the average (PDP) is misleading.

```python
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Single feature PDP + ICE
fig, ax = plt.subplots(figsize=(8, 5))
PartialDependenceDisplay.from_estimator(
    model, X_train,
    features=['loan_amount'],      # feature to plot
    kind='both',                   # 'average' = PDP only, 'individual' = ICE, 'both' = both
    subsample=200,                 # limit ICE lines for readability
    random_state=42,
    ax=ax
)
ax.set_title('Partial Dependence + ICE: Loan Amount')
plt.tight_layout()
plt.show()

# Two-feature interaction PDP (heatmap)
PartialDependenceDisplay.from_estimator(
    model, X_train,
    features=[('loan_amount', 'credit_score')],  # tuple = 2D interaction
    kind='average'
)
plt.suptitle('2D PDP: Loan Amount × Credit Score')
plt.show()
```

**Centered ICE plots:** Subtract each line's starting value so they all begin at 0. This makes
the shape (steepness, direction) visible even when absolute prediction values vary widely.

```python
PartialDependenceDisplay.from_estimator(
    model, X_train,
    features=['loan_amount'],
    kind='both',
    centered=True,   # centers ICE lines at the left edge
    subsample=200
)
```

**Interpretation guide:** A flat PDP line = feature has little effect (after averaging). Steep
or nonlinear PDPs reveal the actual relationship the model learned. Crossing ICE lines signal
important interactions — dig deeper with a 2D PDP for those feature pairs.

---

## Permutation Importance

When you need feature importance that's model-agnostic and comparable across models, use
permutation importance. It measures the drop in model performance when a feature's values are
randomly shuffled — directly measuring how much the model relies on each feature.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,        # shuffle each feature 10 times and average
    random_state=42,
    scoring='roc_auc'    # use your primary metric here
)

# Convert to a readable dataframe
perm_imp = pd.DataFrame({
    'feature': X_test.columns,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values('importance_mean', ascending=False)

# Features with negative importance are actively hurting the model
perm_imp.plot(x='feature', y='importance_mean', kind='barh',
              xerr='importance_std', figsize=(8, 6))
plt.title('Permutation Importance ± std')
plt.show()
```

---

## Counterfactual Explanations

Counterfactuals answer: "What is the minimum change to this input that would flip the prediction?"
They're especially useful in credit, hiring, and healthcare — contexts where a person has a right
to know what they could do differently.

```python
# Using DiCE (Diverse Counterfactual Explanations)
# pip install dice-ml
import dice_ml
from dice_ml import Dice

# Wrap data and model
d = dice_ml.Data(dataframe=train_df,
                 continuous_features=['loan_amount', 'income', 'credit_score'],
                 outcome_name='default')
m = dice_ml.Model(model=model, backend='sklearn')
exp = Dice(d, m, method='random')

# Generate counterfactuals for one instance
instance = X_test.iloc[[42]]
cf = exp.generate_counterfactuals(
    instance,
    total_CFs=3,                    # number of alternatives to generate
    desired_class='opposite',       # flip the prediction
    features_to_vary=['loan_amount', 'credit_score']  # only vary these
)
cf.visualize_as_dataframe()
```

---

## Fairness & Bias Auditing

### Framework: What Does "Fair" Mean?

Before measuring fairness, define what fairness means in your context. Three common definitions
are in tension — you usually can't satisfy all three at once:

- **Demographic parity**: Equal positive prediction rates across groups (P(Ŷ=1|A=a) = P(Ŷ=1|A=b))
- **Equalized odds**: Equal TPR *and* FPR across groups (harder to achieve, stronger guarantee)
- **Predictive parity** (calibration): Equal precision across groups — "when predicted positive, equally likely to be positive"

Which to prioritize depends on the stakes. In criminal justice, equalized odds matters (equal
false positive rates mean equal wrongful detention). In medical screening, recall parity matters
(equal false negative rates mean equal disease caught). Be explicit about the choice.

### Fairlearn

Fairlearn is the easiest library for standard fairness metrics and constrained model training.

```python
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Create a MetricFrame to see metrics broken out by sensitive attribute
sensitive = X_test['race']   # or gender, age_group, etc.

mf = MetricFrame(
    metrics={
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'selection_rate': selection_rate
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive
)

print(mf.overall)
print(mf.by_group)

# Summary metrics
print(f"Demographic parity difference: "
      f"{demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive):.3f}")
# Interpretation: 0 = perfect demographic parity; ±0.1 is often the legal threshold
```

**Fairlearn mitigation (post-processing):**
```python
from fairlearn.postprocessing import ThresholdOptimizer

optimizer = ThresholdOptimizer(
    estimator=model,
    constraints='equalized_odds',   # or 'demographic_parity'
    objective='balanced_accuracy_score'
)
optimizer.fit(X_train, y_train, sensitive_features=X_train['race'])
y_pred_fair = optimizer.predict(X_test, sensitive_features=X_test['race'])
```

### AI Fairness 360 (AIF360)

AIF360 (IBM) has a wider range of fairness metrics and covers the full pipeline: pre-processing
the data, in-processing the model, or post-processing predictions.

```python
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

# Wrap data in AIF360 format
privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]

dataset_orig = BinaryLabelDataset(
    df=df,
    label_names=['loan_approved'],
    protected_attribute_names=['race']
)

# Measure bias in the dataset itself (before any model)
metric = BinaryLabelDatasetMetric(
    dataset_orig,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)
print(f"Disparate impact: {metric.disparate_impact():.3f}")
# 1.0 = perfect parity; <0.8 is the standard legal "four-fifths rule" threshold

# Reweighing — preprocess to reduce bias
from aif360.algorithms.preprocessing import Reweighing
rw = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_reweighed = rw.fit_transform(dataset_orig)
```

**Practical advice:** Start with Fairlearn's `MetricFrame` — it's cleaner and does 90% of what
you need. Reach for AIF360 when you need the four-fifths rule, disparate impact ratio, or
pre/in-processing bias mitigation methods that Fairlearn doesn't cover.

---

## Model Cards

A model card is a short document that communicates everything a deployment decision-maker needs
to know: what the model does, how it was trained, how it performs across subgroups, and where
it should and shouldn't be used. Google popularized the format; it's now the standard for
responsible AI deployment.

### Model Card Template

```markdown
# Model Card: [Model Name]

## Model Details
- **Developed by:** [Team/organization]
- **Date:** [Month, Year]
- **Model type:** [e.g., Gradient Boosted Trees — XGBoost 1.7]
- **License:** [Internal / Apache 2.0 / etc.]
- **Contact:** [Email or Slack channel]

## Intended Use
- **Primary intended uses:** [What the model is designed to do]
- **Primary intended users:** [Who will use it]
- **Out-of-scope uses:** [Explicitly state what it should NOT be used for]

## Training Data
- **Source:** [Dataset name, version, date range]
- **Size:** [N rows, M features]
- **Preprocessing:** [Key steps: imputation, encoding, scaling]
- **Known gaps:** [Underrepresented groups, time periods, geographies]

## Evaluation Data
- **Source:** [Held-out test set / separate dataset]
- **Date range:** [If relevant — model may degrade on out-of-distribution data]

## Performance
| Metric          | Overall | Group A | Group B |
|-----------------|---------|---------|---------|
| Accuracy        | 0.84    | 0.87    | 0.79    |
| Precision       | 0.81    | 0.83    | 0.77    |
| Recall          | 0.79    | 0.82    | 0.71    |

## Fairness Analysis
- **Sensitive attributes evaluated:** [race, gender, age group, zip code]
- **Fairness metric used:** [demographic parity / equalized odds / calibration]
- **Findings:** [e.g., 8-point recall gap between demographic groups X and Y]
- **Mitigation applied:** [threshold adjustment / reweighing / none]

## Ethical Considerations
- [Known risks or harms if misused]
- [Feedback loop risks — does the model's output affect future training data?]
- [Consent and data provenance issues, if any]

## Limitations & Recommendations
- [Where performance degrades — distribution shift, edge cases]
- [What monitoring should be set up post-deployment]
- [Recommended review cadence]
```

### Generating Performance Tables Programmatically

```python
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def performance_by_group(y_true, y_pred, groups, group_col):
    """Generate a performance breakdown table for a model card."""
    rows = []
    for g in groups[group_col].unique():
        mask = groups[group_col] == g
        rows.append({
            'Group': g,
            'N': mask.sum(),
            'Accuracy': accuracy_score(y_true[mask], y_pred[mask]),
            'Precision': precision_score(y_true[mask], y_pred[mask], zero_division=0),
            'Recall': recall_score(y_true[mask], y_pred[mask], zero_division=0),
            'F1': f1_score(y_true[mask], y_pred[mask], zero_division=0)
        })
    return pd.DataFrame(rows).round(3)
```

---

## Regulated Industry Considerations

### Credit / Lending (ECOA, FCRA)
The Equal Credit Opportunity Act and Fair Credit Reporting Act require lenders to provide
"adverse action notices" — specific reasons why credit was denied. SHAP waterfall plots map
directly to this requirement: the top negative-SHAP features become the adverse action reasons.
Use `explanation.top_n_features(4, negative_only=True)` to generate them programmatically.

The "four-fifths rule" (disparate impact ratio ≥ 0.8) is the EEOC/CFPB standard threshold.
Always compute `aif360.metrics.disparate_impact()` before deploying any credit model.

### Healthcare (HIPAA, FDA SaMD guidance)
Models used in clinical decision support must document intended use clearly — the FDA's SaMD
(Software as a Medical Device) guidance requires this. Model cards serve this purpose. Also
document sensitivity/specificity tradeoffs explicitly, and always flag the population the
model was validated on (e.g., "validated on adults 18–65; not validated on pediatric patients").

### EU AI Act / GDPR (Article 22)
GDPR Article 22 gives individuals the right not to be subject to purely automated decisions
with significant effects, and a right to "meaningful information about the logic involved."
SHAP explanations + counterfactuals together satisfy this. For EU deployments, document this
in the model card under "Regulatory Compliance."

The EU AI Act (2024) classifies credit scoring, employment screening, and critical
infrastructure as "high-risk AI." High-risk systems require conformity assessments, human
oversight mechanisms, and detailed technical documentation — model cards are the foundation.

---

## Output Format

When completing an explainability analysis, deliver:

1. **Technique selection rationale** — which methods you used and why, given the model type and use case
2. **Global explanation** — SHAP summary plot or permutation importance, with written interpretation
3. **Local explanation** — at least one example prediction explained with SHAP waterfall or LIME
4. **Fairness report** — MetricFrame output by group, key disparity metrics, any mitigation applied
5. **Model card** — filled-in template, ready for review
6. **Limitations and recommendations** — what the explanation doesn't capture, monitoring suggestions

If writing code: produce runnable Python scripts with comments explaining each section. If the
dataset isn't available in the session, write template code with clear `# TODO: replace with your data` markers.
