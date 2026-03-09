---
name: ds-supervised-modeling
description: Helps build, train, evaluate, and compare supervised machine learning models. Use this skill whenever the user wants to predict a target variable, classify data, run regression, compare algorithms, tune hyperparameters, evaluate model performance, check for bias or fairness issues, interpret model predictions, or select the best model for a task. Trigger when the user says things like "build a model", "predict X", "classify these", "which algorithm should I use", "my model isn't performing well", "tune my model", "is my model fair", "explain these predictions", or drops a dataset and asks for a prediction task. Also use when the user needs help choosing between metrics like accuracy, F1, AUC-ROC, or RMSE.
---

# Supervised Modeling Skill

This skill guides you through the complete supervised learning workflow: algorithm selection, training, evaluation, interpretation, fairness auditing, and hyperparameter tuning. It emphasizes understanding *why* you'd choose one approach over another, not just how to call the API.

## Decision Framework: Choosing an Algorithm

Before writing any code, understand the problem. The right algorithm depends on the answers to these questions:

**What type is the target?**
- Continuous → Regression (Linear, Ridge, Lasso, Random Forest, Gradient Boosting)
- Binary categorical → Binary Classification
- Multi-class categorical → Multi-class Classification

**How much data do you have?**
- < 1,000 rows → Prefer simpler models (logistic regression, small decision tree). Complex models will overfit.
- 1,000–100,000 → Most algorithms are fair game. This is the sweet spot for tree ensembles.
- 100,000+ → Consider training time. Linear models and LightGBM scale well. Deep learning becomes viable.

**Do you need interpretability?**
- High (regulated, stakeholder-facing) → Linear/Logistic Regression, Decision Tree, or use SHAP on any model
- Medium (internal analytics) → Random Forest with feature importances
- Low (pure prediction accuracy) → Gradient Boosting (XGBoost/LightGBM), Neural Networks

### Algorithm Quick Reference

**Linear Models** — Start here. They're fast, interpretable, and establish a baseline.
- Linear Regression: continuous target, assumes roughly linear relationships
- Logistic Regression: classification, outputs calibrated probabilities
- Ridge/Lasso: linear models with regularization to prevent overfitting. Lasso also performs feature selection.

**Tree-Based Models** — Handle nonlinearity and interactions naturally.
- Decision Tree: fully interpretable but overfits aggressively. Use for understanding, not production.
- Random Forest: bags many trees, reduces variance. Robust default choice.
- Gradient Boosting (XGBoost, LightGBM): sequentially corrects errors. Often the best performer on tabular data.

**Support Vector Machines** — Effective in high-dimensional spaces. Use with kernel tricks for nonlinear boundaries. Slow on large datasets.

**K-Nearest Neighbors** — Simple, no training phase. Degrades badly in high dimensions (curse of dimensionality). Good for small datasets and baselines.

**Neural Networks** — Flexible but data-hungry and hard to interpret. For tabular data, tree ensembles usually win. Reserve neural nets for images, text, or very large datasets.

## The Modeling Workflow

### Step 1: Establish a Baseline

Always start with the simplest reasonable model. For classification, this might be logistic regression or even a majority-class classifier. For regression, try linear regression or predicting the mean. The baseline tells you whether your fancier models are actually adding value.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify for classification
)

baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train, y_train)
y_pred = baseline.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Step 2: Train and Compare Multiple Models

Don't just pick one algorithm. Train several and compare. Use cross-validation for honest estimates.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
    results[name] = {'mean': scores.mean(), 'std': scores.std()}
    print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")
```

### Step 3: Choose the Right Metric

This is where many projects go wrong. The metric must match the business objective.

**Classification metrics:**
- **Accuracy**: Only use when classes are balanced. Misleading otherwise.
- **Precision**: When false positives are costly (spam filter, fraud alerts to customers)
- **Recall**: When false negatives are costly (disease screening, fraud detection)
- **F1 Score**: Harmonic mean of precision and recall. Good default when classes are imbalanced.
- **AUC-ROC**: Measures ranking ability across all thresholds. Use when you need to compare models independent of threshold choice.
- **AUC-PR**: Better than AUC-ROC when positive class is rare (<5%)

**Regression metrics:**
- **RMSE**: Penalizes large errors heavily. Use when big misses are unacceptable.
- **MAE**: Treats all errors equally. More robust to outliers.
- **R²**: Proportion of variance explained. Intuitive but can be misleading with nonlinear relationships.
- **MAPE**: Percentage error. Useful for business communication but undefined when actuals are zero.

### Step 4: Evaluate Thoroughly

Go beyond a single number. Produce these evaluation artifacts:

**For classification:**
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Confusion matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# ROC curve (binary)
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

**For regression:**
- Residual plot (residuals vs predicted): should show no pattern
- Actual vs predicted scatter: should hug the diagonal
- Residual distribution: should be approximately normal

### Step 5: Interpret the Model

Predictions without explanations aren't useful in most business contexts.

**Feature importance** (tree-based models):
```python
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(15).plot(kind='barh')
plt.title('Top 15 Feature Importances')
plt.show()
```

**SHAP values** (any model — preferred for nuanced interpretation):
```python
import shap
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
```

SHAP shows not just which features matter globally, but how each feature pushes individual predictions higher or lower. Always prefer SHAP over built-in feature importances when interpretability matters.

### Step 6: Tune Hyperparameters

Only tune after you've selected a promising model. Tuning a bad algorithm is wasted effort.

**For small search spaces** — use GridSearchCV:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_leaf': [1, 2, 5]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.3f}")
```

**For large search spaces** — use RandomizedSearchCV or Optuna for Bayesian optimization.

### Step 7: Fairness Audit

Before deploying any model, check for bias across sensitive groups. This isn't optional — it's professional practice.

**Evaluate performance per group:**
```python
for group in df['sensitive_column'].unique():
    mask = X_test['sensitive_column'] == group
    group_score = metric(y_test[mask], y_pred[mask])
    print(f"  {group}: {group_score:.3f}")
```

**Key fairness metrics:**
- **Demographic parity**: Equal positive prediction rates across groups
- **Equalized odds**: Equal true positive and false positive rates across groups
- **Predictive parity**: Equal precision across groups

If you find disparities, don't just flag them — discuss the tradeoffs. Perfect fairness across all definitions simultaneously is often mathematically impossible. The right choice depends on the application context.

## Output Format

Structure model comparison results as a clear summary:
1. Which models were tested and their cross-validated scores
2. The chosen model and why
3. Key evaluation visualizations
4. Feature importance or SHAP interpretation
5. Any fairness concerns
6. Recommended next steps (more data, feature engineering, deployment considerations)
