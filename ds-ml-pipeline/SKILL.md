---
name: ds-ml-pipeline
description: Builds reproducible, production-quality machine learning pipelines using scikit-learn's Pipeline and ColumnTransformer. Use this skill whenever the user needs to chain preprocessing and modeling steps together, prevent data leakage, handle mixed feature types (numeric + categorical), serialize a trained model, build a reusable ML workflow, or automate feature engineering within a pipeline. Trigger when the user says "build a pipeline", "automate my preprocessing", "I keep getting data leakage", "make this reproducible", "deploy this model", "save my model", "chain these steps together", or when you notice the user is fitting transformers on the full dataset before splitting (a leakage red flag). Also use when the user wants to do GridSearchCV across both preprocessing and model parameters simultaneously.
---

# ML Pipeline Skill

This skill helps you build end-to-end machine learning pipelines that are reproducible, leak-free, and production-ready. If the user is doing preprocessing and modeling as separate ad-hoc steps, this skill shows them the right way.

## Why Pipelines Matter

The most common and most dangerous mistake in applied ML is **data leakage** — when information from the test set contaminates the training process. This happens silently whenever you:
- Scale features using statistics computed on the full dataset
- Impute missing values with the overall mean before splitting
- Encode categories using frequencies from all rows
- Select features based on correlations computed on all data

Pipelines prevent leakage by bundling preprocessing and modeling into a single object. When you call `pipeline.fit(X_train, y_train)`, every step sees only training data. When you call `pipeline.predict(X_test)`, the same transformations are applied using parameters learned from training only.

## Building a Pipeline Step-by-Step

### Step 1: Identify Feature Types

Real datasets have mixed types. The first step is to separate columns by the preprocessing they need.

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numeric ({len(numeric_features)}): {numeric_features}")
print(f"Categorical ({len(categorical_features)}): {categorical_features}")
```

### Step 2: Build Preprocessing Transformers

Create separate pipelines for each feature type, then combine them with ColumnTransformer.

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Numeric: impute missing → scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical: impute missing → one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'  # drop columns not listed — explicit is better than silent
)
```

**Key design decisions to explain to the user:**
- `strategy='median'` for numeric imputation: more robust to outliers than mean
- `handle_unknown='ignore'` for OneHotEncoder: prevents crashes when test data has categories not seen in training
- `sparse_output=False`: returns a dense array, which some downstream models need
- `remainder='drop'`: makes it explicit that unlisted columns are excluded. Use `'passthrough'` to keep them unchanged.

### Step 3: Create the Full Pipeline

Append the model as the final step.

```python
from sklearn.ensemble import RandomForestClassifier

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
```

Now the entire workflow — imputation, scaling, encoding, and modeling — is a single object.

### Step 4: Train and Evaluate

```python
from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Cross-validate on training data only
cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring='f1_weighted')
print(f"CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Final fit and evaluation
full_pipeline.fit(X_train, y_train)
test_score = full_pipeline.score(X_test, y_test)
print(f"Test Score: {test_score:.3f}")
```

### Step 5: Hyperparameter Tuning Through the Pipeline

The double-underscore notation lets you tune parameters at any level of the pipeline.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    # Preprocessing parameters
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    # Model parameters
    'classifier__n_estimators': [100, 200, 500],
    'classifier__max_depth': [5, 10, 20, None],
}

grid_search = GridSearchCV(
    full_pipeline, param_grid,
    cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

The naming convention is `stepname__nested_stepname__parameter`. For example, `preprocessor__num__imputer__strategy` reaches into the preprocessor → numeric transformer → imputer → strategy parameter.

### Step 6: Custom Transformers

When built-in transformers aren't enough, create custom ones. This is common for domain-specific feature engineering.

```python
from sklearn.base import BaseEstimator, TransformerMixin

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract year, month, day_of_week from datetime columns."""

    def __init__(self, date_columns):
        self.date_columns = date_columns

    def fit(self, X, y=None):
        return self  # nothing to learn

    def transform(self, X):
        X = X.copy()
        for col in self.date_columns:
            dt = pd.to_datetime(X[col])
            X[f'{col}_year'] = dt.dt.year
            X[f'{col}_month'] = dt.dt.month
            X[f'{col}_dayofweek'] = dt.dt.dayofweek
            X = X.drop(columns=[col])
        return X
```

Custom transformers must implement `fit()` and `transform()` (or `fit_transform()`). Inheriting from `BaseEstimator` and `TransformerMixin` gives you `get_params()`, `set_params()`, and `fit_transform()` for free — which means they'll work seamlessly with GridSearchCV.

### Step 7: Serialize the Pipeline

Save the entire trained pipeline as a single artifact. This is the object you deploy.

```python
import joblib

# Save
joblib.dump(full_pipeline, 'model_pipeline.joblib')

# Load (in production or another notebook)
loaded_pipeline = joblib.load('model_pipeline.joblib')
predictions = loaded_pipeline.predict(new_data)
```

Because the pipeline includes all preprocessing, the loaded model can accept raw data in the same format as the original training data. No separate preprocessing script needed.

## Common Pipeline Patterns

**Regression pipeline with feature selection:**
```python
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge

Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(f_regression, k=20)),
    ('regressor', Ridge(alpha=1.0))
])
```

**Pipeline with dimensionality reduction:**
```python
from sklearn.decomposition import PCA

Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95)),  # keep 95% of variance
    ('classifier', LogisticRegression())
])
```

**Swapping models easily:**
```python
from sklearn.ensemble import GradientBoostingClassifier

# Just replace the last step
full_pipeline.set_params(classifier=GradientBoostingClassifier())
```

## Red Flags to Watch For

If the user is doing any of these, gently redirect them to a pipeline approach:
- Calling `.fit_transform()` on the full dataset before train/test split
- Using separate scripts for preprocessing and modeling
- Manually tracking which transformations to apply at prediction time
- Getting mysteriously high validation scores that don't hold on new data (leakage symptom)

## Output Checklist

Every pipeline you build should include:
1. Clear identification of feature types
2. Appropriate transformer for each type
3. A ColumnTransformer combining them
4. The full pipeline with model
5. Cross-validation results
6. Hyperparameter tuning (at least basic)
7. Serialization code for deployment
