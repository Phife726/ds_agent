---
name: ds-feature-engineering
description: |
  Expert-level feature engineering: transforms raw data into predictive signals. Use whenever the user needs to create better features, encode categoricals strategically, extract signals from dates or IDs, build interaction/ratio features, apply target encoding correctly, handle high-cardinality columns, or select important features. Trigger on: "engineer features", "create new features", "feature selection", "target encoding", "how should I encode this column", "ordinal vs one-hot", "interaction features", "my model isn't improving despite tuning", "what features should I create", "high-cardinality categoricals", "aggregation features", "lagged features", or anything about transforming raw columns into better signals. Also trigger when the user finishes EDA and is preparing to model — feature engineering is the bridge between ds-eda-process and ds-supervised-modeling.
---

# Feature Engineering Skill

Feature engineering is the craft of transforming raw data into the specific signals a model needs to learn well. It's where domain expertise meets statistics — and it has more leverage on model performance than algorithm choice or hyperparameter tuning in most real-world problems.

## Why This is a Separate Skill

EDA (`ds-eda-process`) tells you what's in your data. ML pipelines (`ds-ml-pipeline`) implement transformations reproducibly. Feature engineering is the creative layer in between — it requires you to think deeply about what signals are actually predictive, why, and how to extract them. A mediocre algorithm with great features usually beats a great algorithm with mediocre features.

---

## The Feature Engineering Mindset

Before writing code, answer these questions:

1. **What does the model actually need to know to make this prediction?** Think from the model's perspective — what information, if it knew it, would most change the prediction?
2. **What does each raw column *represent*, not just *contain*?** A numeric column isn't just a number — it might be a count, a rate, an absolute amount, an ID, or a rank. Each requires different treatment.
3. **What domain knowledge can be encoded?** The best features often encode something a domain expert would immediately recognize as meaningful: "days since last purchase," "ratio of actual to expected," "deviation from personal baseline."
4. **What interactions might matter?** Individual features are sometimes less predictive than combinations. "Purchase amount" and "customer age" might each be weak; "purchase amount per year of customer age" might be strong.

---

## Encoding Categorical Variables

### Decision Framework: Which Encoding?

```
How many unique values (cardinality)?
│
├─ Low cardinality (≤15 categories)
│   Is there an inherent order?
│   ├─ YES → Ordinal encoding
│   └─ NO  → One-hot encoding
│
├─ Medium cardinality (15–50)
│   High target correlation?
│   ├─ YES, large dataset → Target encoding (with CV to prevent leakage)
│   └─ NO, or small data → Frequency encoding or binary encoding
│
└─ High cardinality (50+ categories, e.g., ZIP codes, user IDs)
    └─ Target encoding, frequency encoding, or embedding (for neural nets)
        NEVER one-hot encode — creates too many sparse columns
```

### One-Hot Encoding (Low Cardinality, Nominal)

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# For sklearn pipelines
encoder = OneHotEncoder(
    handle_unknown='ignore',  # Critical: prevents crash on unseen categories at predict time
    sparse_output=False,
    drop='first'              # Drop first category to avoid multicollinearity (for linear models)
)

# For pandas (exploratory use, not for pipelines)
df_encoded = pd.get_dummies(df, columns=['contract_type', 'payment_method'], drop_first=True)
```

**When to use**: Nominal categories with no inherent order, low to medium cardinality.
**When NOT to use**: More than ~15-20 categories (explodes feature space); tree models often work better with integer codes.

### Ordinal Encoding (Ordered Categories)

```python
from sklearn.preprocessing import OrdinalEncoder

# The order you specify here matters — encode your domain knowledge!
size_order = [['Small', 'Medium', 'Large', 'XLarge']]
risk_order = [['Low', 'Medium', 'High', 'Very High']]

encoder = OrdinalEncoder(
    categories=size_order,
    handle_unknown='use_encoded_value',
    unknown_value=-1
)
```

**The most common mistake**: Applying OrdinalEncoder to nominal data. "Red", "Blue", "Green" have no inherent order — ordinal-encoding them tells the model they're on a scale, which is wrong.

### Target Encoding (High Cardinality)

Target encoding replaces each category with the mean of the target variable for that category. It's powerful but will cause **severe data leakage** if done naively — because you're using the target to create the feature.

```python
from sklearn.model_selection import KFold
import numpy as np

def target_encode_cv(df: pd.DataFrame, col: str, target: str, n_folds: int = 5) -> pd.Series:
    """
    Cross-validated target encoding to prevent leakage.
    Each row is encoded using statistics computed WITHOUT that row's fold.
    """
    encoded = df[col].copy().astype(float)
    global_mean = df[target].mean()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df):
        train_fold = df.iloc[train_idx]
        category_means = train_fold.groupby(col)[target].mean()
        encoded.iloc[val_idx] = df.iloc[val_idx][col].map(category_means).fillna(global_mean)

    return encoded

# For new (test) data: use category means computed on ALL training data
def target_encode_apply(train_df, test_df, col, target):
    """Apply training-time encoding to test data."""
    global_mean = train_df[target].mean()
    category_means = train_df.groupby(col)[target].mean()
    return test_df[col].map(category_means).fillna(global_mean)
```

**Smoothing for rare categories** — Categories with few observations have noisy target means. Add smoothing to pull them toward the global mean:

```python
def smoothed_target_encode(df, col, target, min_samples=10, smoothing=10):
    """Additive smoothing for target encoding."""
    global_mean = df[target].mean()
    stats = df.groupby(col)[target].agg(['mean', 'count'])
    # Weighted average between category mean and global mean
    stats['encoded'] = (
        (stats['mean'] * stats['count'] + global_mean * smoothing) /
        (stats['count'] + smoothing)
    )
    return df[col].map(stats['encoded']).fillna(global_mean)
```

---

## Numeric Feature Transformations

### When and Why to Transform

```python
import numpy as np
import pandas as pd
from scipy import stats

def suggest_transform(series: pd.Series) -> str:
    """Diagnose a numeric series and suggest the right transformation."""
    skewness = series.skew()
    has_negatives = (series < 0).any()
    has_zeros = (series == 0).any()

    if abs(skewness) < 0.5:
        return "No transformation needed (roughly symmetric)"
    elif skewness > 1 and not has_negatives and not has_zeros:
        return "Log transform: np.log1p(x)  [right-skewed, all positive]"
    elif skewness > 1 and has_zeros:
        return "Log1p transform: np.log1p(x)  [right-skewed, has zeros]"
    elif abs(skewness) > 0.5:
        return "Square root: np.sqrt(x) or Box-Cox transform"
    return "Consider Yeo-Johnson (handles negatives)"

# Applying transforms
df['revenue_log'] = np.log1p(df['revenue'])  # log(1 + x), handles zeros
df['revenue_sqrt'] = np.sqrt(df['revenue'].clip(0))  # clip negatives first

# Box-Cox (requires strictly positive values)
df['revenue_boxcox'], lambda_val = stats.boxcox(df['revenue'].clip(1))
```

**Why transformations matter**: Linear models and distance-based models (KNN, SVM) assume numeric features are on similar scales with roughly symmetric distributions. A revenue column with a long right tail will dominate distance calculations. Log transformation compresses the tail and makes the feature more informative. Tree models don't need transformations — but they still sometimes benefit from them.

### Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler: zero mean, unit variance
# Use for: linear models, SVM, neural networks, PCA, KNN
# Don't use: when you need original units in outputs, or for tree models

# MinMaxScaler: maps to [0, 1]
# Use for: neural networks, when you need bounded output
# Don't use: when there are outliers (they compress everything else)

# RobustScaler: uses median and IQR, robust to outliers
# Use for: when outliers are real (not errors) and linear model
```

**For tree models (Random Forest, XGBoost, LightGBM)**: You do NOT need to scale. Trees split on thresholds, not distances. Scaling adds complexity without improving performance.

### Binning / Discretization

Sometimes converting a continuous feature into categories is more predictive, because the relationship with the target isn't linear.

```python
# Equal-width bins
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 50, 65, 100],
                          labels=['<18', '18-24', '25-34', '35-49', '50-64', '65+'])

# Quantile bins (equal frequency) — avoids empty bins for skewed data
df['revenue_quartile'] = pd.qcut(df['revenue'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Custom domain-driven bins
df['tenure_segment'] = pd.cut(df['tenure_days'],
    bins=[0, 30, 90, 365, np.inf],
    labels=['new', 'developing', 'established', 'loyal'])
```

---

## Date and Time Features

Datetime columns are almost always worth expanding into multiple features. The raw timestamp is rarely as informative as its components.

```python
def extract_datetime_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Extract a rich set of features from a datetime column."""
    dt = pd.to_datetime(df[col])
    prefix = col.replace('_date', '').replace('_at', '').replace('_time', '')

    features = {
        # Calendar features — capture seasonality and calendar effects
        f'{prefix}_year': dt.dt.year,
        f'{prefix}_month': dt.dt.month,
        f'{prefix}_quarter': dt.dt.quarter,
        f'{prefix}_day_of_week': dt.dt.dayofweek,  # 0=Monday
        f'{prefix}_day_of_year': dt.dt.dayofyear,
        f'{prefix}_week_of_year': dt.dt.isocalendar().week,

        # Binary flags — often more useful than raw numbers
        f'{prefix}_is_weekend': (dt.dt.dayofweek >= 5).astype(int),
        f'{prefix}_is_month_end': dt.dt.is_month_end.astype(int),
        f'{prefix}_is_month_start': dt.dt.is_month_start.astype(int),
        f'{prefix}_is_quarter_end': dt.dt.is_quarter_end.astype(int),

        # Time of day (if time component exists)
        f'{prefix}_hour': dt.dt.hour,
        f'{prefix}_is_business_hours': ((dt.dt.hour >= 9) & (dt.dt.hour < 17)).astype(int),
    }

    # Cyclical encoding for periodic features (avoids discontinuity at boundaries)
    # The 23→0 jump in hours is not actually a big jump — sin/cos encoding fixes this
    features[f'{prefix}_month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
    features[f'{prefix}_month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
    features[f'{prefix}_dow_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    features[f'{prefix}_dow_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    return pd.DataFrame(features, index=df.index)
```

**Recency features** — Often more predictive than the raw date:

```python
reference_date = pd.Timestamp.now()

df['days_since_signup'] = (reference_date - pd.to_datetime(df['signup_date'])).dt.days
df['days_since_last_purchase'] = (reference_date - pd.to_datetime(df['last_purchase_date'])).dt.days
df['days_since_last_login'] = (reference_date - pd.to_datetime(df['last_login_date'])).dt.days
```

---

## Ratio and Interaction Features

Domain-meaningful ratios and interactions often capture non-linear relationships that individual features miss.

```python
# Ratio features — normalize by a denominator to get rates
df['revenue_per_session'] = df['total_revenue'] / (df['num_sessions'] + 1)  # +1 avoids division by zero
df['avg_order_value'] = df['total_revenue'] / (df['num_orders'] + 1)
df['support_tickets_per_order'] = df['support_tickets'] / (df['num_orders'] + 1)

# Delta features — change relative to baseline
df['revenue_vs_last_month'] = df['revenue_this_month'] - df['revenue_last_month']
df['revenue_growth_rate'] = df['revenue_vs_last_month'] / (df['revenue_last_month'] + 1)

# Deviation from group mean — captures relative standing
df['revenue_vs_cohort'] = (
    df['revenue'] - df.groupby('acquisition_channel')['revenue'].transform('mean')
)

# Interaction terms (for linear models that can't learn them automatically)
df['tenure_x_engagement'] = df['tenure_days'] * df['weekly_sessions']
df['premium_x_age'] = (df['is_premium'] == 1).astype(int) * df['account_age_days']
```

---

## Aggregation Features (Customer/Entity-Level)

When you have transactional data and need to model at the customer level, aggregation features are essential. They summarize behavioral history into single-row features.

```python
def build_customer_features(transactions_df: pd.DataFrame, customer_id_col: str = 'customer_id') -> pd.DataFrame:
    """
    Build a customer-level feature table from transaction history.
    This is the RFM (Recency, Frequency, Monetary) framework plus extensions.
    """
    now = pd.Timestamp.now()

    customer_features = transactions_df.groupby(customer_id_col).agg(
        # Frequency features
        n_transactions=('transaction_id', 'count'),
        n_unique_products=('product_id', 'nunique'),
        n_unique_channels=('channel', 'nunique'),

        # Monetary features
        total_revenue=('amount', 'sum'),
        avg_order_value=('amount', 'mean'),
        max_order_value=('amount', 'max'),
        revenue_std=('amount', 'std'),

        # Recency features
        last_transaction_date=('transaction_date', 'max'),
        first_transaction_date=('transaction_date', 'min'),
    ).reset_index()

    # Derived features
    customer_features['days_since_last_purchase'] = (
        now - pd.to_datetime(customer_features['last_transaction_date'])
    ).dt.days

    customer_features['customer_age_days'] = (
        now - pd.to_datetime(customer_features['first_transaction_date'])
    ).dt.days

    customer_features['purchase_frequency'] = (
        customer_features['n_transactions'] /
        customer_features['customer_age_days'].clip(1) * 30  # per 30 days
    )

    return customer_features
```

---

## Feature Selection

More features is not always better. Irrelevant features add noise, increase overfitting risk, and slow training.

### Selection by Importance

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def select_features_by_importance(X_train, y_train, threshold=0.01):
    """Use a quick Random Forest to identify useful features."""
    selector = RandomForestClassifier(n_estimators=100, random_state=42)
    selector.fit(X_train, y_train)
    importances = pd.Series(selector.feature_importances_, index=X_train.columns)
    selected = importances[importances >= threshold].index.tolist()
    print(f"Selected {len(selected)}/{len(X_train.columns)} features")
    print(importances.nlargest(20))
    return selected
```

### Variance Inflation Factor (Multicollinearity Detection)

Linear models suffer when features are highly correlated. Detect this and remove redundant features:

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def check_vif(df: pd.DataFrame, numeric_features: list, threshold: float = 10.0) -> pd.DataFrame:
    """
    VIF > 10 = severe multicollinearity. VIF > 5 = worth investigating.
    """
    X = df[numeric_features].dropna()
    vif_data = pd.DataFrame({
        'feature': numeric_features,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(len(numeric_features))]
    }).sort_values('VIF', ascending=False)
    high_vif = vif_data[vif_data['VIF'] > threshold]
    if len(high_vif) > 0:
        print(f"⚠️  {len(high_vif)} features with VIF > {threshold}:")
        print(high_vif.to_string())
    return vif_data
```

### Recursive Feature Elimination

```python
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

selector = RFECV(
    estimator=LogisticRegression(max_iter=1000),
    step=1,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)
selector.fit(X_train_scaled, y_train)
selected_features = X_train.columns[selector.support_].tolist()
print(f"Optimal features: {selector.n_features_}")
```

---

## Feature Engineering Anti-Patterns

**Target leakage** — A feature that encodes or correlates with the target variable in a way that won't exist at prediction time. Example: including `"is_canceled"` as a feature when predicting `"churn"`. Always ask: "would this value be available when I make a real prediction?"

**Encoding before splitting** — Fitting target encoders, scalers, or imputers on the full dataset before train/test split leaks future information. Always fit transformers on training data only.

**Over-engineering sparse features** — Creating 50 interaction terms when you have 500 rows. The model can't learn from this; you'll just overfit.

**Dropping categoricals because they're "strings"** — High-cardinality categoricals like user IDs contain real signal when aggregated or target-encoded. Don't drop them reflexively.

---

## Output: Feature Engineering Plan

For every feature engineering session, produce a plan in this format before writing code:

```markdown
## Feature Engineering Plan

### Raw columns and their nature
- tenure_days: numeric, right-skewed → log transform
- contract_type: nominal, 3 categories → one-hot encode
- last_login_date: datetime → recency + calendar features
- product_category: nominal, 120 categories → target encode with CV smoothing

### New derived features to create
- days_since_last_login (recency)
- avg_monthly_revenue (ratio: total_revenue / tenure_months)
- is_high_value_long_tenure (interaction: revenue_quartile * tenure_segment)

### Feature selection plan
- Drop ID columns: customer_id, account_number
- Check VIF after encoding; remove features VIF > 10
- Evaluate importances after quick Random Forest pass

### Leakage checks
- Confirm no future data encoded in features
- Confirm all encoders fitted on train split only
```
