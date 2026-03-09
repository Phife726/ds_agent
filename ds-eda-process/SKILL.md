---
name: ds-eda-process
description: Guides structured exploratory data analysis using the CRISP-DM framework. Use this skill whenever the user wants to explore a dataset, perform EDA, understand data distributions, clean or prepare data, investigate missing values, detect outliers, visualize relationships between variables, or get a high-level summary of a dataset before modeling. Also trigger when the user asks "what does this data look like?", "help me understand this dataset", "summarize my data", "clean this data", "prepare data for modeling", or anything involving initial data investigation. Even if the user just drops a CSV and says "take a look", this skill applies.
---

# EDA Process Skill

This skill helps you conduct thorough, structured exploratory data analysis. It follows the CRISP-DM framework's first three phases (Business Understanding → Data Understanding → Data Preparation) and produces actionable insights with clean, well-commented Python code.

## Why Structure Matters

Jumping straight into modeling without understanding your data is the most common mistake in data science. EDA reveals data quality issues, surprising distributions, and hidden relationships that directly affect which models will work and which will fail. This skill enforces a disciplined process so nothing gets missed.

## The EDA Workflow

Follow these phases in order. Each phase builds on the previous one.

### Phase 1: Business Understanding (Frame the Problem)

Before touching the data, clarify the goal. Ask the user (or infer from context):
- What question are we trying to answer?
- What would a useful outcome look like?
- Are there known constraints (timeline, tools, audience)?

If the user hasn't stated a goal, ask. A dataset without a question is just a spreadsheet.

### Phase 2: Data Understanding (First Look)

Load the data and produce a structured summary. Always generate these outputs:

**Shape and types overview:**
```python
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumn types:\n{df.dtypes.value_counts()}")
print(f"\nFirst 5 rows:")
df.head()
```

**Missing value audit:**
```python
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_report = pd.DataFrame({
    'missing_count': missing,
    'missing_pct': missing_pct
}).query('missing_count > 0').sort_values('missing_pct', ascending=False)
print(missing_report)
```

**Descriptive statistics:** Use `df.describe(include='all')` and highlight anything surprising — extreme ranges, suspicious zeros, constant columns, or unexpected category counts.

**Uniqueness check:** Identify potential ID columns, near-constant features, and high-cardinality categoricals.

### Phase 3: Visual Exploration

Choose visualizations based on what the data actually contains. Don't produce every possible chart — pick the ones that answer questions.

**For numeric features:**
- Histograms with KDE overlays to understand distributions
- Box plots to spot outliers
- Correlation heatmap for relationships between numeric columns

**For categorical features:**
- Value counts with bar charts (especially for imbalanced classes)
- Cross-tabulations with the target variable if one exists

**For relationships:**
- Scatter plots for continuous-vs-continuous (color by target if classification)
- Violin or box plots for continuous-vs-categorical
- Pair plots if the dataset has ≤8 numeric features

Always use this visualization setup for consistency:
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
```

### Phase 4: Data Quality Assessment

After visual exploration, produce a structured data quality report covering:

1. **Missing data strategy** — For each column with missing values, recommend: drop rows, drop column, impute with median/mode/model, or flag as a feature. Explain the reasoning (e.g., "Column X is 85% missing — drop it" vs "Column Y is 3% missing and appears MCAR — median imputation is safe").

2. **Outlier detection** — Use IQR method or z-scores. Don't automatically remove outliers — flag them and discuss whether they're errors or legitimate extreme values.

3. **Type corrections** — Identify columns stored as the wrong type (dates as strings, categories as integers, numeric codes that should be categorical).

4. **Duplicate detection** — Check for exact and near-duplicates. Report count and recommend action.

5. **Feature engineering opportunities** — Note any obvious derived features (e.g., extracting year/month from dates, creating ratios from existing columns, binning continuous variables).

### Phase 5: Data Preparation Output

Produce a clean, documented data preparation script that:
- Handles missing values according to the strategy from Phase 4
- Corrects data types
- Removes or flags duplicates
- Creates recommended engineered features
- Separates features and target (if applicable)
- Notes any assumptions made

The script should be modular — each transformation step in its own clearly commented block so the user can modify individual decisions.

## Output Format

Always structure the EDA as a narrative, not just code dumps. For each finding, include:
1. What you observed (the fact)
2. Why it matters (the implication)
3. What to do about it (the recommendation)

## Common Pitfalls to Flag

- **Leakage columns**: Features that encode the target variable or future information. Flag any column with suspiciously perfect correlation to the target.
- **Simpson's paradox**: Trends that reverse when you control for a confounding variable. If you see a surprising correlation, suggest checking subgroups.
- **Survivorship bias**: The data you have may not represent the population you care about. Note any obvious selection effects.
- **Scale differences**: Features on wildly different scales will dominate distance-based models. Flag this and recommend scaling.
