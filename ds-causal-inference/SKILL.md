---
name: ds-causal-inference
description: |
  Answers the question "did it actually work?" — turning correlations and model predictions into defensible causal claims and business decisions. Use this skill whenever the user wants to measure the true effect of an intervention, run or analyze an A/B test, design an experiment, attribute outcomes to causes, or understand whether a business action actually caused an improvement (rather than just correlating with it). Trigger when the user says "A/B test", "experiment", "did this campaign work?", "causal", "treatment effect", "control group", "uplift modeling", "incrementality", "attribution", "propensity score", "did it work?", "difference in differences", "regression discontinuity", "counterfactual", "lift", "ROI of our intervention", or anything where the goal is to establish causation rather than prediction. Also trigger when the user has a supervised model but needs to know whether acting on its predictions will actually produce the outcomes they care about.
---

# Causal Inference Skill

This skill answers the hardest question in applied data science: **did this thing actually cause that outcome, or did we just get lucky?** Prediction models tell you what will happen. Causal inference tells you what will happen *if you act*.

## Why This is a Distinct Problem

A churn model might predict that customers with low usage are likely to churn. But if you offer a discount to low-usage customers, will that reduce churn — or were those customers going to stay anyway, and you just gave away margin for free?

Correlation says: low usage → churn. Causal inference asks: does *reducing* low usage (or targeting those customers) actually *prevent* churn? These are different questions, and confusing them is one of the most expensive mistakes in applied DS.

---

## The Causal Hierarchy

```
Prediction      "Customers with feature X have 80% churn probability"
(What will happen)

Association     "High-discount customers churned less"
(What happened together)

Causation       "The discount *caused* a 12% reduction in churn"
(What would happen if we intervened)

Counterfactual  "Had we not sent the email, this customer would have churned"
(What would have happened otherwise)
```

Every step up this ladder requires stronger assumptions and more careful methodology. This skill helps you determine which level is achievable with your data and design.

---

## Method Selection: Which Tool for Which Question?

| You have... | You want... | Use |
|---|---|---|
| A randomized experiment (A/B test) | Measure effect of one change | T-test / OLS on experiment data |
| Observational data, similar groups | Control for confounders | Propensity Score Matching / Weighting |
| Policy with a sharp cutoff | Effect near the threshold | Regression Discontinuity Design |
| Pre/post data with a control group | Remove time trends | Difference-in-Differences |
| An instrument (exogenous variation) | Isolate causal effect | Instrumental Variables |
| Individual-level response data | Optimize who to target | Uplift Modeling |

---

## Method 1: A/B Testing (Randomized Controlled Trial)

The gold standard. When you can randomize, do it. Every other method is an approximation of this.

### Pre-Test: Power Analysis

Run a power analysis *before* collecting data. Under-powered experiments waste time and money; over-powered experiments are inefficient.

```python
from scipy import stats
import numpy as np

def minimum_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """
    baseline_rate: current conversion/churn/click rate
    minimum_detectable_effect: smallest effect you care about (e.g., 0.02 = 2pp lift)
    """
    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    effect_size = abs(p1 - p2) / np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    analysis = stats.tt_ind_solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative='two-sided'
    )
    n = int(np.ceil(analysis))
    print(f"Baseline rate: {p1:.2%}")
    print(f"Target rate (with effect): {p2:.2%}")
    print(f"Required n per group: {n:,}")
    print(f"Total experiment size: {2*n:,}")
    return n

# Example: email campaign
min_n = minimum_sample_size(
    baseline_rate=0.05,      # 5% baseline click rate
    minimum_detectable_effect=0.01  # want to detect a 1pp lift
)
```

**Rule of thumb**: Don't start an experiment if you can't run it long enough to reach the required sample size. Under-powered experiments are worse than not running them — they generate false confidence.

### Running the Analysis

```python
from scipy import stats
import pandas as pd
import numpy as np

def analyze_ab_test(
    df: pd.DataFrame,
    group_col: str = 'group',          # 'control' or 'treatment'
    outcome_col: str = 'converted',    # binary or continuous
    alpha: float = 0.05
) -> dict:
    control = df[df[group_col] == 'control'][outcome_col]
    treatment = df[df[group_col] == 'treatment'][outcome_col]

    # Basic stats
    results = {
        'control_mean': control.mean(),
        'treatment_mean': treatment.mean(),
        'absolute_lift': treatment.mean() - control.mean(),
        'relative_lift': (treatment.mean() - control.mean()) / control.mean(),
        'n_control': len(control),
        'n_treatment': len(treatment),
    }

    # Statistical test
    if df[outcome_col].nunique() == 2:  # Binary outcome
        contingency = pd.crosstab(df[group_col], df[outcome_col])
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        results['test'] = 'chi-squared'
        results['p_value'] = p_value
    else:  # Continuous outcome
        t_stat, p_value = stats.ttest_ind(control, treatment, equal_var=False)
        results['test'] = "Welch's t-test"
        results['p_value'] = p_value

    results['significant'] = p_value < alpha

    # Confidence interval on the lift
    se = np.sqrt(
        (control.std()**2 / len(control)) +
        (treatment.std()**2 / len(treatment))
    )
    ci_low = results['absolute_lift'] - 1.96 * se
    ci_high = results['absolute_lift'] + 1.96 * se
    results['ci_95'] = (ci_low, ci_high)

    return results

def print_ab_results(results: dict):
    print("=" * 50)
    print("A/B TEST RESULTS")
    print("=" * 50)
    print(f"Control:   {results['control_mean']:.4f} (n={results['n_control']:,})")
    print(f"Treatment: {results['treatment_mean']:.4f} (n={results['n_treatment']:,})")
    print(f"Absolute lift:  {results['absolute_lift']:+.4f}")
    print(f"Relative lift:  {results['relative_lift']:+.2%}")
    print(f"95% CI: [{results['ci_95'][0]:+.4f}, {results['ci_95'][1]:+.4f}]")
    print(f"p-value: {results['p_value']:.4f}  ({'✓ Significant' if results['significant'] else '✗ Not significant'})")
```

### A/B Test Pitfalls to Warn About

**Peeking / early stopping** — Looking at results before the experiment is done and stopping when you see significance inflates your false positive rate. Fix: commit to a sample size upfront and don't analyze until you have it.

**Multiple comparisons** — Testing 10 variants with α=0.05 means you expect ~0.5 false positives by chance. Fix: use Bonferroni correction (`α/n_tests`) or a family-wise error rate correction.

**SUTVA violations (Spillover)** — If treatment of one unit affects controls (e.g., email recipients tell friends about the discount), your effect estimate is biased. This is common in social products and marketplaces.

**Novelty effect** — Treatments often show a spike because users are reacting to *newness*, not to the actual change. Run experiments long enough to let this wear off.

---

## Method 2: Propensity Score Methods

When you can't randomize (observational data), people who received a treatment are different from those who didn't. Propensity scores balance the groups.

**The core idea**: Model the probability that each unit received treatment given their covariates. Use this to create comparable groups.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def estimate_propensity_scores(df: pd.DataFrame, treatment_col: str, feature_cols: list) -> pd.Series:
    """
    Estimate the probability of receiving treatment.
    Using logistic regression — simple and interpretable.
    """
    X = df[feature_cols]
    T = df[treatment_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X_scaled, T)

    propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]
    return pd.Series(propensity_scores, index=df.index, name='propensity_score')


def inverse_probability_weighting(df: pd.DataFrame, treatment_col: str, propensity_col: str) -> pd.Series:
    """
    IPW weights: treated units are weighted by 1/p, control by 1/(1-p).
    Creates a pseudo-population where treatment is independent of covariates.
    """
    weights = np.where(
        df[treatment_col] == 1,
        1 / df[propensity_col],
        1 / (1 - df[propensity_col])
    )
    # Stabilize: multiply by marginal probability of treatment
    p_treatment = df[treatment_col].mean()
    stabilized_weights = np.where(
        df[treatment_col] == 1,
        p_treatment / df[propensity_col],
        (1 - p_treatment) / (1 - df[propensity_col])
    )
    return pd.Series(stabilized_weights, index=df.index, name='ipw_weight')


def compute_ate_ipw(df: pd.DataFrame, treatment_col: str, outcome_col: str, weight_col: str) -> float:
    """Average Treatment Effect via IPW estimator."""
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    ate = (
        (treated[outcome_col] * treated[weight_col]).sum() / treated[weight_col].sum() -
        (control[outcome_col] * control[weight_col]).sum() / control[weight_col].sum()
    )
    return ate
```

**Check balance after weighting** — Compare covariate distributions between treated and control after applying weights. A good propensity model makes treated and control look similar. Use Standardized Mean Difference (SMD) < 0.1 as the target for each covariate.

---

## Method 3: Difference-in-Differences (DiD)

For when you have pre/post measurements and a natural control group. DiD removes time trends and group-level confounders by using the control group as a counterfactual.

**The assumption (parallel trends)**: Without the treatment, the treated group would have followed the same trend as the control group.

```python
import statsmodels.formula.api as smf

def run_did(df: pd.DataFrame, outcome_col: str = 'outcome') -> None:
    """
    DiD regression: outcome ~ treated + post + treated:post + controls
    The coefficient on treated:post is the DiD estimate.

    Assumes df has columns:
    - treated: 1 if unit received treatment, 0 if control
    - post: 1 if observation is post-treatment, 0 if pre
    - [outcome_col]: the outcome variable
    """
    formula = f"{outcome_col} ~ treated * post"
    model = smf.ols(formula, data=df).fit(cov_type='HC3')  # HC3 = robust SEs
    print(model.summary())

    did_estimate = model.params['treated:post']
    ci = model.conf_int().loc['treated:post']
    print(f"\nDiD Estimate (ATT): {did_estimate:.4f}")
    print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"Interpretation: Treatment caused a {did_estimate:+.4f} change in {outcome_col}")
```

**Validating the parallel trends assumption** — This cannot be tested statistically (it's counterfactual), but you can provide evidence for it:
- Plot pre-treatment trends for treated and control groups — they should look similar
- Run a "placebo test": apply the DiD to a pre-treatment period where no effect should exist. A significant result would undermine the assumption.

---

## Method 4: Uplift Modeling (Heterogeneous Treatment Effects)

Uplift modeling doesn't just ask "does the treatment work on average?" It asks "for *which individuals* does the treatment work?" This is the bridge between causal inference and operational decision-making.

**The four quadrants of customer response:**

```
                    Outcome without treatment
                    Bad              Good
Outcome with  ┌─────────────────────────────┐
treatment     │                             │
Good          │  PERSUADABLES    SURE BETS  │  ← Target persuadables,
              │  (Treat these)   (Leave be) │    not sure bets
              ├─────────────────────────────┤
Bad           │  LOST CAUSES     SLEEPING   │
              │  (Never treat)   DOGS       │
              │                (Never treat)│
              └─────────────────────────────┘
```

Most marketing models predict "who will convert" — but some of those people would have converted anyway. Uplift models predict "who will convert *because* of the treatment."

```python
# Approach 1: T-Learner (train separate models for treated and control)
from sklearn.ensemble import GradientBoostingRegressor

def train_t_learner(df_treated, df_control, feature_cols, outcome_col):
    """
    Fit separate outcome models for treated and control.
    Individual treatment effect = mu_1(x) - mu_0(x)
    """
    model_treated = GradientBoostingRegressor(random_state=42)
    model_treated.fit(df_treated[feature_cols], df_treated[outcome_col])

    model_control = GradientBoostingRegressor(random_state=42)
    model_control.fit(df_control[feature_cols], df_control[outcome_col])

    return model_treated, model_control

def predict_uplift_t_learner(X, model_treated, model_control):
    """Estimated individual treatment effect."""
    return model_treated.predict(X) - model_control.predict(X)

# Approach 2: Use the CausalML library (recommended for production)
# pip install causalml
# from causalml.inference.tree import UpliftRandomForestClassifier
```

---

## Communicating Results

Causal results need to be communicated differently from predictive results.

**Wrong:** "The model has an 81% AUC."
**Right:** "The campaign caused a 2.3 percentage point increase in 30-day retention (95% CI: +1.1pp to +3.5pp, p=0.001), equivalent to approximately 4,200 retained customers per month at the current user base."

Always include:
1. **The effect size** in business-relevant units (not just p-value)
2. **A confidence interval** — a point estimate without uncertainty is misleading
3. **The assumptions** required for the causal interpretation to be valid
4. **The limitations** — what could invalidate the estimate
5. **The business implication** — what action should change based on this result

---

## Red Flags: When Not to Claim Causation

Stop the user from making causal claims when the design doesn't support it:

- **No control group**: "Sales went up 20% after the campaign" — there's no counterfactual. Maybe sales were going up anyway.
- **Self-selection**: Users who opt into a feature are different from those who don't. Comparing them is not an experiment.
- **Reverse causality**: "Heavy users receive more emails" — but are they heavy users *because* of the emails, or do they get more emails *because* they're heavy users?
- **Confounders**: "Cities with more bike lanes have lower obesity rates." Temperature, urban density, and income all confound this.

When a proper causal design isn't possible, be explicit: "We observe an association between X and Y. We cannot establish causation from this data because [reason]. To establish causation, we would need [experiment/method]."
