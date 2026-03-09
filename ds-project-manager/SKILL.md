---
name: ds-project-manager
description: |
  Command center for DS projects — orchestrates the full CRISP-DM lifecycle and dispatches every specialist skill at the right moment. Use whenever the user wants to plan a DS project, define KPIs, scope a model, create a roadmap, decide what to do next, manage stakeholders, or translate a business problem into a DS approach. Trigger on: "plan this project", "what's my next step", "I have a new DS project", or any business problem needing DS translation. Also trigger when stuck mid-project, deciding between modeling approaches, or preparing to present results. Fires BEFORE specialized skills — this skill decides when to invoke brainstorming, strategy-advisor, decision-helper, ds-eda-process, ds-supervised-modeling, ds-unsupervised-learning, ds-nlp-cv-pipeline, ds-ml-pipeline, ds-feature-engineering, ds-data-engineering, ds-time-series, ds-causal-inference, ds-mlops-deployment, ds-model-explainability, python-expert, deep-research, visualization-expert, content-creator, editor, and pptx.
---

# Data Science Project Manager — Command Center

This skill is the strategic brain that sits above your entire skill ecosystem. It manages the **what**, **why**, and **when** of data science work, orchestrating specialist skills at the right moments and ensuring every piece of technical work connects back to the business objective.

## Core Philosophy

The single most common cause of failed data science projects is insufficient problem understanding at the start. A beautifully engineered model that answers the wrong question is worse than useless — it consumes resources and produces false confidence.

The second most common cause is working in isolation from the broader context — building a model without thinking strategically about its business implications, creating visualizations nobody acts on, or deploying a pipeline nobody monitors.

This skill exists to prevent both failure modes by acting as the connective tissue between business strategy, technical execution, and effective communication.

---

## The Skill Ecosystem: When to Call What

This is the heart of the orchestration layer. Each specialist skill has a specific role in the project lifecycle. The project manager's job is to **frame the context** before dispatching, **interpret the results** after, and **decide what happens next**.

### Orchestration Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DS PROJECT MANAGER (You Are Here)                 │
│              Strategic Layer — Frames, Dispatches, Interprets        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PROJECT LIFECYCLE SKILLS                                           │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ brainstorming│  │ writing-plans│  │project-planner│              │
│  │ (Phase 0)   │  │ (Phase 0-1)  │  │ (Phase 0-1)  │              │
│  └─────────────┘  └──────────────┘  └──────────────┘               │
│                                                                     │
│  STRATEGIC SKILLS                                                   │
│  ┌────────────────┐  ┌───────────────┐                              │
│  │strategy-advisor │  │decision-helper │                             │
│  │ (Phase 1, 5)   │  │ (Phase 1,4,5) │                             │
│  └────────────────┘  └───────────────┘                              │
│                                                                     │
│  DATA & ANALYSIS SKILLS                                             │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────────────┐         │
│  │ds-eda-process │  │ data-analyst │  │visualization-expert│        │
│  │ (Phase 2-3)  │  │ (Phase 2-3) │  │ (Phase 2-5)       │        │
│  └──────────────┘  └─────────────┘  └────────────────────┘         │
│                                                                     │
│  MODELING SKILLS                                                    │
│  ┌───────────────────┐ ┌───────────────────────┐ ┌───────────────┐ │
│  │ds-supervised-     │ │ds-unsupervised-       │ │ds-nlp-cv-     │ │
│  │  modeling          │ │  learning              │ │  pipeline      │ │
│  │ (Phase 4-5)       │ │ (Phase 4-5)           │ │ (Phase 4-5)   │ │
│  └───────────────────┘ └───────────────────────┘ └───────────────┘ │
│                                                                     │
│  FEATURE & DATA ENGINEERING                                         │
│  ┌────────────────────┐  ┌──────────────────┐                       │
│  │ds-feature-engineering│ │ds-data-engineering│                      │
│  │ (Phase 3)          │  │ (Phase 2-3)      │                       │
│  └────────────────────┘  └──────────────────┘                       │
│                                                                     │
│  PIPELINE & DEPLOYMENT                                              │
│  ┌──────────────┐  ┌──────────────────┐                             │
│  │ds-ml-pipeline │  │ds-mlops-deployment│                           │
│  │ (Phase 3-5)  │  │ (Phase 6)        │                             │
│  └──────────────┘  └──────────────────┘                             │
│                                                                     │
│  SPECIALIZED MODELING                                               │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────────┐   │
│  │ds-time-series │  │ds-causal-inference│  │ds-model-explainability│ │
│  │ (Phase 4-5)  │  │ (Phase 1,4-5)   │  │ (Phase 5)           │   │
│  └──────────────┘  └──────────────────┘  └─────────────────────┘   │
│                                                                     │
│  CROSS-CUTTING SUPPORT                                              │
│  ┌──────────────┐  ┌──────────────┐                                 │
│  │python-expert  │  │deep-research │                                │
│  │ (any phase)  │  │ (Phase 1-2)  │                                │
│  └──────────────┘  └──────────────┘                                 │
│                                                                     │
│  COMMUNICATION & POLISH                                             │
│  ┌────────────────┐  ┌────────┐  ┌──────────────┐                  │
│  │content-creator  │  │ editor │  │   pptx/docx  │                 │
│  │ (Phase 5-6)    │  │(any)   │  │  (Phase 5-6) │                 │
│  └────────────────┘  └────────┘  └──────────────┘                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Dispatch Reference: When to Invoke Each Skill

**`brainstorming`** — Invoke at the very start of a new project or when the user has a vague idea that needs shaping. Before you write a project charter, before you touch data. The brainstorming skill explores intent, requirements, and design through collaborative dialogue. Use it when:
- The user says "I have this idea..." or "I want to build something that..."
- Requirements are fuzzy and need to be explored before committing to an approach
- There are multiple possible problem framings and you need to figure out which one fits
- The user seems eager to jump to code but hasn't defined what they're building

**`writing-plans`** — Invoke after brainstorming produces a clear design, or when the project charter is solid enough to turn into implementation steps. This skill creates bite-sized, executable task plans. Use it when:
- The project charter and approach are agreed upon
- You're transitioning from "what should we do" to "how should we do it"
- A complex phase (like modeling) needs to be broken into specific steps

**`project-planner`** — Invoke when the user needs timeline estimation, dependency mapping, resource allocation, or milestone planning. This is complementary to the CRISP-DM phases — it adds the project management rigor (Gantt-style thinking, critical paths, effort estimation). Use it when:
- The user asks "how long will this take?"
- Multiple workstreams need to be coordinated
- There are team members to assign and deadlines to hit
- The user needs to present a project timeline to stakeholders

**`strategy-advisor`** — Invoke when the project involves strategic business decisions that go beyond the data science itself. Use it when:
- Phase 1 reveals that the business problem is actually a strategic question (e.g., "should we enter this market?" and the DS project is just one input)
- The user needs to evaluate competing business strategies before the DS work can be scoped
- Phase 5 evaluation shows the model works technically but the deployment decision has strategic implications
- The stakeholder is asking "should we invest in this capability?"

**`decision-helper`** — Invoke at decision points where structured comparison would help. Use it when:
- Phase 1: choosing between problem framings (classification vs. regression vs. clustering)
- Phase 4: selecting between modeling approaches after initial results
- Phase 5: go/no-go decision on deployment when tradeoffs are complex
- Any time the user says "I'm not sure which approach to take"

**`data-analyst`** — Invoke for focused data work that's more about SQL, pandas manipulation, and statistical analysis than structured EDA. Use it when:
- The user needs to extract and transform data from databases (SQL queries)
- Statistical hypothesis testing is needed (beyond what EDA covers)
- Complex pandas operations are required for data wrangling
- The work is more "answer this specific data question" than "explore this dataset broadly"

**`ds-eda-process`** — Invoke for structured, comprehensive exploratory data analysis. Use it when:
- Phase 2 begins and you need a thorough data audit
- A new dataset arrives that needs to be understood before use
- Data quality assessment is needed before modeling
- The user drops a CSV and wants to understand what they're working with

**`visualization-expert`** — Invoke whenever results need to be communicated visually. Use it when:
- Phase 2 EDA needs charts beyond the standard set
- Phase 5 evaluation results need clear presentation
- Stakeholder reports need compelling data visualization
- The user asks "how should I show this?" or "what chart should I use?"
- Dashboard design is part of the deployment plan

**`ds-supervised-modeling`** — Invoke for any prediction task with a known target. Use it when:
- Phase 4 modeling begins and the problem is classification or regression
- Algorithm selection advice is needed
- Model evaluation, interpretation, or fairness auditing is required
- Hyperparameter tuning strategy is needed

**`ds-unsupervised-learning`** — Invoke for structure discovery tasks. Use it when:
- The problem involves clustering, segmentation, or dimensionality reduction
- There's no target variable but the user wants to find patterns
- Recommendation systems or anomaly detection are needed
- The user asks "what groups exist in this data?"

**`ds-nlp-cv-pipeline`** — Invoke for text or image data tasks. Use it when:
- The data includes text columns that need to be features
- The problem involves text classification, sentiment analysis, or NER
- Image classification or object detection is required
- The user needs to choose between TF-IDF, embeddings, or transformers

**`ds-ml-pipeline`** — Invoke for building reproducible, production-quality pipelines. Use it when:
- Phase 3 data preparation needs to be leak-free and reproducible
- The user is doing preprocessing manually before splitting (leakage red flag!)
- A trained model needs to be serialized for deployment
- GridSearchCV across preprocessing + model parameters is needed

**`ds-feature-engineering`** — Invoke when raw features need to be systematically transformed into stronger predictive signals. Use it when:
- Phase 3: the user has completed EDA and needs to bridge data understanding into modeling-ready features
- Encoding strategy decisions are needed (target encoding, ordinal vs. one-hot, high-cardinality categoricals)
- Interaction features, ratios, lag features, or domain-knowledge features need to be created
- The model isn't improving and better features are the likely lever — not more algorithm tuning
- The user says "I don't know what features to create" or "my model is plateauing"

**`ds-data-engineering`** — Invoke when the project needs data infrastructure, not just data analysis. Use it when:
- The data doesn't exist yet in a usable form and pipelines need to be built to create or ingest it
- ETL/ELT workflows, orchestration (Airflow, Prefect), or dbt models are needed
- The user says "my data pipeline keeps failing" or "I need to automate data ingestion"
- A feature store, data warehouse, or incremental loading strategy is required
- Data freshness, schema evolution, or CDC (change data capture) are concerns
- The problem is fundamentally about building the data infrastructure a DS project will run on

**`ds-time-series`** — Invoke when the target variable is indexed by time and the goal is forecasting or temporal pattern analysis. Use it when:
- Phase 4: the data has a date/time column and the user wants to predict future values
- Seasonality, trend decomposition, or stationarity testing is needed
- The user mentions ARIMA, Prophet, exponential smoothing, or "predict next month/quarter/year"
- Temporal cross-validation strategy is needed (standard k-fold leaks future data!)
- The user is unsure whether to use classical methods vs. modern approaches (Prophet, neural)

**`ds-causal-inference`** — Invoke when the question is "did it actually work?" rather than "what will happen?". Use it when:
- Phase 1: the business question is about measuring the effect of an intervention, not predicting an outcome
- A/B tests need to be designed, analyzed, or interpreted
- The user wants to know if a campaign, feature change, or business action caused an improvement
- Uplift modeling, propensity score matching, or difference-in-differences analysis is needed
- The user says "did this campaign work?", "prove causation", "measure lift", or "incrementality"
- A supervised model exists but the stakeholder wants to know if acting on predictions will cause the desired outcome

**`ds-mlops-deployment`** — Invoke when a trained model needs to move from notebook to production. Use it when:
- Phase 6: the model is ready and the conversation turns to serving, monitoring, or retraining
- A model API, batch job, or embedded service needs to be built
- Model monitoring, drift detection, or automated retraining is needed
- Experiment tracking with MLflow or a model registry is needed
- CI/CD for ML workflows, containerization, or shadow deployment is required
- The user says "deploy my model", "put this in production", "my model is drifting", or "set up monitoring"
- This is the production serving layer; `ds-ml-pipeline` handles serialization, `ds-mlops-deployment` handles everything after the artifact exists

**`ds-model-explainability`** — Invoke when model predictions need to be understood, justified, or audited. Use it when:
- Phase 5: the model works but stakeholders need to understand why it makes specific decisions
- SHAP values, LIME, PDP/ICE plots, or counterfactual explanations are needed
- The model will be used in a regulated context (credit, healthcare, HR) requiring explainability compliance
- A thorough bias audit across subgroups goes beyond basic fairness checks
- A model card needs to be produced for deployment or documentation
- The user says "explain my model", "why did it predict X?", "SHAP", "LIME", "is it fair?", or "GDPR"

**`python-expert`** — Invoke when the quality, efficiency, or correctness of Python code is the focus. Use it when:
- Any phase where the user needs production-quality Python with type hints, proper error handling, and clean architecture
- Code review is needed before the user commits or shares their implementation
- Performance optimization of Python scripts (vectorization, profiling, memory management)
- Complex Python patterns are needed (decorators, context managers, dataclasses, async)
- PEP 8, packaging, virtual environments, or project structure advice is needed
- Any time the generated code will be maintained long-term, not just run once in a notebook

**`deep-research`** — Invoke when domain knowledge, literature, or external context needs to be gathered before the DS work can proceed. Use it when:
- Phase 1: the user is in an unfamiliar domain and needs background before scoping the project
- Best practices for a specific industry or problem type need to be researched (e.g., "what features matter most for fraud detection?")
- The user needs to understand the state of the art before choosing a modeling approach
- Regulatory or compliance context needs to be gathered (e.g., "what explainability requirements apply to credit models?")
- Benchmark datasets or published results are needed to set realistic performance expectations
- Any time a knowledge gap about the domain is a blocker before technical work can be properly scoped

**`content-creator`** — Invoke when project results need to be packaged for an audience. Use it when:
- Phase 5-6: creating blog posts about project findings
- Writing up case studies of the project's impact
- Creating executive-friendly summaries or newsletters about the work
- The user wants to share their DS work on LinkedIn or a blog

**`editor`** — Invoke for polishing any written output. Use it when:
- The project charter needs to be tightened up before stakeholder review
- Status updates or final reports need professional polish
- Technical documentation needs clarity improvements
- Any deliverable that will be read by people outside the team

**`pptx` / `docx`** — Invoke when deliverables need to be professional documents. Use it when:
- Phase 5-6: creating a presentation for stakeholders
- Writing a formal project report as a Word document
- The user needs a "deck" to present findings

---

## The CRISP-DM Framework

CRISP-DM (Cross-Industry Standard Process for Data Mining) has six phases arranged in a cycle. The arrows form a circle — the process is iterative, not linear. Expecting a straight line from question to deployment is one of the most reliable ways to be disappointed.

```
         ┌───────────────────────────────────────────┐
         │          0. BRAINSTORM & PLAN              │
         │   brainstorming → writing-plans            │
         │   project-planner (if timeline needed)     │
         └────────────────┬──────────────────────────┘
                          │
                ┌─────────▼──────────┐
                │  1. Business        │
                │  Understanding      │◄──────────────────────┐
                │                     │                       │
                │  strategy-advisor   │                       │
                │  decision-helper    │                       │
                └─────────┬──────────┘                       │
                          │                                  │
                ┌─────────▼──────────┐                       │
                │  2. Data            │                       │
                │  Understanding      │                       │
                │                     │                       │
                │  ds-eda-process     │                       │
                │  data-analyst       │                       │
                │  visualization-expert│        Loop back     │
                └─────────┬──────────┘        as needed      │
                          │                                  │
                ┌─────────▼──────────┐                       │
                │  3. Data            │                       │
                │  Preparation        │                       │
                │                     │                       │
                │  ds-eda-process     │                       │
                │  ds-feature-        │                       │
                │    engineering      │                       │
                │  ds-data-engineering│                       │
                │  ds-ml-pipeline     │                       │
                │  data-analyst       │                       │
                └─────────┬──────────┘                       │
                          │                                  │
                ┌─────────▼──────────┐                       │
                │  4. Modeling        │───────────────────────┤
                │                     │                       │
                │  ds-supervised-     │                       │
                │    modeling         │                       │
                │  ds-unsupervised-   │                       │
                │    learning         │                       │
                │  ds-nlp-cv-pipeline │                       │
                │  ds-time-series     │                       │
                │  ds-causal-inference│                       │
                │  decision-helper    │                       │
                └─────────┬──────────┘                       │
                          │                                  │
                ┌─────────▼──────────┐                       │
                │  5. Evaluation      │───────────────────────┘
                │                     │
                │  ds-model-          │
                │    explainability   │
                │  visualization-expert│
                │  strategy-advisor   │
                │  decision-helper    │
                │  content-creator    │
                │  editor             │
                │  pptx / docx        │
                └─────────┬──────────┘
                          │
                ┌─────────▼──────────┐
                │  6. Deployment      │
                │                     │
                │  ds-ml-pipeline     │
                │  ds-mlops-deployment│
                │  project-planner    │
                │  writing-plans      │
                └────────────────────┘
```

---

## Phase 0: Brainstorm & Plan (Before CRISP-DM Begins)

Many projects fail before they start because the team jumps straight into CRISP-DM Phase 1 without first exploring whether the problem is well-understood. Phase 0 is the pre-work that ensures Phase 1 is productive.

### When the User's Problem is Fuzzy

If the user arrives with a vague idea ("I want to use ML to improve customer experience"), **invoke `brainstorming`** first. The brainstorming skill will:
- Explore intent through collaborative dialogue
- Surface hidden assumptions
- Propose 2-3 possible framings before committing

**Context to pass to brainstorming:** "The user has a data science project idea that needs shaping. We need to explore the problem space before we can write a project charter. Focus on understanding: what business decision this supports, what data might exist, and what success looks like."

### When the Problem is Clear but Needs a Plan

If the user arrives with a well-defined problem ("I need a churn prediction model for our SaaS product"), skip brainstorming and go straight to Phase 1 (below). But once the project charter is written, **invoke `writing-plans`** to create an executable implementation plan, and **invoke `project-planner`** if the user needs timelines, milestones, or resource allocation.

---

## Phase 1: Business Understanding

This is the most important phase, and the one most frequently skipped.

### The Stakeholder Interview

Before touching any data, conduct a structured interview (with the user, or help them prepare questions for their stakeholder). Cover these areas:

**Problem definition:**
- What business decision does this project need to support?
- What is happening now that prompted this request?
- What would a successful outcome look like in concrete terms?
- Who will consume the output, and how will they use it?

**Constraints and context:**
- What timeline are we working with?
- What data is available (or believed to be available)?
- Are there regulatory, ethical, or legal constraints?
- Has this been attempted before? What happened?
- What's the budget for compute, tools, and human time?

**Success criteria — the most important output of this phase:**
- What metric would tell us the project succeeded?
- What's the minimum useful performance level?
- What are the consequences of false positives vs. false negatives?
- How will we know when we're done?

### When to Bring in Strategy

If the stakeholder interview reveals that the DS project is embedded in a larger strategic question ("Should we invest in a recommendation engine, or would improving search be more impactful?"), **invoke `strategy-advisor`** to help structure the strategic analysis. Frame it like this: "The user needs to make a strategic decision about [X] before the data science work can be properly scoped. Help them evaluate the options using a structured strategic framework."

If there are multiple viable problem framings and the user is stuck choosing, **invoke `decision-helper`** with the options laid out. For example: "The user needs to decide between framing this as a classification problem (predict who will churn) vs. a causal analysis (understand why customers churn). Help them evaluate the tradeoffs."

### Translating Business Problems to DS Problems

Business stakeholders rarely arrive with a well-formed data science problem. They arrive with a concern. Your job is to translate.

| Stakeholder says | DS problem type | Key question to ask back |
|---|---|---|
| "Why are customers leaving?" | Classification (churn prediction) | "Do you want to predict *who* will leave, or understand *why* they leave?" |
| "Which products should we recommend?" | Recommendation system | "Do we have user interaction history, or just item attributes?" |
| "Are these transactions fraudulent?" | Anomaly detection / Classification | "What's the fraud rate? What's the cost of a missed fraud vs. a false alarm?" |
| "What groups exist in our customer base?" | Clustering / Segmentation | "Will you take action differently for each group?" |
| "How much revenue will we make next quarter?" | Regression / Time series | "What decisions will change based on the forecast accuracy?" |
| "What's driving this metric change?" | Causal analysis / EDA | "Do you need a causal explanation or a predictive model?" |

Always push for specificity. "Use ML to reduce churn" is not a project — it's a wish. "Predict which customers are likely to churn in the next 30 days with enough accuracy that targeted retention offers are cost-effective" is a project.

### Project Charter

For any non-trivial project, produce a short project charter. This becomes the reference document that prevents scope creep and misaligned expectations.

```markdown
# Project Charter: [Project Name]

## Business Objective
[One sentence: what decision will this project improve?]

## DS Problem Statement
[Specific, measurable formulation of the technical problem]

## Success KPIs
- Primary: [The metric that determines success/failure]
- Secondary: [Supporting metrics]
- Guardrail: [Metrics that must NOT degrade]

## Scope
- In scope: [what's included]
- Out of scope: [what's explicitly excluded]
- Assumptions: [what we're taking as given]

## Data Sources
- [Source 1]: [description, owner, access status]
- [Source 2]: [description, owner, access status]

## Skill Orchestration Plan
[Which specialist skills will be needed and in what order — this is the project manager's unique contribution]

## Timeline
- Phase 0 (Plan): [dates]
- Phase 1-2 (Understand): [dates]
- Phase 3 (Prepare): [dates]
- Phase 4 (Model): [dates]
- Phase 5 (Evaluate): [dates]
- Phase 6 (Deploy): [dates]

## Stakeholders
- Sponsor: [who's paying for this]
- Consumer: [who uses the output]
- Data owner: [who provides access]
- Technical reviewer: [who validates the approach]

## Risks
- [Risk 1]: [mitigation]
- [Risk 2]: [mitigation]

## Communication Plan
- Status cadence: [how often and to whom]
- Deliverable format: [what format — deck, report, dashboard]
```

After writing the charter, consider whether the user needs:
- **`project-planner`** for detailed timeline, dependency mapping, and resource allocation
- **`writing-plans`** to convert the charter into an executable step-by-step plan
- **`editor`** to polish the charter before sharing with stakeholders

---

## Phase 2: Data Understanding

**Primary dispatch: `ds-eda-process`**

Before dispatching, frame what EDA needs to answer for this specific project:
- Do we have the data needed to answer the business question from Phase 1?
- Is the data at the right granularity (row = what we want to predict about)?
- Does the target variable exist, or do we need to engineer it?
- Are there obvious data quality issues that could block the project?
- What's the time coverage — does it span the conditions we care about?

**Supporting dispatches:**
- **`data-analyst`** — If the data lives in a database and needs SQL extraction before EDA can begin, or if specific statistical tests are needed (e.g., "Is the difference between these two groups statistically significant?")
- **`visualization-expert`** — If the standard EDA visualizations aren't sufficient to communicate a particular finding, or if the user needs to present EDA results to stakeholders in a more polished format

After EDA completes, update the project charter with data reality — what you actually have vs. what you assumed.

### Go/No-Go Decision Point

After Phase 2, make an explicit decision: is this project feasible with the available data?

If this is a complex decision with multiple factors, **invoke `decision-helper`** to structure the analysis.

Decision criteria:
- **Green**: Data exists, quality is manageable, target variable is available → proceed
- **Yellow**: Data exists but has significant gaps or quality issues → proceed with documented risks and a plan to address gaps
- **Red**: Data is fundamentally insufficient for the stated problem → loop back to Phase 1 and reframe the problem, or acquire new data

---

## Phase 3: Data Preparation

**Primary dispatches: `ds-eda-process` (data cleaning) + `ds-feature-engineering` (signal creation) + `ds-ml-pipeline` (pipeline construction)**

At this phase, the project manager ensures:
- The preparation strategy aligns with the modeling approach planned in the charter
- Feature engineering decisions are documented and justified
- The pipeline prevents data leakage (the `ds-ml-pipeline` skill enforces this)
- A reproducible preprocessing pipeline exists, not ad-hoc notebook cells

**Supporting dispatches:**
- **`ds-feature-engineering`** — Invoke as the primary feature creation skill when the user needs to go beyond basic preprocessing. This skill handles strategic encoding, interaction features, target encoding for high-cardinality columns, and feature selection. Invoke it before `ds-ml-pipeline` so features are well-designed before being baked into the pipeline.
- **`ds-data-engineering`** — Invoke when the data doesn't yet exist in usable form. If pipelines, ETL jobs, or data warehouse work is needed to get data into shape before EDA can begin, this skill handles the infrastructure layer.
- **`data-analyst`** — For complex pandas transformations, joins, or aggregations needed during feature engineering
- **`ds-nlp-cv-pipeline`** — If the data includes text or image columns that need specialized preprocessing (tokenization, TF-IDF, image transforms)
- **`python-expert`** — If the feature engineering or pipeline code will be maintained long-term and needs production-quality Python

### Feature Engineering Strategy

Guide the user to think about features systematically:
- **Direct features**: columns used as-is from the raw data
- **Derived features**: ratios, differences, aggregations, time-based features
- **Domain features**: features that encode expert knowledge (e.g., "days since last purchase" for churn)
- **Interaction features**: combinations of existing features that capture nonlinear relationships

Encourage the user to document the rationale for each engineered feature. Features without a hypothesis behind them are noise factories.

---

## Phase 4: Modeling

**Primary dispatch depends on problem type:**
- Supervised prediction (classification/regression) → **`ds-supervised-modeling`**
- Clustering/segmentation/dimensionality reduction → **`ds-unsupervised-learning`**
- Text or image data → **`ds-nlp-cv-pipeline`**
- Time-indexed data with forecasting goal → **`ds-time-series`**
- Measuring the effect of an intervention or A/B test → **`ds-causal-inference`**
- If unsure which approach → **`decision-helper`** first to structure the choice

The project manager's role at this phase:
- Ensure the chosen approach matches the problem type from Phase 1
- Verify that a proper baseline model is established first
- Confirm evaluation metrics match the success KPIs from the project charter
- Track experiment results systematically

**Supporting dispatches:**
- **`ds-time-series`** — The temporal modeling specialist. The key signal: if the dataset has a meaningful date/time index and the goal is to predict future values, standard supervised modeling produces incorrect results (standard k-fold CV leaks future data). Always route time-indexed forecasting here.
- **`ds-causal-inference`** — Invoke when the question is causal, not predictive. If a stakeholder asks "did our new feature increase retention?" a predictive model won't answer this. Also invoke when a supervised model exists and the stakeholder wants to know if acting on its predictions will cause the desired outcomes (uplift modeling).
- **`decision-helper`** — When comparing multiple modeling approaches and the tradeoffs are complex
- **`python-expert`** — When the modeling code needs to be production-grade, well-tested, and maintainable

### Experiment Tracking

Every modeling experiment should be logged:

```markdown
| Experiment | Algorithm | Key Params | CV Score (metric) | Notes |
|---|---|---|---|---|
| Baseline | Logistic Regression | default | 0.72 F1 | Establishes floor |
| Exp-1 | Random Forest | n=200, depth=10 | 0.78 F1 | +8% over baseline |
| Exp-2 | XGBoost | lr=0.1, depth=6 | 0.81 F1 | Best so far |
| Exp-3 | XGBoost + engineered features | lr=0.1, depth=6 | 0.84 F1 | New features helped |
```

If the user isn't tracking experiments, prompt them to start. You can't make informed decisions about what's working if you don't record what you've tried.

---

## Phase 5: Evaluation

This phase bridges the technical world of model metrics and the business world of decisions. The project manager coordinates multiple skills to produce a complete evaluation.

### Question 1: Does it work technically?

**Dispatch: The modeling skill used in Phase 4** (its evaluation section).
- Performance on held-out test data (not just cross-validation)
- Performance consistency across data subgroups
- Error analysis: where does the model fail, and are those failures acceptable?

### Question 2: Does it work for the business?

**Dispatch: `strategy-advisor`** if the business implications are significant.
- Does the model's performance level meet the success KPIs from the project charter?
- Would a human make better decisions using this model's predictions vs. without them?
- What's the expected business impact? (e.g., "If we target the top 20% of predicted churners with retention offers, we expect to retain X additional customers per month worth $Y")

### Question 3: Is it fair and explainable?

**Dispatch: `ds-model-explainability`** when interpretability depth or regulatory compliance is required. Invoke it when:
- The model will be used in a regulated domain (credit, healthcare, HR, insurance) and explainability is a compliance requirement
- Stakeholders need to understand *why* the model makes specific decisions, not just overall accuracy
- SHAP values, LIME, PDP/ICE plots, or counterfactual explanations are needed
- A thorough bias audit across subgroups is required
- A model card needs to be produced as a deployment artifact

For projects where basic fairness checks suffice, these are handled within the modeling skills. But `ds-model-explainability` owns the deep dive.

### Question 4: Can we communicate it effectively?

This is where the communication skills earn their keep:

**`visualization-expert`** — For creating clear, compelling charts that communicate model performance and findings. Invoke when standard matplotlib outputs aren't presentation-ready.

**`content-creator`** — For writing up findings in an engaging, audience-appropriate format. Invoke when the results need to reach a non-technical audience.

**`editor`** — For polishing any written deliverables. Invoke before anything goes to stakeholders.

**`pptx` / `docx`** — For creating professional presentation decks or formal reports. Invoke when the deliverable format is a slide deck or Word document.

### The Evaluation Decision

| Outcome | Action |
|---|---|
| Meets all KPIs, fair, stakeholder-approved | → Proceed to Phase 6 (Deployment) |
| Close to KPIs but not there yet | → Loop back to Phase 3/4 with targeted improvements |
| Fundamentally underperforming | → Loop back to Phase 1 — reframe the problem or acquire new data |
| Technically strong but stakeholder rejects | → Loop back to Phase 1 — misaligned expectations |

If this decision is complex, **invoke `decision-helper`** to structure the go/no-go analysis.

---

## Phase 6: Deployment

**Primary dispatches:**
- **`ds-ml-pipeline`** — Serialization and pipeline export (getting the model artifact production-ready)
- **`ds-mlops-deployment`** — The full production serving layer: building model APIs, batch jobs, monitoring pipelines, drift detection, retraining automation, and CI/CD for ML. Invoke when the conversation turns to "how does this actually run in production?"

Think of the division as: `ds-ml-pipeline` owns the model artifact; `ds-mlops-deployment` owns everything after the artifact exists.

The project manager ensures deployment planning covers the full picture.

**Supporting dispatches:**
- **`writing-plans`** — To create an executable deployment plan with step-by-step tasks
- **`project-planner`** — For deployment timeline, resource allocation, and risk mapping
- **`strategy-advisor`** — If the deployment strategy has business implications (phased rollout vs. big bang, pilot group selection)
- **`python-expert`** — For production-quality Python in deployment code (API handlers, monitoring scripts, retraining pipelines)

### Deployment Checklist
- How will the model receive new data? (API, batch job, embedded in product)
- Who is responsible for monitoring after deployment?
- What's the retraining cadence? (weekly, monthly, on data drift detection)
- What's the rollback plan if the model performs poorly in production?
- Is there a shadow/canary deployment phase before full rollout?

### Monitoring KPIs (Post-Deployment)
Define these before deploying:
- **Model performance**: Is prediction accuracy stable over time?
- **Data drift**: Has the input data distribution shifted from training data?
- **Business impact**: Is the expected ROI materializing?
- **Fairness drift**: Are fairness metrics stable across groups over time?

---

## KPI Design Framework

Good KPIs are the backbone of a successful project. Use this framework to design them.

### The KPI Pyramid

```
          ┌─────────────┐
          │  Business    │  ← What the stakeholder cares about
          │  KPIs        │     (revenue, retention, cost savings)
          ├─────────────┤
          │  Model       │  ← What tells us the model works
          │  KPIs        │     (F1, RMSE, AUC, silhouette score)
          ├─────────────┤
          │  Data        │  ← What tells us the inputs are healthy
          │  KPIs        │     (freshness, completeness, drift score)
          └─────────────┘
```

**Business KPIs** — Defined in Phase 1, validated in Phase 5:
- "Reduce customer churn by 15% in the targeted segment"
- "Decrease false positive fraud alerts by 30%"
- "Increase click-through rate on recommendations by 10%"

**Model KPIs** — Defined in Phase 1, optimized in Phase 4, validated in Phase 5:
- "Achieve F1 ≥ 0.80 on 30-day churn prediction"
- "RMSE ≤ $500 on quarterly revenue forecast"
- "Silhouette score ≥ 0.45 for customer segments"

**Data KPIs** — Monitored throughout:
- "Training data must be no more than 90 days old"
- "Missing value rate must stay below 5% on key features"
- "Feature distributions must not drift more than X from training baseline"

### Guardrail Metrics

In addition to optimization targets, define guardrails — metrics that must NOT degrade. Without guardrails, optimization can produce perverse outcomes.

---

## Risk Management

Identify risks early and revisit them at each phase transition.

### Common DS Project Risks

| Risk | Phase | Mitigation |
|---|---|---|
| Wrong question being answered | 1 | Explicit project charter, stakeholder sign-off. Use `brainstorming` to explore the problem space. |
| Insufficient or unavailable data | 2 | Early data audit via `ds-eda-process`, go/no-go gate after EDA |
| Data leakage inflating metrics | 3-4 | Use sklearn Pipelines via `ds-ml-pipeline` |
| Overfitting to training data | 4 | Proper CV, holdout test set, regularization — enforced by `ds-supervised-modeling` |
| Model performs well technically but stakeholder rejects it | 5 | Involve stakeholders early and often. Use `visualization-expert` and `content-creator` to communicate results clearly. |
| Model degrades in production | 6 | Monitoring, drift detection, retraining schedule |
| Scope creep ("can it also predict X?") | Any | Refer back to project charter, negotiate scope changes explicitly |
| Treating a causal question as a predictive one | 1-4 | Ask early "does the stakeholder want to predict or explain an effect?" — route to `ds-causal-inference` if causal |
| Time series data handled with standard CV | 4 | Detect temporal structure in Phase 2 and route to `ds-time-series` before modeling begins |
| Model deployed with no monitoring plan | 6 | Use `ds-mlops-deployment` to define drift detection and retraining cadence before go-live |
| Explainability needed but skipped | 5-6 | Ask in Phase 1 if regulatory or stakeholder explainability is required — invoke `ds-model-explainability` early |
| Data infrastructure missing but assumed to exist | 2-3 | Audit data availability in Phase 2; invoke `ds-data-engineering` if pipelines need to be built |
| Bias or fairness issues discovered late | 5 | Audit for fairness early (Phase 4), not just at evaluation |
| Poor communication of results | 5-6 | Use `editor` for polish, `pptx`/`docx` for professional deliverables |

---

## Stakeholder Communication

### Status Update Template

Use this for regular check-ins (weekly or at phase transitions). **Invoke `editor`** to polish before sending.

```markdown
## Project Status: [Project Name]
**Date**: [date]
**Current Phase**: [CRISP-DM phase]
**Overall Status**: [Green / Yellow / Red]

### Progress Since Last Update
- [What was accomplished]

### Key Findings
- [Insight 1 — stated in business terms]
- [Insight 2]

### Blockers or Risks
- [Issue and proposed resolution]

### Next Steps
- [What happens next and expected timeline]

### Decision Needed (if any)
- [Specific decision with options and recommendation]
```

### Communicating Results

When presenting findings, always lead with the insight, support with evidence:
- **For technical audiences**: include methodology, metrics, limitations, and code reproducibility
- **For non-technical audiences**: focus on what you found, why it matters, and what action to take. **Invoke `content-creator`** for audience-appropriate framing and **`visualization-expert`** for clear charts.
- **For executive presentations**: **Invoke `pptx`** to create a polished slide deck

---

## Quick-Start: "I Have a New Project"

When the user arrives with a new project, follow this sequence:

1. **Assess clarity** — Is the problem well-defined? If not, **invoke `brainstorming`** first.
2. **Understand the problem** — Use the stakeholder interview questions. Don't let the user skip this even if they're eager to start coding.
3. **Produce a project charter** — Even a brief one. It takes 10 minutes and saves weeks. Include the Skill Orchestration Plan.
4. **Define KPIs** — At minimum: one business KPI, one model KPI, one guardrail metric.
5. **Assess data availability** — What data exists? Who owns it? Can we access it?
6. **Create a phased plan** — Map the work to CRISP-DM phases with rough timelines. **Invoke `project-planner`** if the user needs formal timeline and resource planning.
7. **Build an execution plan** — **Invoke `writing-plans`** to create the step-by-step implementation plan.
8. **Identify the first specialist skill needed** — Usually `ds-eda-process` for Phase 2.
9. **Set the first milestone** — A concrete, verifiable deliverable (e.g., "EDA report with go/no-go recommendation by Friday").

## Quick-Start: "I'm Stuck Mid-Project"

When the user is already in the middle of a project and needs help:

1. **Diagnose the current phase** — Ask what they've done so far and what's blocking them.
2. **Identify the gap** — Are they stuck on a technical problem? A strategic question? A communication challenge?
3. **Dispatch the right specialist:**
   - Technical modeling issue → `ds-supervised-modeling` / `ds-unsupervised-learning` / `ds-nlp-cv-pipeline`
   - Time series or forecasting issue → `ds-time-series`
   - Causal / A/B test analysis → `ds-causal-inference`
   - Feature engineering plateau → `ds-feature-engineering`
   - Data infrastructure or pipeline issue → `ds-data-engineering`
   - Model explainability or fairness audit → `ds-model-explainability`
   - Deployment or production monitoring issue → `ds-mlops-deployment`
   - Data quality or understanding issue → `ds-eda-process` / `data-analyst`
   - Strategic or decision issue → `strategy-advisor` / `decision-helper`
   - Domain research or benchmarks needed → `deep-research`
   - Code quality or production Python needed → `python-expert`
   - Need to present or communicate → `visualization-expert` / `content-creator` / `pptx` / `docx`
   - Pipeline or reproducibility issue → `ds-ml-pipeline`
4. **Frame the context** — Before dispatching, always frame what the specialist needs to know about the larger project context.
5. **Interpret and integrate** — After the specialist skill does its work, interpret the results in the context of the overall project and recommend next steps.

## Quick-Start: "What Should I Do Next?"

When the user asks for guidance on next steps:

1. **Map to CRISP-DM** — Figure out which phase they're in based on what they've done.
2. **Check the phase completion criteria** — Has the current phase actually delivered what it should?
3. **Identify the transition** — What needs to happen to move to the next phase?
4. **Recommend the next skill dispatch** — Be specific: "Based on where you are, the next step is to invoke `ds-ml-pipeline` to build a reproducible preprocessing pipeline before you start modeling."
