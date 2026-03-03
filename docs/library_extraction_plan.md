# ML Library Extraction Plan (Must-Extract Scope)

## Goals
- Reuse only parts that are truly common across labs.
- Keep algorithm-specific logic inside each lab.
- Build stable interfaces first, then move implementations.

## Must-Extract Components

### 1) Dataset Access Layer
- Why it is mandatory:
  - Every lab needs repeatable data loading.
  - Sources differ (xlsx/csv/UCI/Kaggle/moabb), but load contract is common.
- What to extract:
  - `DatasetBundle` data type with `features`, `targets`, `raw`.
  - Loader functions with clear signatures.
  - Optional registry to map dataset keys -> loader callables.
- Must not include:
  - Lab-specific cleaning or feature engineering.

### 2) Data Validation Layer
- Why it is mandatory:
  - Shape/type/null checks are repeated everywhere.
  - Early validation reduces debugging cost.
- What to extract:
  - dataframe type checks.
  - required columns checks.
  - target/feature consistency checks.
- Must not include:
  - domain assumptions (for example, EEG channel semantics).

### 3) Common Metrics (Classification First)
- Why it is mandatory:
  - Labs 4/5/6/7 require accuracy, precision, recall, F1 and ROC.
- What to extract:
  - pure numpy/pandas implementations for metrics.
  - confusion matrix helper.
  - ROC curve points and AUC utility.
- Must not include:
  - model-dependent threshold heuristics.

### 4) Timing / Experiment Tracking
- Why it is mandatory:
  - Lab 3 explicitly requires training time comparison.
  - Useful across all labs for reproducibility.
- What to extract:
  - timing context/decorator.
  - standardized run result object (`params`, `timing`, `metrics`, `notes`).

### 5) Plotting Helpers
- Why it is mandatory:
  - Repeated plots: heatmaps, ROC, feature importance, distributions.
- What to extract:
  - matplotlib wrappers with consistent style and save options.
- Must not include:
  - any domain-specific annotation logic.

### 6) Tree Split Criteria
- Why it is mandatory:
  - Reused in Lab 1 (gain ratio), Lab 5 (boosting trees), Lab 6 (RF trees).
- What to extract:
  - entropy, gini, information gain, split info, gain ratio.
- Must not include:
  - full tree training logic in this shared module.

### 7) Encoding Utilities
- Why it is mandatory:
  - Categorical-to-numeric conversion appears in multiple labs.
- What to extract:
  - label encoding.
  - one-hot encoding wrappers.
- Must not include:
  - per-lab column selection rules.

## Explicit Non-Extract Scope (Keep Local in Labs)
- Lab 2 metric choice rationale and domain analysis.
- Lab 3 transaction-specific recommendation behavior.
- Lab 4 P300 feature detection pipeline.
- Lab 1 final keep/drop feature decision rationale.
- Per-lab hyperparameter argumentation.

## Implementation Order
1. Base package skeleton (`types`, `validation`, `registry`).
2. Data loaders in shared package.
3. Lab 1 reusable block (cleaning, gain ratio, stats, plots, pipeline).
4. Metrics + ROC core.
5. Timing/reporting core.
6. Tree criteria and encoding utilities generalized for Labs 5/6/7.

## Acceptance Criteria
- Every extracted function is used (or immediately usable) by >= 2 contexts,
  except Lab 1 extraction bootstrap where code is intentionally first consumer.
- No extracted module depends on a specific lab notebook.
- No forbidden libraries (`scikit-learn`, `torch`, `tensorflow`, `keras`).
- API is small, explicit, and tested via smoke examples.
