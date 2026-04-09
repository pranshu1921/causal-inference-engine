# Causal Inference A/B Testing Engine

A reusable Python toolkit for running causal inference analysis on A/B test and observational study data. Supports Propensity Score Matching (PSM), Difference-in-Differences (DiD), and Conditional Average Treatment Effect (CATE) estimation — all through an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-20%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen)

---

## Screenshots

### Dataset overview
![Dataset overview](docs/screenshots/01_dataset_overview.png)

### Propensity Score Matching — results and balance diagnostics
![PSM results](docs/screenshots/02_psm_results.png)

### Covariate balance table
![Balance table](docs/screenshots/03_psm_balance_table.png)

### CATE Estimation — heterogeneous treatment effects by subgroup
![CATE results](docs/screenshots/04_cate_results.png)

### Statistical Tests — t-test, Mann-Whitney, Bootstrap CI, power calculator
![Stats tests](docs/screenshots/05_stats_tests.png)

---

## What it does

Most A/B test analyses stop at a t-test. This engine goes further:

- **PSM** — matches treated and control units on propensity scores to reduce selection bias
- **DiD** — estimates treatment effect by comparing pre/post changes across groups
- **CATE** — reveals which subgroups benefit most from the treatment
- **Statistical tests** — t-test, chi-square, Mann-Whitney U, bootstrap confidence intervals
- **Balance diagnostics** — covariate balance plots before and after matching

## Project structure

```
causal-inference-engine/
├── src/
│   ├── data_loader.py      # CSV ingestion and validation
│   ├── psm.py              # Propensity Score Matching
│   ├── did.py              # Difference-in-Differences
│   ├── stats_tests.py      # Statistical testing utilities
│   ├── cate.py             # Conditional ATE estimation
│   └── visualizations.py  # All Plotly chart functions
├── app/
│   └── streamlit_app.py    # Main dashboard
├── tests/
│   ├── test_psm.py
│   ├── test_did.py
│   └── test_stats.py
├── data/                   # Place your CSV datasets here
├── requirements.txt
└── Dockerfile
```

## Quickstart

```bash
git clone https://github.com/pranshu1921/causal-inference-engine.git
cd causal-inference-engine
pip install -r requirements.txt
python data/generate_sample.py
streamlit run app/streamlit_app.py
```

> For the complete step-by-step setup guide including virtual environment setup,
> test instructions, Docker, and troubleshooting — see [RUNNING.md](RUNNING.md).

## Dataset format

Your CSV needs at minimum:
- A **treatment column** (binary: 0 or 1)
- An **outcome column** (numeric)
- One or more **covariate columns** for matching

A sample dataset based on the IBM HR Analytics Attrition dataset is included in `data/`.

## Methods explained

### Propensity Score Matching
Estimates each unit's probability of receiving treatment given its covariates. Treated units are then matched to the most similar control units, reducing confounding.

### Difference-in-Differences
Compares the change in outcomes over time between the treatment and control groups. Requires a pre/post time indicator column.

### CATE Estimation
Uses a GradientBoosting model to estimate how the treatment effect varies across different subgroups (e.g. age, tenure, department).

## Running tests

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Docker

```bash
docker build -t causal-engine .
docker run -p 8501:8501 causal-engine
```

Then open `http://localhost:8501`.

## Tech stack

| Package | Version | Purpose |
|---|---|---|
| statsmodels | 0.14.x | DiD, logistic regression |
| scikit-learn | 1.4.x | PSM, CATE model |
| scipy | 1.13.x | Statistical tests |
| pandas / numpy | 2.2 / 1.26 | Data manipulation |
| streamlit | 1.35.x | Dashboard UI |
| plotly | 5.22.x | All visualizations |
| pytest + hypothesis | 8.x / 6.x | Testing |

## License

MIT
