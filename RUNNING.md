# Running the Causal Inference Engine — Complete Setup Guide

This guide walks through every step from a fresh clone to a running dashboard,
including how to generate the sample dataset, run the test suite, and capture
screenshots for the README.

---

## Prerequisites

Before starting, confirm you have the following installed:

| Tool | Minimum version | Check command |
|---|---|---|
| Python | 3.11 | `python --version` |
| pip | 23.x | `pip --version` |
| Git | any | `git --version` |
| Docker (optional) | any | `docker --version` |

> **Python version matters.** Use 3.11 strictly. Some dependencies have known
> issues with 3.12. If you have multiple Python versions, use `python3.11`
> explicitly in the commands below.

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/pranshu1921/causal-inference-engine.git
cd causal-inference-engine
```

Verify the folder structure looks like this before continuing:

```
causal-inference-engine/
├── app/
│   └── streamlit_app.py
├── data/
│   └── generate_sample.py
├── src/
│   ├── __init__.py
│   ├── cate.py
│   ├── data_loader.py
│   ├── did.py
│   ├── psm.py
│   ├── stats_tests.py
│   └── visualizations.py
├── tests/
│   ├── test_did.py
│   ├── test_psm.py
│   └── test_stats.py
├── Dockerfile
├── README.md
├── RUNNING.md
├── pytest.ini
└── requirements.txt
```

---

## Step 2 — Create a virtual environment

Always work inside a virtual environment to avoid dependency conflicts.

**On Mac / Linux:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal prompt after activation.

---

## Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all 10 packages. Expected output ends with something like:

```
Successfully installed hypothesis-6.103.0 numpy-1.26.4 pandas-2.2.2
plotly-5.22.0 pytest-8.2.0 pytest-cov-5.0.0 scikit-learn-1.4.2
scipy-1.13.0 statsmodels-0.14.2 streamlit-1.35.0
```

If you see any red error lines, see the Troubleshooting section at the bottom.

---

## Step 4 — Generate the sample dataset

The app can use the built-in synthetic generator directly, but running this
script first saves the CSV to disk so you can also upload it manually:

```bash
python data/generate_sample.py
```

Expected output:
```
Sample dataset saved to data/hr_sample.csv
   treatment  performance_score  age  tenure  department  prior_score
0          0              63.45   34       8           2        61.23
1          1              71.20   45      12           0        66.87
...

Shape: (1000, 6)
Treatment rate: 43.20%
```

You should now see `data/hr_sample.csv` in the project folder.

---

## Step 5 — Run the test suite

Run all tests with coverage before launching the app. This confirms everything
is wired up correctly:

```bash
pytest
```

The `pytest.ini` file already sets `--cov=src --cov-report=term-missing -v`
so you get a full verbose output automatically.

Expected output:
```
tests/test_did.py::test_did_runs_without_error PASSED
tests/test_did.py::test_did_estimate_is_close_to_truth PASSED
tests/test_did.py::test_did_summary_keys PASSED
tests/test_did.py::test_did_group_means_shape PASSED
tests/test_did.py::test_did_raises_on_missing_column PASSED
tests/test_did.py::test_did_raises_on_non_binary_time PASSED
tests/test_psm.py::test_psm_runs_without_error PASSED
tests/test_psm.py::test_psm_matched_df_is_balanced PASSED
tests/test_psm.py::test_psm_summary_keys PASSED
tests/test_psm.py::test_psm_propensity_scores_between_0_and_1 PASSED
tests/test_psm.py::test_psm_raises_on_missing_column PASSED
tests/test_psm.py::test_psm_raises_on_empty_group PASSED
tests/test_stats.py::test_ttest_returns_expected_keys PASSED
tests/test_stats.py::test_ttest_detects_true_difference PASSED
...

---------- coverage: src ----------
Name                     Stmts   Miss  Cover
--------------------------------------------
src/cate.py                 68      4    94%
src/data_loader.py          61      3    95%
src/did.py                  72      5    93%
src/psm.py                  82      4    95%
src/stats_tests.py          78      2    97%
src/visualizations.py       96      8    92%
--------------------------------------------
TOTAL                       457     26    94%

20 passed in 12.43s
```

> **Screenshot opportunity #1:** Take a screenshot of this terminal output
> showing all tests passing with coverage. Save as `docs/screenshots/tests_passing.png`.

---

## Step 6 — Launch the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

Expected terminal output:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Your browser should open automatically. If it doesn't, open
`http://localhost:8501` manually.

---

## Step 7 — Walk through the dashboard and take screenshots

Follow each step below in the browser. After each one, take a screenshot and
save it to the `docs/screenshots/` folder using the filename shown.

### Create the screenshots folder first

```bash
mkdir -p docs/screenshots
```

---

### Screenshot 1 — Dataset overview

1. The app loads with the **built-in sample dataset** checked by default
2. You will see 4 metric cards at the top: Total rows, Treated, Control, Raw mean difference
3. Scroll down slightly so all 4 cards and the "Preview data" expander are visible
4. **Take screenshot** → save as `docs/screenshots/01_dataset_overview.png`

---

### Screenshot 2 — PSM results

1. Click the **"Propensity Score Matching"** tab
2. Click the **"Run PSM"** button
3. Wait for the spinner to finish (2–3 seconds)
4. You will see:
   - 3 metric cards: ATE (matched), 95% CI, Matched pairs
   - Propensity score distribution histogram
   - Covariate balance love plot
   - ATE bar chart with confidence interval
   - Balance diagnostics table
5. Scroll so the two charts are both visible
6. **Take screenshot** → save as `docs/screenshots/02_psm_results.png`

---

### Screenshot 3 — PSM balance diagnostics table

1. Still on the PSM tab, scroll down to the **"Balance diagnostics"** table
2. The table shows SMD before and after matching for each covariate
3. You should see a green success message: "All covariates are balanced after matching"
4. **Take screenshot** → save as `docs/screenshots/03_psm_balance_table.png`

---

### Screenshot 4 — CATE estimation

1. Click the **"CATE Estimation"** tab
2. In the **"Subgroup column"** dropdown, select `department`
3. Click **"Run CATE"**
4. Wait for the spinner (3–4 seconds — GradientBoosting trains two models)
5. You will see:
   - Overall ATE metric
   - ITE distribution histogram
   - CATE by department bar chart
   - Feature importance chart
6. **Take screenshot** → save as `docs/screenshots/04_cate_results.png`

---

### Screenshot 5 — Statistical tests

1. Click the **"Statistical Tests"** tab
2. This tab runs automatically — no button to press
3. You will see:
   - Outcome distribution box plot (treated vs control)
   - Normality test result
   - Three-column comparison: t-test, Mann-Whitney U, Bootstrap CI
   - Sample size calculator at the bottom
4. In the sample size calculator, change **Min detectable effect** to `3.0`
5. **Take screenshot** → save as `docs/screenshots/05_stats_tests.png`

---

### Screenshot 6 — Docker running (optional but recommended)

If you have Docker installed:

```bash
# Stop the streamlit process first (Ctrl+C), then:
docker build -t causal-engine .
docker run -p 8501:8501 causal-engine
```

Open `http://localhost:8501` again — the app should look identical.

**Take screenshot** of the running Docker container in terminal:
```bash
docker ps
```
Save as `docs/screenshots/06_docker_running.png`

---

## Step 8 — Add screenshots to the README

Open `README.md` and add the following section after the **Quickstart** section:

```markdown
## Screenshots

### Dataset overview
![Dataset overview](docs/screenshots/01_dataset_overview.png)

### Propensity Score Matching
![PSM results](docs/screenshots/02_psm_results.png)

### Balance diagnostics
![Balance table](docs/screenshots/03_psm_balance_table.png)

### CATE Estimation
![CATE results](docs/screenshots/04_cate_results.png)

### Statistical Tests
![Stats tests](docs/screenshots/05_stats_tests.png)
```

---

## Step 9 — Commit everything

```bash
git add docs/screenshots/
git add README.md
git commit -m "docs: add screenshots and running guide"
git push origin main
```

---

## Running with Docker (full walkthrough)

If you prefer Docker over a local Python environment:

```bash
# Build the image
docker build -t causal-engine .

# Run the container
docker run -p 8501:8501 causal-engine

# Open in browser
open http://localhost:8501        # Mac
start http://localhost:8501       # Windows
xdg-open http://localhost:8501    # Linux
```

To stop the container:
```bash
docker ps                         # get the container ID
docker stop <container_id>
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

You are running the app from the wrong directory. Make sure you are inside
the `causal-inference-engine/` folder:

```bash
cd causal-inference-engine
streamlit run app/streamlit_app.py
```

---

### "pip install fails on statsmodels or scikit-learn"

Try upgrading pip first:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

### "Python version is 3.12"

DoWhy and some scipy internals have issues on 3.12. Install Python 3.11:

- **Mac:** `brew install python@3.11`
- **Windows:** Download from [python.org](https://www.python.org/downloads/release/python-3119/)
- **Linux:** `sudo apt install python3.11`

Then recreate your virtual environment:

```bash
deactivate
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### "Streamlit opens but shows a blank white screen"

Hard-refresh the browser: `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` (Windows/Linux).

---

### "Tests pass but coverage is below 80%"

Check which lines are missing in the coverage report (the `Miss` column).
The most common cause is untested error paths. You can ignore this for now —
94%+ coverage is already excellent for a portfolio project.

---

### Port 8501 already in use

```bash
# Find what is using the port
lsof -i :8501          # Mac/Linux
netstat -ano | findstr :8501   # Windows

# Kill it, then restart
kill -9 <PID>          # Mac/Linux
taskkill /PID <PID> /F # Windows
```

---

## What a successful run looks like

After following all steps you should have:

- ✅ All 20 tests passing with ≥90% coverage
- ✅ Dashboard running at `http://localhost:8501`
- ✅ PSM producing matched pairs with balanced covariates
- ✅ CATE showing heterogeneous treatment effects by subgroup
- ✅ 5 screenshots saved to `docs/screenshots/`
- ✅ README updated with inline screenshot images
- ✅ Everything committed and pushed to GitHub
