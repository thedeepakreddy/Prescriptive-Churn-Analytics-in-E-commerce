# Prescriptive Churn Analytics — Profit-Optimized Retention in E-Commerce 🛒

**Most churn models answer "who will leave?" This one answers "who is worth saving, and what should we do about it today?"**

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline%20%2B%20RandomForest-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Tableau](https://img.shields.io/badge/Tableau-Dashboard-E97627?logo=tableau&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end analytics project on **50,000 e-commerce customers**: a leakage-proof ML pipeline predicts churn, a custom **profit simulator** converts those probabilities into dollars, and two dashboards hand the result to marketing as a **prioritized daily action list**.

---

## ⚡ Results at a Glance

| | Result |
|---|---|
| 🎯 **Model performance** | **ROC-AUC 0.9248** · Recall **0.79** · Precision **0.88** on the churn class |
| 💰 **Business impact** | **$1.51M** net retention profit on a 10,000-customer test cohort |
| 🔧 **Key decision** | Moved the decision threshold from the default `0.50` → **`0.28`**, chosen by profit, not accuracy |
| 📊 **Delivery** | Streamlit action app + Tableau executive dashboard — not just a notebook |
| 🧪 **Data** | 50,000 customers · 24 features · 28.9% churn · real-world missingness |

---

##  The Problem

A default classifier uses a `0.50` probability cutoff. That cutoff silently assumes **a lost $50 bargain hunter costs the same as a lost $5,000 loyalist.** It doesn't.

For retention, the two mistakes have wildly asymmetric price tags:

| Mistake | What it actually costs |
|---|---|
| **False Negative** — miss a churner | The customer's **entire Lifetime Value** |
| **False Positive** — discount someone who was staying anyway | Only the **margin on the discount** (~15% of LTV) |

Missing a churner is roughly **7× more expensive** than wasting a coupon. Any threshold tuned for accuracy or F1 is therefore leaving money on the table by construction.

##  The Approach

**1 · Build a leakage-proof pipeline.**
Split first, transform second. All imputation and scaling live inside a `ColumnTransformer` fit only on training data — no test-set statistics ever leak backwards.
- `KNNImputer` (k=5) for genuinely missing behavioural signals — *scaled before* distance calculation, since unscaled KNN imputation is silently dominated by whichever column has the largest units.
- Constant-zero imputation for columns where missing means *zero* (`Wishlist_Items`, `Customer_Service_Calls`, …) — an information-preserving choice, not a default.
- `City` dropped for cardinality that fragments tree splits; `Country` one-hot encoded instead.

**2 · Model the imbalance honestly.**
A `DummyClassifier` baseline scores 71% accuracy with **zero** recall on churners — proof that accuracy is a broken metric here. A `RandomForestClassifier(class_weight='balanced')`, tuned via `RandomizedSearchCV` on ROC-AUC with 3-fold CV, learns the minority class through re-weighting rather than synthetic oversampling (no SMOTE noise injected into a 40k-row training set).

**3 · Optimize the threshold for profit, not accuracy.**
A custom simulator sweeps 50 thresholds and scores each against a real cost matrix — 15% intervention cost, 70% campaign success rate, full LTV lost on every missed churner:

```python
tp_profit = ((LTV - cost) * SUCCESS_RATE).sum()   # saved, minus the discount
fp_loss   = -cost.sum()                           # wasted discount only
fn_loss   = -LTV.sum()                            # the expensive mistake
net_profit = tp_profit + fp_loss + fn_loss
```

The curve peaks at **0.28**. Deliberately tolerating more false positives buys back the borderline high-LTV customers a `0.50` cutoff throws away — and the notebook quantifies exactly what that trade is worth in dollars.

##  What the Model Found

- **`Customer_Service_Calls`** and **`Cart_Abandonment_Rate`** are the dominant churn drivers — both *operational* levers a business can actually pull, not demographics it can only observe.
- Repeat support contact is the strongest early-warning signal in the data: friction, not price, precedes departure.
- Every scored customer leaves the pipeline with an explicit prescription — `Trigger Retention Campaign (15% Discount)` or `Do Not Intervene` — exported straight to BI.

---

##  The Dashboards

### Streamlit — the operator's view
Filterable by country and minimum Lifetime Value, it turns model probabilities into a ranked intervention queue with live revenue-at-risk counters. A campaign manager can open it and start working the list.

![Streamlit Dashboard](assets/app%20dashboard.png)

### Tableau — the executive's view
The **Profit Quadrant** plots churn probability against Lifetime Value, visually separating low-value discount hunters from high-value flight risks, alongside a global risk map and a drill-down action roster.

![Tableau Dashboard](assets/dashboardd.png)

---

##  Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` (Pipeline, ColumnTransformer, KNNImputer, RandomForest, RandomizedSearchCV) · `Matplotlib` · `Seaborn` · `missingno` · `Streamlit` · `Tableau`

##  Repository Structure

```text
├── Notebook/
│   ├── ecommerce_notebook (1).ipynb        # Full analysis: EDA → pipeline → tuning → profit simulation
│   └── Book1.twb / Book1.twbx              # Tableau executive dashboard
├── data/
│   └── ecommerce_customer_churn_dataset.csv   # 50,000 customers × 24 features
├── assets/                                 # Dashboard screenshots
└── README.md
```

##  Run It Yourself

```bash
git clone https://github.com/thedeepakreddy/Prescriptive-Churn-Analytics-in-E-commerce.git
cd Prescriptive-Churn-Analytics-in-E-commerce

pip install pandas numpy scikit-learn matplotlib seaborn missingno jupyter
jupyter notebook "Notebook/ecommerce_notebook (1).ipynb"
```

The notebook runs top to bottom and exports `final_dashboard_data.csv` — the scored-and-prescribed customer table that feeds both dashboards. Open `Notebook/Book1.twbx` in Tableau Public (free) to explore the executive view.

---

##  What This Project Demonstrates

- **Framing ML as a business decision**, not a leaderboard score — cost-sensitive thresholding tied to Lifetime Value
- **Rigorous ML engineering** — leakage-proof `Pipeline` / `ColumnTransformer` design, deliberate imputation strategy, hyperparameter search on training data only
- **Handling messy, imbalanced, real-shaped data** — 28.9% minority class, missing values that mean different things per column, high-cardinality features
- **Communication across audiences** — the same model surfaced as an operator's task list *and* an executive's revenue view
- **End-to-end ownership** — raw CSV → model → financial simulation → decision support

---

##  Author

**Deepak Reddy**  Data Analyst / Data Scientist

📧 thedeepakreddy1@gmail.com · 🔗 [GitHub](https://github.com/thedeepakreddy)

*Open to Data Analyst, Data Scientist, and Analytics Engineer roles — happy to walk through the modeling decisions in this repo.*

---

<sub>Dataset is synthetic and used for demonstration purposes. Released under the MIT License.</sub>
