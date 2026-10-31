# Credit Risk & Fraud Detection System

A production-style credit risk model and fraud detection system built on 150,000 real borrower records. Covers the full pipeline from raw data to a combined lending decision per borrower.

**Author:** [Vishnu K Nair](https://github.com/Vkn5712)

## Results

| Model | AUC-ROC | Gini | KS Statistic |
|-------|---------|------|-------------|
| Logistic Regression | 0.863 | 0.727 | — |
| Decision Tree | 0.857 | 0.715 | — |
| **XGBoost** | **0.868** | **0.735** | **0.577** |

**Risk Tier Scorecard (XGBoost)**

| Tier | % of Portfolio | Actual Default Rate |
|------|---------------|-------------------|
| Low Risk | 8.35% | 0.48% |
| Medium Risk | 31.29% | 0.84% |
| High Risk | 60.36% | 10.57% |

22x default rate spread between Low and High risk tiers.

**Fraud Detection (Isolation Forest)**

| Group | Default Rate |
|-------|-------------|
| Normal borrowers | 5.84% |
| Anomalous borrowers (flagged) | 22.81% |

4x default rate lift in flagged group.

**Combined Lending Decision Matrix**

| Decision | % of Portfolio | Actual Default Rate |
|----------|---------------|-------------------|
| Approve | 8.2% | 0.13% |
| Approve with Conditions | 31.0% | 0.48% |
| Reject | 55.5% | 9.64% |
| Review -- Fraud Flag | 5.3% | 22.24% |

171x default rate difference between the Approve and Review buckets.

## What This Project Covers

**Part 1 -- Credit Risk**
- Exploratory data analysis: class imbalance (6.68% default rate, 14:1 ratio), distributions, correlation heatmap
- Data cleaning: median imputation for missing income (19.82% missing), 99th percentile outlier capping
- Feature engineering: 7 derived features including DTI ratio, composite delinquency score, and high utilization flag -- top 2 engineered features accounted for 70% of XGBoost model gain
- Model training: Logistic Regression (baseline), Decision Tree, XGBoost with class imbalance handling via scale_pos_weight
- Evaluation: AUC-ROC, Gini coefficient, KS statistic, confusion matrix, classification report
- Feature importance analysis
- Risk tier segmentation: Low / Medium / High risk with business decision logic

**Part 2 -- Fraud Detection**
- Rule-based fraud signals: income inflation, identity theft, loan stacking, round income flags
- Isolation Forest anomaly detection (no fraud labels required)
- Anomalous borrower profiling: debt ratio 12x higher, delinquencies 5.75x higher than normal population
- Combined credit + fraud decision matrix: Approve / Approve with Conditions / Reject / Review

## Tech Stack

- **Python** -- pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, scipy
- **Models** -- Logistic Regression, Decision Tree, XGBoost, Isolation Forest
- **Metrics** -- AUC-ROC, Gini coefficient, KS statistic, confusion matrix
- **Environment** -- Jupyter Notebook / Google Colab

## Dataset

**Give Me Some Credit** -- Kaggle
Link: https://www.kaggle.com/c/GiveMeSomeCredit/data
File needed: `cs-training.csv` (150,000 rows, 11 features)

The dataset is not included in this repo. Download it from Kaggle and place it in the same directory as the notebook before running.

## How to Run

**Option 1 -- Google Colab (recommended)**
1. Open `credit_risk_model.ipynb` in Google Colab
2. Run the first cell -- it will prompt you to upload `cs-training.csv` directly
3. Run all remaining cells in order

**Option 2 -- Local (Jupyter)**
1. Install dependencies:
2. Place `cs-training.csv` in the same folder as the notebook
3. Run all cells

## Key Findings

1. **Past delinquency behaviour dominates prediction** -- has_delinquency and total_delinquencies together account for 70% of XGBoost model gain, outperforming raw income and debt ratio features
2. **KS of 0.577 exceeds production thresholds** -- industry benchmark for a good credit model is KS > 0.3; very good is KS > 0.4
3. **Fraud detection catches what credit scoring misses** -- the Review bucket has a 22.24% default rate vs 0.13% in the clean Approve bucket, a 171x difference
4. **Isolation Forest works without fraud labels** -- anomalous profiles identified purely from statistical deviation, with a 4x default rate lift over the normal population

## Project Structure
