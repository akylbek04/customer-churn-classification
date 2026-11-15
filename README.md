# Customer Churn Classification

Predicting which telecom customers will leave, using scikit-learn. The project focuses on handling class imbalance and evaluating with precision, recall and F1 rather than accuracy alone.

**Dataset:** Telco Customer Churn (7,043 customers, 19 features, 26.5% churned), loaded from OpenML (`id=42178`).

## Results (20% stratified test set)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | 5-fold CV F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.806 | 0.657 | 0.559 | 0.604 | 0.842 | 0.592 ± 0.033 |
| Logistic Regression (balanced) | 0.738 | 0.504 | 0.783 | 0.614 | 0.842 | 0.628 ± 0.026 |
| Random Forest (balanced) | 0.763 | 0.537 | 0.781 | 0.636 | 0.840 | 0.632 ± 0.017 |

Class weighting raises recall on churners from 0.56 to 0.78. Main drivers: short tenure, month-to-month contracts, fiber optic internet and electronic-check payment.

## Project structure

```
data/        dataset CSV
notebooks/   01_eda.ipynb, 02_model_evaluation.ipynb
src/         data download, cleaning, split, preprocessing, metrics, training scripts
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data            # download dataset
python -m src.train_logreg    # baseline
python -m src.train_balanced  # class-weighted
python -m src.train_forest    # Random Forest
jupyter notebook notebooks/
```

## Possible next steps
- Tune the decision threshold against a business cost
- Try gradient boosting
- Engineer features (e.g. number of services, charges per month of tenure)
