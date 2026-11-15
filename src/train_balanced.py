"""Logistic Regression with class_weight='balanced' to boost recall on churners."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.metrics import evaluate
from src.preprocess import make_preprocessor
from src.split import load_split

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    model = make_pipeline(
        make_preprocessor(), LogisticRegression(max_iter=1000, class_weight="balanced")
    ).fit(X_train, y_train)
    print("test:", {k: round(v, 3) for k, v in evaluate(model, X_test, y_test).items()})
