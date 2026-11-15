"""Random Forest with balanced class weights."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from src.metrics import evaluate
from src.preprocess import make_preprocessor
from src.split import load_split

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    model = make_pipeline(
        make_preprocessor(),
        RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                               n_jobs=-1, random_state=42),
    ).fit(X_train, y_train)
    print("test:", {k: round(v, 3) for k, v in evaluate(model, X_test, y_test).items()})
