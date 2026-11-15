"""Stratified train/test split (keeps the churn ratio equal in both sets)."""
from sklearn.model_selection import train_test_split

from src.clean import load_clean


def load_split(test_size=0.2, seed=42):
    df = load_clean()
    X, y = df.drop(columns="Churn"), df["Churn"]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    print("train:", X_train.shape, "churn", round(y_train.mean(), 3))
    print("test: ", X_test.shape, "churn", round(y_test.mean(), 3))
