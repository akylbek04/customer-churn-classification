"""Column transformer: one-hot encode categories, scale numbers."""
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), make_column_selector(dtype_include="number")),
        ("cat", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_exclude="number")),
    ])


if __name__ == "__main__":
    from src.split import load_split
    X_train, *_ = load_split()
    Xt = make_preprocessor().fit_transform(X_train)
    print("before:", X_train.shape, "after:", Xt.shape)
