"""Load the raw churn CSV and fix its known problems."""
import pandas as pd

from src.data import CSV_PATH


def load_clean(path=CSV_PATH):
    df = pd.read_csv(path)
    # OpenML wraps some category values in quotes, e.g. 'One year'
    for col in df.select_dtypes(exclude="number"):
        df[col] = df[col].astype(str).str.strip("'")
    # TotalCharges is text; 11 blanks belong to customers with tenure 0 (never billed)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


if __name__ == "__main__":
    df = load_clean()
    print(df.shape, "churn rate:", round(df["Churn"].mean(), 3))
    print(df["Contract"].unique())
