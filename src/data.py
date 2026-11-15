"""Download the Telco customer churn dataset (OpenML id 42178) and save it as a CSV."""
from pathlib import Path

from sklearn.datasets import fetch_openml

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "telco_churn.csv"


def download(path=CSV_PATH):
    df = fetch_openml(data_id=42178, as_frame=True).frame
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    df = download()
    print(f"Saved {len(df)} rows x {df.shape[1]} columns to {CSV_PATH}")
