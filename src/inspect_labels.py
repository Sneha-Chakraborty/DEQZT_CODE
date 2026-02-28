import pandas as pd
import argparse

def main(data_path):
    print("Loading dataset...")
    df = pd.read_parquet(data_path, engine="pyarrow")
    print("\n=== BASIC INFO ===")
    print("Total rows:", len(df))
    print("\n=== HYPOTHESIS CLASS DISTRIBUTION ===")
    print(df["hypothesis"].value_counts(dropna=False))
    print("\n=== ATTACK VS BENIGN DISTRIBUTION ===")
    print(df["is_attack"].value_counts(dropna=False))
    print("\n=== DATASET SOURCE DISTRIBUTION ===")
    print(df["dataset"].value_counts(dropna=False))
    print("\n=== LABELED VS UNLABELED ===")
    labeled = df["hypothesis"].ne("UNLABELED").sum()
    unlabeled = (df["hypothesis"] == "UNLABELED").sum()
    print("Labeled:", labeled)
    print("Unlabeled:", unlabeled)
    if "uncertainty_u" in df.columns:
        print("\n=== UNCERTAINTY STATISTICS ===")
        print(df["uncertainty_u"].describe())
    if "S" in df.columns:
        print("\n=== DIRICHLET STRENGTH S STATISTICS ===")
        print(df["S"].describe())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    main(args.data)
