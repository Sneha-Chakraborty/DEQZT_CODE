from __future__ import annotations

import argparse
import os
import yaml

import pandas as pd

from utils.io import read_parquet, resolve_data_path, time_aware_split
from deqzt.edl_pipeline import train_edl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to dirichlet_training.parquet (or folder containing it).")
    ap.add_argument("--config", default="configs/config.yaml", help="Config YAML.")
    ap.add_argument("--out", default="models/edl", help="Output directory for EDL artifacts.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    label_col = cfg["data"]["label_column"]
    time_col = cfg["data"]["time_column"]
    hypotheses = cfg["dirichlet"]["hypotheses"]
    ratios = cfg["split"]

    edl_cfg = cfg.get("edl", {})

    df = read_parquet(resolve_data_path(args.data))
    train, val, _ = time_aware_split(df, time_col=time_col, ratios=ratios)

    artifacts, _ = train_edl(
        train, val,
        label_col=label_col,
        hypotheses=hypotheses,
        prefixes=tuple(edl_cfg.get("feature_prefixes", ["e__", "f__"])),
        max_features=int(edl_cfg.get("max_features", 256)),
        hidden_sizes=tuple(edl_cfg.get("hidden_sizes", [256, 128])),
        dropout=float(edl_cfg.get("dropout", 0.2)),
        lr=float(edl_cfg.get("lr", 1e-3)),
        epochs=int(edl_cfg.get("epochs", 15)),
        batch_size=int(edl_cfg.get("batch_size", 1024)),
        anneal_epochs=int(edl_cfg.get("anneal_epochs", 10)),
        weight_decay=float(edl_cfg.get("weight_decay", 1e-4)),
        loss_type=str(edl_cfg.get("loss_type", "log")),
        seed=int(edl_cfg.get("seed", 42)),
        device=str(edl_cfg.get("device", "cpu")),
        verbose=True,
    )

    artifacts.save(args.out)
    print(f"[OK] Saved EDL artifacts to: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
