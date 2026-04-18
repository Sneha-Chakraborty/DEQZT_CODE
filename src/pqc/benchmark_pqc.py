from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import pandas as pd

try:
    from pqc.zt_session import rotate_session
except ModuleNotFoundError as e:  # pragma: no cover
    rotate_session = None
    _IMPORT_ERR = e


def bench_pqc(n: int = 50) -> Dict:
    if rotate_session is None:
        raise ModuleNotFoundError(
            f"{_IMPORT_ERR}. Install PQC deps with: pip install -r requirements.txt (or: pip install pqcrypto)."
        )
    rows: List[Dict] = []
    for i in range(int(n)):
        out = rotate_session("userA", "aws", "STEP_UP", f"ctxhash{i}")
        out["run_index"] = i
        rows.append(out)
    df = pd.DataFrame(rows)
    return {
        "rotations_tested": int(len(df)),
        "avg_rotation_time_ms": float(df["rotation_time_ms"].mean()) if len(df) else 0.0,
        "p50_rotation_time_ms": float(df["rotation_time_ms"].median()) if len(df) else 0.0,
        "p95_rotation_time_ms": float(df["rotation_time_ms"].quantile(0.95)) if len(df) else 0.0,
        "avg_ciphertext_bytes": float(df["ciphertext_len"].mean()) if len(df) else 0.0,
        "avg_signature_bytes": float(df["signature_len"].mean()) if len(df) else 0.0,
        "avg_total_wire_bytes": float((df["ciphertext_len"] + df["signature_len"]).mean()) if len(df) else 0.0,
        "verify_ok_rate": float(df["verify_ok"].mean()) if len(df) else 0.0,
        "skipped": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=int(os.environ.get("DEQZT_PQC_BENCH_N", "50")))
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    summary = bench_pqc(args.n)
    out_path = os.path.join(args.out, "pqc_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Wrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
