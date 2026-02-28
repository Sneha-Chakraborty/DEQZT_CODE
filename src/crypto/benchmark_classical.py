# src/crypto/benchmark_classical.py
"""
Microbenchmark for classical (non-PQC) session rotation:
  - ECDHE (X25519 or P-256 ECDH)
  - Signature (Ed25519 or ECDSA P-256)

Writes summary JSON compatible with the pipeline's pqc_summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Dict, List

import pandas as pd

from crypto.classical_session import rotate_session_classical


def bench_classical(
    n: int = 200,
    *,
    kex: str = "x25519",
    sig: str = "ed25519",
) -> Dict:
    rows: List[Dict] = []
    for i in range(n):
        out = rotate_session_classical(
            subject="userA",
            cloud="aws",
            decision="STEP_UP",
            context_hash=f"ctx{i}",
            kex=kex,  # type: ignore[arg-type]
            sig_scheme=sig,  # type: ignore[arg-type]
        )
        out["run_index"] = i
        rows.append(out)

    df = pd.DataFrame(rows)
    summary = {
        "rotations_tested": int(len(df)),
        "avg_rotation_time_ms": float(df["rotation_time_ms"].mean()) if len(df) else 0.0,
        "p50_rotation_time_ms": float(df["rotation_time_ms"].median()) if len(df) else 0.0,
        "p95_rotation_time_ms": float(df["rotation_time_ms"].quantile(0.95)) if len(df) else 0.0,
        "avg_kex_wire_bytes": float(df["kex_total_wire_bytes"].mean()) if len(df) else 0.0,
        "avg_signature_bytes": float(df["signature_len"].mean()) if len(df) else 0.0,
        "avg_total_wire_bytes": float(df["total_wire_bytes"].mean()) if len(df) else 0.0,
        "verify_ok_rate": float(df["verify_ok"].mean()) if len(df) else 0.0,
        "kex_scheme": str(kex),
        "sig_scheme": str(sig),
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=int(os.environ.get("DEQZT_CLASSICAL_N", "200")))
    ap.add_argument("--kex", choices=["x25519", "p256"], default="x25519")
    ap.add_argument("--sig", choices=["ed25519", "ecdsa_p256"], default="ed25519")
    ap.add_argument("--out", type=str, default="results/tables")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    summary = bench_classical(args.n, kex=args.kex, sig=args.sig)

    out_path = os.path.join(args.out, f"classical_summary_{args.kex}_{args.sig}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Wrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
