# DEQ-ZT : Dirichlet-Evidential, Quantum-Resilient Zero Trust Control-Loop Framework in Multi-Cloud Environments

### DEQZT: Benchmarking Codebase (2 Baselines + Proposed Dirichlet Uncertainty-Aware ZT Loop)

This project evaluates:
1) **Isolation Forest baseline**
2) **Static rule-based Zero Trust baseline**
3) **Proposed DEQZT**: Dirichlet mean + uncertainty-aware ZT decision loop (step-up / revoke)

It consumes a single input file:
- `dirichlet_training.parquet` (your unified Dirichlet-ready training table)

## 0) Folder layout to use on Windows
Recommended:
- `D:\DEQZT\data\dirichlet_training.parquet`
- Put this codebase anywhere, e.g. `D:\DEQZT_CODE\`

## 1) Install (PowerShell)
```powershell
cd <this_folder>
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Notes on PQC dependency
This codebase uses **PQCrypto** for ML-KEM-768 and ML-DSA-65. The package name is `pqcrypto`.

If you only want to run the analytics + baselines (and skip the PQC session-rotation step), you can run with:
```powershell
python src\run_all.py --data D:\DEQZT\data\dirichlet_training.parquet --config configs\config.yaml --skip-pqc
```

## 2) Run end-to-end
Edit `configs/config.yaml` if you want to change weights / sweep ranges.

Then run:
```powershell
python src\run_all.py --data D:\DEQZT\data\dirichlet_training.parquet --config configs\config.yaml
```

Outputs:
- `results/tables/metrics_event_level.csv`
- `results/tables/metrics_session_level.csv`
- `results/figures/*.png`
- `results/tables/thresholds.json`
## 3) Crypto overhead add-on (PQC vs Classical) for paper tables

After you have run the full pipeline once (so `results/tables/pqc_summary.json` exists), generate a
"PQC cost vs classical" comparison table:

```powershell
python src\compare_crypto.py --pqc-summary results\tables\pqc_summary.json --out results\tables --n 200
```

This produces:
- `results/tables/classical_summary_x25519_ed25519.json`
- `results/tables/classical_summary_p256_ecdsa_p256.json`
- `results/tables/crypto_comparison.csv`
- `results/tables/crypto_comparison.tex` (ready to paste into IEEE/LNCS LaTeX)

Notes:
- Classical baseline models an ECDHE key agreement + signature over the transcript (microbenchmark, not full TLS).
- Total wire bytes are approximated as (ECDHE public keys exchanged) + (signature).


## 4) Security/control-loop crypto metrics (10-year scenario + rotation analytics)

The pipeline now writes an evaluation trace used to compute control-loop metrics:
- `results/tables/eval_trace.csv`  (risk, uncertainty, decisions, session id, attack indicator)

After running `src/run_all.py`, generate:
- Raw overhead table: `crypto_comparison.csv` / `crypto_comparison.tex`
- Security/control tables: `crypto_security_control_metrics.csv`,
  `crypto_security_control_metrics_econ.tex`,
  `crypto_security_control_metrics_control.tex`

Example (10-year scenario):
```powershell
python src\compare_crypto.py ^
  --pqc-summary results\tables\pqc_summary.json ^
  --eval-trace results\tables\eval_trace.csv ^
  --session-metrics results\tables\metrics_session_level.csv ^
  --out results\tables ^
  --n 200 ^
  --horizon-years 10 ^
  --impact 1.0 ^
  --p-break-classical 0.30 ^
  --p-break-pqc 0.05 ^
  --ttl-steps 10
```

### Metrics included in the security/control tables
- Expected Loss: `E[L] = P(broken within 10y) * Impact`
- Loss Remaining / Loss Reduction (%)
- Rotation Coverage Gain (high-risk coverage × survival probability)
- Risk-Weighted Security Gain (risk-weighted coverage × survival probability)
- Detection-to-Rotation delay (P95 in steps; and seconds if timestamps are available)
- Crypto overhead per attack session contained (Bytes/Contain, Ms/Contain)
- Rotation selectivity: precision and recall w.r.t. attack-labeled events
