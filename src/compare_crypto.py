# src/compare_crypto.py
"""
Convenience wrapper to run crypto comparison from project root:

  python src\compare_crypto.py --pqc-summary results\tables\pqc_summary.json

Keeps the same CLI as crypto/compare_crypto.py.
"""
from __future__ import annotations

from crypto.compare_crypto import main

if __name__ == "__main__":
    main()
