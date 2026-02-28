# src/benchmark_classical.py
"""
Convenience wrapper to run classical crypto benchmark from project root:

  python src\benchmark_classical.py --kex x25519 --sig ed25519
"""
from __future__ import annotations

from crypto.benchmark_classical import main

if __name__ == "__main__":
    main()
