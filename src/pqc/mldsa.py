# src/pqc/mldsa.py
from __future__ import annotations

# pqcrypto is an external dependency providing ML-DSA.
# We keep the import optional so the rest of the pipeline can run with --skip-pqc.
try:
    from pqcrypto.sign import ml_dsa_65
except ModuleNotFoundError:  # pragma: no cover
    ml_dsa_65 = None


def _require_pqcrypto():
    if ml_dsa_65 is None:
        raise ModuleNotFoundError(
            "pqcrypto is not installed. Install it with: pip install -r requirements.txt "
            "(or: pip install pqcrypto). If you only want to run the ML baselines, run src\\run_all.py with --skip-pqc."
        )

# ML-DSA-65 secret key size enforced by pqcrypto/PQClean
_MLDSA65_SK_LEN = 4032  # bytes

class MLDSA:
    def __init__(self):
        self.alg = "ML-DSA-65"

    def generate_keypair(self):
        """Return (public_key, secret_key) robustly regardless of underlying order."""
        _require_pqcrypto()
        a, b = ml_dsa_65.generate_keypair()

        # Some builds may return (sk, pk). Detect by length.
        if len(a) == _MLDSA65_SK_LEN and len(b) != _MLDSA65_SK_LEN:
            sk, pk = a, b
        elif len(b) == _MLDSA65_SK_LEN and len(a) != _MLDSA65_SK_LEN:
            pk, sk = a, b
        else:
            # Fallback: assume documented order (pk, sk)
            pk, sk = a, b

        return pk, sk

    def sign(self, secret_key: bytes, message: bytes) -> bytes:
        _require_pqcrypto()
        if len(secret_key) != _MLDSA65_SK_LEN:
            raise ValueError(
                f"ML-DSA-65 secret_key length must be {_MLDSA65_SK_LEN} bytes, got {len(secret_key)}. " 
                "Keys may be swapped; pass the SECRET key from generate_keypair()."
            )
        # pqcrypto API: sign(secret_key, message)
        return ml_dsa_65.sign(secret_key, message)

    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        _require_pqcrypto()
        try:
            # pqcrypto API: verify(public_key, message, signature) -> bool
            return bool(ml_dsa_65.verify(public_key, message, signature))
        except Exception:
            return False
