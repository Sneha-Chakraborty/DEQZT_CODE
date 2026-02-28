# src/pqc/mlkem.py
from __future__ import annotations
import hashlib

# pqcrypto is an external dependency providing ML-KEM.
# We keep the import optional so the rest of the pipeline can run with --skip-pqc.
try:
    from pqcrypto.kem import ml_kem_768
except ModuleNotFoundError:  # pragma: no cover
    ml_kem_768 = None


def _require_pqcrypto():
    if ml_kem_768 is None:
        raise ModuleNotFoundError(
            "pqcrypto is not installed. Install it with: pip install -r requirements.txt "
            "(or: pip install pqcrypto). If you only want to run the ML baselines, run src\\run_all.py with --skip-pqc."
        )

def hkdf_sha256(ikm: bytes, info: bytes = b"DEQZT-SESSION-KEY", length: int = 32) -> bytes:
    """Lightweight KDF for prototype."""
    t = hashlib.sha256(ikm + info).digest()
    out = t
    while len(out) < length:
        t = hashlib.sha256(t + ikm + info).digest()
        out += t
    return out[:length]

class MLKEMSession:
    def __init__(self):
        self.alg = "ML-KEM-768"

    def server_generate_keypair(self):
        _require_pqcrypto()
        pk, sk = ml_kem_768.generate_keypair()
        return pk, sk

    def client_encapsulate(self, server_public_key: bytes):
        _require_pqcrypto()
        ct, ss = ml_kem_768.encrypt(server_public_key)
        return ct, ss

    def server_decapsulate(self, server_secret_key: bytes, ciphertext: bytes):
        _require_pqcrypto()
        ss = ml_kem_768.decrypt(server_secret_key, ciphertext)
        return ss

    def derive_aead_key(self, shared_secret: bytes) -> bytes:
        return hkdf_sha256(shared_secret, length=32)
