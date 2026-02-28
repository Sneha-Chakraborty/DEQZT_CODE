# src/pqc/zt_session.py
from __future__ import annotations
import time, json
from dataclasses import dataclass

from pqc.mlkem import MLKEMSession
from pqc.mldsa import MLDSA

@dataclass
class RotationToken:
    subject: str
    cloud: str
    issued_at: float
    expires_at: float
    decision: str
    context_hash: str
    kem_alg: str
    sig_alg: str
    ciphertext_len: int
    signature_len: int
    rotation_time_ms: float

def rotate_session(subject: str, cloud: str, decision: str, context_hash: str, ttl_seconds: int = 300):
    """Quantum-resilient session rotation (ML-KEM-768 + ML-DSA-65)."""
    kem_alg = "ML-KEM-768"
    sig_alg = "ML-DSA-65"

    t0 = time.perf_counter()

    kem = MLKEMSession()
    sig = MLDSA()

    # Server keys (in production: HSM/KMS)
    server_pub, server_sec = kem.server_generate_keypair()
    id_pub, id_sec = sig.generate_keypair()

    # Key establishment
    ct, ss_client = kem.client_encapsulate(server_pub)
    ss_server = kem.server_decapsulate(server_sec, ct)
    if ss_client != ss_server:
        raise RuntimeError("ML-KEM shared secret mismatch")

    session_key = kem.derive_aead_key(ss_client)

    now = time.time()
    tok = RotationToken(
        subject=subject, cloud=cloud, issued_at=now, expires_at=now + float(ttl_seconds),
        decision=decision, context_hash=context_hash,
        kem_alg=kem_alg, sig_alg=sig_alg,
        ciphertext_len=len(ct), signature_len=0, rotation_time_ms=0.0
    )

    payload = json.dumps(tok.__dict__, separators=(",", ":"), sort_keys=True).encode("utf-8")
    signature = sig.sign(id_sec, payload)
    ok = sig.verify(id_pub, payload, signature)

    t1 = time.perf_counter()
    tok.signature_len = len(signature)
    tok.rotation_time_ms = (t1 - t0) * 1000.0

    return {
        "subject": subject,
        "cloud": cloud,
        "decision": decision,
        "context_hash": context_hash,
        "verify_ok": bool(ok),
        "kem_alg": kem_alg,
        "sig_alg": sig_alg,
        "ciphertext_len": int(len(ct)),
        "signature_len": int(len(signature)),
        "session_key_len": int(len(session_key)),
        "rotation_time_ms": float(tok.rotation_time_ms),
        "expires_at": float(tok.expires_at),
        "rotation_token_json": payload.decode("utf-8"),
        "ciphertext_preview_hex": ct[:16].hex(),
        "signature_preview_hex": signature[:16].hex(),
    }
