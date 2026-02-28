# src/crypto/classical_session.py
"""
Classical baseline session rotation primitives to compare with PQC (ML-KEM/ML-DSA).

We model a rotation as:
  - ephemeral ECDH key agreement (ECDHE) to derive a fresh session key
  - a signature over the transcript to provide authenticity (Ed25519 or ECDSA)

This is *not* a full TLS handshake implementation; it is a microbenchmark that
approximates the cryptographic costs and wire sizes of a ZT "session rotation" message.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import Literal, Tuple

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# ECDH options
from cryptography.hazmat.primitives.asymmetric import ec, x25519
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

# Signature options
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.utils import Prehashed
from cryptography.exceptions import InvalidSignature


KEX_SCHEME = Literal["x25519", "p256"]
SIG_SCHEME = Literal["ed25519", "ecdsa_p256"]


@dataclass
class ClassicalRotationResult:
    rotation_time_ms: float
    # "wire" size approximation:
    initiator_pub_len: int
    responder_pub_len: int
    kex_total_wire_bytes: int  # initiator_pub + responder_pub
    signature_len: int
    total_wire_bytes: int  # kex_total_wire_bytes + signature_len
    verify_ok: bool
    session_key_len: int
    rotation_token_json: str


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _derive_session_key(shared_secret: bytes, info: bytes, length: int = 32) -> bytes:
    """
    HKDF-SHA256 derives a fixed-length session key from the ECDH shared secret.
    """
    hkdf = HKDF(algorithm=hashes.SHA256(), length=length, salt=None, info=info)
    return hkdf.derive(shared_secret)


def _ecdhe_exchange(kex: KEX_SCHEME) -> Tuple[bytes, bytes, bytes]:
    """
    Returns (initiator_pub_bytes, responder_pub_bytes, shared_secret_bytes).
    Shared secret is confirmed to match on both sides.
    """
    if kex == "x25519":
        a_priv = x25519.X25519PrivateKey.generate()
        b_priv = x25519.X25519PrivateKey.generate()

        a_pub = a_priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        b_pub = b_priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

        # Both sides compute the same secret:
        secret_a = a_priv.exchange(b_priv.public_key())
        secret_b = b_priv.exchange(a_priv.public_key())
        if secret_a != secret_b:
            raise RuntimeError("X25519 shared secret mismatch (should never happen).")
        return a_pub, b_pub, secret_a

    if kex == "p256":
        a_priv = ec.generate_private_key(ec.SECP256R1())
        b_priv = ec.generate_private_key(ec.SECP256R1())

        a_pub = a_priv.public_key().public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
        b_pub = b_priv.public_key().public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)

        secret_a = a_priv.exchange(ec.ECDH(), b_priv.public_key())
        secret_b = b_priv.exchange(ec.ECDH(), a_priv.public_key())
        if secret_a != secret_b:
            raise RuntimeError("P-256 ECDH shared secret mismatch (should never happen).")
        return a_pub, b_pub, secret_a

    raise ValueError(f"Unknown kex scheme: {kex}")


def _sign_and_verify(sig_scheme: SIG_SCHEME, msg: bytes) -> Tuple[bytes, bool]:
    if sig_scheme == "ed25519":
        sk = Ed25519PrivateKey.generate()
        pk = sk.public_key()
        sig = sk.sign(msg)
        try:
            pk.verify(sig, msg)
            return sig, True
        except InvalidSignature:
            return sig, False

    if sig_scheme == "ecdsa_p256":
        sk = ec.generate_private_key(ec.SECP256R1())
        pk = sk.public_key()
        # ECDSA signs a hash; we'll hash the transcript first for stable measurement.
        digest = hashes.Hash(hashes.SHA256())
        digest.update(msg)
        msg_hash = digest.finalize()
        sig = sk.sign(msg_hash, ec.ECDSA(Prehashed(hashes.SHA256())))
        try:
            pk.verify(sig, msg_hash, ec.ECDSA(Prehashed(hashes.SHA256())))
            return sig, True
        except InvalidSignature:
            return sig, False

    raise ValueError(f"Unknown signature scheme: {sig_scheme}")


def rotate_session_classical(
    subject: str,
    cloud: str,
    decision: str,
    context_hash: str,
    *,
    kex: KEX_SCHEME = "x25519",
    sig_scheme: SIG_SCHEME = "ed25519",
) -> dict:
    """
    API-compatible (shape) with pqc.zt_session.rotate_session, so the pipeline can log similar fields.

    Returns a dict with:
      rotation_time_ms, ciphertext_len (0 for classical), signature_len, session_key_len,
      verify_ok, rotation_token_json, plus classical kex sizing fields.
    """
    t0 = time.perf_counter()

    a_pub, b_pub, shared = _ecdhe_exchange(kex)
    transcript = (
        f"{subject}|{cloud}|{decision}|{context_hash}|kex={kex}|sig={sig_scheme}".encode("utf-8")
        + b"|A=" + a_pub + b"|B=" + b_pub
    )
    sig, ok = _sign_and_verify(sig_scheme, transcript)

    session_key = _derive_session_key(shared, info=transcript[:64], length=32)

    token = {
        "v": 1,
        "alg": {"kex": kex, "sig": sig_scheme},
        "sub": subject,
        "cloud": cloud,
        "decision": decision,
        "ctx": context_hash,
        "a_pub_b64": _b64(a_pub),
        "b_pub_b64": _b64(b_pub),
        "sig_b64": _b64(sig),
        "ts_ms": int(time.time() * 1000),
    }
    token_json = json.dumps(token, separators=(",", ":"))

    t1 = time.perf_counter()

    initiator_len = len(a_pub)
    responder_len = len(b_pub)
    kex_wire = initiator_len + responder_len
    sig_len = len(sig)
    total_wire = kex_wire + sig_len

    return {
        "rotation_time_ms": (t1 - t0) * 1000.0,
        # PQC fields (to keep logs consistent):
        "ciphertext_len": 0,  # classical ECDH has no KEM ciphertext
        "signature_len": sig_len,
        "session_key_len": len(session_key),
        "verify_ok": bool(ok),
        "rotation_token_json": token_json,
        # Classical sizing fields:
        "initiator_pub_len": initiator_len,
        "responder_pub_len": responder_len,
        "kex_total_wire_bytes": kex_wire,
        "total_wire_bytes": total_wire,
        "kex_scheme": kex,
        "sig_scheme": sig_scheme,
    }
