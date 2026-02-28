# src/pqc/benchmark_pqc.py
import time

try:
    from pqc.zt_session import rotate_session
except ModuleNotFoundError as e:
    rotate_session = None
    _IMPORT_ERR = e

def bench(n=50):
    if rotate_session is None:
        raise ModuleNotFoundError(
            f"{_IMPORT_ERR}. Install PQC deps with: pip install -r requirements.txt (or: pip install pqcrypto)."
        )
    times = []
    sizes = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = rotate_session("userA", "aws", "STEP_UP", "ctxhash123")
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        sizes.append((out["ciphertext_len"], out["signature_len"], out["session_key_len"]))
    avg_ms = sum(times)/len(times)
    ct_avg = sum(s[0] for s in sizes)//len(sizes)
    sig_avg = sum(s[1] for s in sizes)//len(sizes)
    print(f"Avg rotation time: {avg_ms:.2f} ms over {n} runs")
    print(f"Avg ciphertext bytes: {ct_avg}")
    print(f"Avg signature bytes: {sig_avg}")
    print("Example token json:", out["rotation_token_json"][:120] + "...")

if __name__ == "__main__":
    bench()