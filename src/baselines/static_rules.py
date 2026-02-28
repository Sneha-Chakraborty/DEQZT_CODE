from __future__ import annotations
import numpy as np
import pandas as pd

def static_rule_risk(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    risk = np.zeros(n, dtype=np.float32)
    status = df.get("status", pd.Series([""]*n)).fillna("").astype(str).str.lower().values
    action = df.get("action", pd.Series([""]*n)).fillna("").astype(str).values
    actcat = df.get("action_category", pd.Series([""]*n)).fillna("").astype(str).str.lower().values
    plane = df.get("plane", pd.Series([""]*n)).fillna("").astype(str).str.lower().values

    for i in range(n):
        r = 0.05
        st = status[i]; a = action[i]; ac = actcat[i]; pl = plane[i]

        if ("fail" in st) or ("denied" in st) or ("unauthorized" in st):
            r += 0.20

        al = a.lower()
        if ("login" in al) or ("consolelogin" in al) or ("assumerole" in al):
            r += 0.15
            if ("fail" in st) or ("denied" in st):
                r += 0.20

        if any(k.lower() in al for k in ["attach","putuserpolicy","putrolepolicy","createrole","setiampolicy","roleassignments"]):
            r += 0.35

        if any(k.lower() in al for k in ["stoplogging","deletetrail","updatetrail","deleteeventdatastore","disable"]):
            r += 0.40

        if any(k.lower() in al for k in ["getobject","listobjects","storage.objects","download","export"]):
            r += 0.35
            if ("read" in ac) or ("resource_read" in ac):
                r += 0.10

        if pl == "network":
            r += 0.10

        risk[i] = float(np.clip(r, 0.0, 1.0))
    return risk
