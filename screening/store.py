"""Privacy-preserving history store used for duplicate / velocity detection.

Identity numbers, phones and e-mails are stored only as salted SHA-256 hashes.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from typing import Dict, List, Optional

from .findings import Finding
from .forensics import hamming
from .ocr import name_similarity

SALT = os.environ.get("IDSCREEN_SALT", "change-me-in-production")


def h(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = "".join(str(value).split()).upper()
    return hashlib.sha256((SALT + v).encode()).hexdigest()


class HistoryStore:
    def __init__(self, path: str = "idscreen.db"):
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS submissions(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, name TEXT, dob TEXT,
            aadhaar TEXT, pan TEXT, phone TEXT, email TEXT, dhash TEXT,
            score REAL, decision TEXT)""")
        self.db.commit()

    def check(self, d: Dict[str, str], dhash: Optional[str]) -> List[Finding]:
        out: List[Finding] = []
        name, dob = d.get("name", ""), d.get("dob", "")
        cur = self.db.cursor()

        for col, label in (("aadhaar", "Aadhaar"), ("pan", "PAN")):
            key = h(d.get(col))
            if not key:
                continue
            for (pname, pdob) in cur.execute(f"SELECT name, dob FROM submissions WHERE {col}=?", (key,)):
                same_person = name_similarity(name, pname or "") > 0.8 and (pdob == dob)
                if not same_person:
                    out.append(Finding("duplicate", f"{label.upper()}_REUSED", "critical",
                                       f"This {label} number was previously submitted under a different name or date of birth.",
                                       {"previous_name": pname}))
                    break

        for col, label, limit in (("phone", "phone number", 2), ("email", "e-mail address", 2)):
            key = h(d.get(col))
            if not key:
                continue
            rows = cur.execute(f"SELECT DISTINCT aadhaar FROM submissions WHERE {col}=? AND aadhaar IS NOT NULL", (key,)).fetchall()
            mine = h(d.get("aadhaar"))
            others = {r[0] for r in rows if r[0] != mine}
            if len(others) >= limit:
                out.append(Finding("duplicate", f"{col.upper()}_SHARED", "high",
                                   f"The same {label} is linked to {len(others) + 1} different identities.",
                                   {"distinct_identities": len(others) + 1}))
            elif others:
                out.append(Finding("duplicate", f"{col.upper()}_SHARED", "medium",
                                   f"The same {label} is already linked to another identity.", {}))

        if dhash:
            mine = h(d.get("aadhaar")) or h(d.get("pan"))
            for (hh, a, p) in cur.execute("SELECT dhash, aadhaar, pan FROM submissions WHERE dhash IS NOT NULL"):
                if hh and hamming(hh, dhash) <= 10 and (a or p) != mine and (a or p):
                    out.append(Finding("duplicate", "IMAGE_REUSED", "high",
                                       "This document image is visually identical to one submitted for a different identity.", {}))
                    break

        cutoff = time.time() - 3600
        n = cur.execute("SELECT COUNT(*) FROM submissions WHERE ts>? AND (phone=? OR email=?)",
                        (cutoff, h(d.get("phone")), h(d.get("email")))).fetchone()[0]
        if n >= 3:
            out.append(Finding("duplicate", "VELOCITY", "medium", f"{n} submissions from the same contact in the last hour.", {"count": n}))
        return out

    def record(self, d: Dict[str, str], dhash: Optional[str], score: float, decision: str) -> None:
        self.db.execute(
            "INSERT INTO submissions(ts,name,dob,aadhaar,pan,phone,email,dhash,score,decision) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (time.time(), d.get("name"), d.get("dob"), h(d.get("aadhaar")), h(d.get("pan")),
             h(d.get("phone")), h(d.get("email")), dhash, score, decision))
        self.db.commit()

    def stats(self) -> Dict:
        c = self.db.cursor()
        rows = c.execute("SELECT decision, COUNT(*) FROM submissions GROUP BY decision").fetchall()
        total = sum(r[1] for r in rows)
        recent = c.execute("SELECT ts, name, score, decision FROM submissions ORDER BY id DESC LIMIT 10").fetchall()
        return {"total": total, "by_decision": dict(rows),
                "recent": [{"ts": r[0], "name": r[1], "score": r[2], "decision": r[3]} for r in recent]}
