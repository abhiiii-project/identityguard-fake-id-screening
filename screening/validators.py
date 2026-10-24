"""Deterministic structural validators for identity numbers and fields.

These catch the large class of fakes that simply have invalid numbers:
 * Aadhaar  - 12 digits with a Verhoeff checksum
 * PAN      - AAAAA9999A with a valid holder-type letter
 * Passport - ICAO 9303 MRZ (TD3) check digits
 * Dates    - DOB / issue / expiry sanity
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import List, Optional

from .findings import Finding

# ------------------------------------------------------------------ Verhoeff
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_valid(number: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_generate_check_digit(number: str) -> str:
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return str(_INV[c])


# ------------------------------------------------------------------ Aadhaar
def validate_aadhaar(value: str) -> List[Finding]:
    out: List[Finding] = []
    digits = re.sub(r"[\s-]", "", value or "")
    if not re.fullmatch(r"\d{12}", digits):
        return [Finding("validation", "AADHAAR_FORMAT", "high",
                        "Aadhaar number must contain exactly 12 digits.", {"value": value})]
    if digits[0] in "01":
        out.append(Finding("validation", "AADHAAR_PREFIX", "high",
                           "Aadhaar numbers never begin with 0 or 1.", {"value": digits}))
    if not verhoeff_valid(digits):
        out.append(Finding("validation", "AADHAAR_CHECKSUM", "critical",
                           "Aadhaar number fails the Verhoeff checksum, so it cannot be a real number.",
                           {"value": digits}))
    if len(set(digits)) <= 2 or digits in {"123456789012", "012345678901"}:
        out.append(Finding("validation", "AADHAAR_PATTERN", "high",
                           "Aadhaar number looks like a repeated or sequential placeholder.", {"value": digits}))
    return out


# ---------------------------------------------------------------------- PAN
PAN_HOLDER_TYPES = {
    "P": "Individual", "C": "Company", "H": "HUF", "F": "Firm", "A": "AOP",
    "T": "Trust", "B": "BOI", "L": "Local Authority", "J": "Artificial Juridical Person",
    "G": "Government",
}


def validate_pan(value: str, surname: Optional[str] = None) -> List[Finding]:
    pan = (value or "").strip().upper()
    if not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan):
        return [Finding("validation", "PAN_FORMAT", "high",
                        "PAN must follow the pattern AAAAA9999A.", {"value": value})]
    out: List[Finding] = []
    if pan[3] not in PAN_HOLDER_TYPES:
        out.append(Finding("validation", "PAN_HOLDER_TYPE", "high",
                           f"4th character '{pan[3]}' is not a valid PAN holder type.", {"value": pan}))
    if surname and pan[3] == "P" and surname.strip():
        if pan[4] != surname.strip()[0].upper():
            out.append(Finding("consistency", "PAN_SURNAME_MISMATCH", "medium",
                               "5th PAN character should be the first letter of the holder's surname.",
                               {"pan_char": pan[4], "surname_initial": surname.strip()[0].upper()}))
    if pan[5:9] == "0000":
        out.append(Finding("validation", "PAN_SEQUENCE", "medium",
                           "PAN sequence number 0000 is not issued.", {"value": pan}))
    return out


# ----------------------------------------------------------------- Passport
def _mrz_value(ch: str) -> int:
    if ch.isdigit():
        return int(ch)
    if ch == "<":
        return 0
    return ord(ch) - 55  # A=10 ...


def mrz_check_digit(field: str) -> str:
    weights = [7, 3, 1]
    return str(sum(_mrz_value(c) * weights[i % 3] for i, c in enumerate(field)) % 10)


def validate_mrz_td3(line1: str, line2: str) -> List[Finding]:
    """Validate a 2x44 passport machine readable zone."""
    l1, l2 = (line1 or "").strip().upper(), (line2 or "").strip().upper()
    if len(l1) != 44 or len(l2) != 44:
        return [Finding("validation", "MRZ_LENGTH", "high",
                        "Passport MRZ must be two lines of 44 characters.",
                        {"line1_len": len(l1), "line2_len": len(l2)})]
    out: List[Finding] = []
    checks = {
        "document number": (l2[0:9], l2[9]),
        "date of birth": (l2[13:19], l2[19]),
        "expiry date": (l2[21:27], l2[27]),
        "personal number": (l2[28:42], l2[42]),
    }
    for name, (data, chk) in checks.items():
        if chk != "<" or name != "personal number":
            if mrz_check_digit(data) != chk:
                out.append(Finding("validation", "MRZ_CHECKSUM", "critical",
                                   f"MRZ check digit for {name} is wrong.",
                                   {"field": name, "data": data, "expected": mrz_check_digit(data), "found": chk}))
    composite = l2[0:10] + l2[13:20] + l2[21:43]
    if mrz_check_digit(composite) != l2[43]:
        out.append(Finding("validation", "MRZ_COMPOSITE", "critical",
                           "MRZ composite check digit is wrong.", {}))
    return out


def parse_mrz_td3(line1: str, line2: str) -> dict:
    l1, l2 = line1.upper(), line2.upper()
    names = l1[5:44].split("<<")
    return {
        "surname": names[0].replace("<", " ").strip(),
        "given_names": (names[1] if len(names) > 1 else "").replace("<", " ").strip(),
        "document_number": l2[0:9].replace("<", ""),
        "nationality": l2[10:13],
        "dob": l2[13:19], "sex": l2[20], "expiry": l2[21:27],
    }


# -------------------------------------------------------------------- Dates
DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y", "%d %b %Y", "%d %B %Y")


def parse_date(value: str) -> Optional[date]:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except (ValueError, AttributeError):
            continue
    return None


def validate_dates(dob: Optional[str] = None, issue: Optional[str] = None,
                   expiry: Optional[str] = None, today: Optional[date] = None) -> List[Finding]:
    today = today or date.today()
    out: List[Finding] = []
    d_dob, d_iss, d_exp = (parse_date(x) if x else None for x in (dob, issue, expiry))
    if dob and not d_dob:
        out.append(Finding("validation", "DOB_UNPARSEABLE", "medium", "Date of birth is not a valid date.", {"value": dob}))
    if d_dob:
        age = (today - d_dob).days / 365.25
        if d_dob > today:
            out.append(Finding("validation", "DOB_FUTURE", "critical", "Date of birth is in the future.", {"value": dob}))
        elif age < 18:
            out.append(Finding("validation", "UNDERAGE", "medium", f"Holder appears to be under 18 (age {age:.1f}).", {"age": round(age, 1)}))
        elif age > 100:
            out.append(Finding("validation", "AGE_IMPLAUSIBLE", "high", f"Implausible age ({age:.0f}).", {"age": round(age)}))
    if d_iss and d_iss > today:
        out.append(Finding("validation", "ISSUE_FUTURE", "high", "Issue date is in the future.", {"value": issue}))
    if d_iss and d_dob and d_iss < d_dob:
        out.append(Finding("validation", "ISSUE_BEFORE_BIRTH", "critical", "Document issued before the holder was born.", {}))
    if d_exp and d_iss and d_exp <= d_iss:
        out.append(Finding("validation", "EXPIRY_BEFORE_ISSUE", "critical", "Expiry date is on or before issue date.", {}))
    if d_exp and d_exp < today:
        out.append(Finding("validation", "DOC_EXPIRED", "high", f"Document expired on {d_exp.isoformat()}.", {"expiry": d_exp.isoformat()}))
    return out
