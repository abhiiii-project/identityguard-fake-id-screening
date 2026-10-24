"""Synthetic / fake identity signals computed from the declared applicant data."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

from .findings import Finding

DISPOSABLE_DOMAINS = {
    "mailinator.com", "guerrillamail.com", "10minutemail.com", "tempmail.com", "temp-mail.org",
    "yopmail.com", "trashmail.com", "throwawaymail.com", "getnada.com", "sharklasers.com",
    "dispostable.com", "maildrop.cc", "fakeinbox.com", "mintemail.com", "moakt.com", "emailondeck.com",
}
PLACEHOLDER_NAMES = {"test", "testing", "demo", "sample", "fake", "abc", "xyz", "asdf", "qwerty",
                     "john doe", "jane doe", "name", "user", "admin", "unknown", "na", "none"}


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s)
    return -sum(v / len(s) * math.log2(v / len(s)) for v in c.values())


def _looks_random(token: str) -> bool:
    t = re.sub(r"[^a-z0-9]", "", token.lower())
    if len(t) < 8:
        return False
    digit_ratio = sum(ch.isdigit() for ch in t) / len(t)
    vowels = sum(ch in "aeiou" for ch in t) / len(t)
    return (digit_ratio > 0.45) or (_entropy(t) > 3.3 and vowels < 0.2)


def check_name(name: str) -> List[Finding]:
    out: List[Finding] = []
    n = (name or "").strip()
    if not n:
        return [Finding("identity", "NAME_MISSING", "medium", "No name was supplied.", {})]
    low = n.lower()
    if low in PLACEHOLDER_NAMES or any(w in PLACEHOLDER_NAMES for w in low.split() if len(w) > 3):
        out.append(Finding("identity", "NAME_PLACEHOLDER", "high", "Name looks like a placeholder or test value.", {"name": n}))
    if re.search(r"\d", n):
        out.append(Finding("identity", "NAME_DIGITS", "high", "Name contains digits.", {"name": n}))
    if re.search(r"(.)\1{3,}", low):
        out.append(Finding("identity", "NAME_REPEAT", "medium", "Name contains a long run of repeated characters.", {"name": n}))
    letters = re.sub(r"[^a-z]", "", low)
    if len(letters) >= 5 and sum(ch in "aeiouy" for ch in letters) / len(letters) < 0.15:
        out.append(Finding("identity", "NAME_GIBBERISH", "medium", "Name has too few vowels to look like a real name.", {"name": n}))
    if len(n.split()) < 2:
        out.append(Finding("identity", "NAME_SINGLE_TOKEN", "low", "Only one name token was supplied.", {"name": n}))
    return out


def check_email(email: str, name: str = "") -> List[Finding]:
    out: List[Finding] = []
    e = (email or "").strip().lower()
    if not e:
        return out
    if not re.fullmatch(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", e):
        return [Finding("identity", "EMAIL_FORMAT", "medium", "Email address is not valid.", {"email": email})]
    local, domain = e.split("@")
    if domain in DISPOSABLE_DOMAINS:
        out.append(Finding("identity", "EMAIL_DISPOSABLE", "high", "Email uses a disposable-mail domain.", {"domain": domain}))
    if _looks_random(local):
        out.append(Finding("identity", "EMAIL_RANDOM", "low", "Email local-part looks randomly generated.", {"local": local}))
    return out


def check_phone(phone: str) -> List[Finding]:
    if not phone:
        return []
    p = re.sub(r"[\s\-()]", "", phone)
    p = re.sub(r"^(\+91|0091|91|0)", "", p) if len(p) > 10 else p
    if not re.fullmatch(r"[6-9]\d{9}", p):
        return [Finding("identity", "PHONE_FORMAT", "medium", "Not a valid Indian mobile number (10 digits starting 6-9).", {"phone": phone})]
    out: List[Finding] = []
    if len(set(p)) <= 3 or p in "01234567890123456789" or p in "98765432109876543210":
        out.append(Finding("identity", "PHONE_PATTERN", "medium", "Phone number is a repeated or sequential pattern.", {"phone": p}))
    return out


def check_pincode(pin: str) -> List[Finding]:
    if not pin:
        return []
    if not re.fullmatch(r"[1-9]\d{5}", pin.strip()):
        return [Finding("identity", "PINCODE_FORMAT", "low", "PIN code must be 6 digits and not start with 0.", {"pin": pin})]
    return []


def check_declared(declared: Dict[str, str]) -> List[Finding]:
    out: List[Finding] = []
    out += check_name(declared.get("name", ""))
    out += check_email(declared.get("email", ""), declared.get("name", ""))
    out += check_phone(declared.get("phone", ""))
    out += check_pincode(declared.get("pincode", ""))
    return out
