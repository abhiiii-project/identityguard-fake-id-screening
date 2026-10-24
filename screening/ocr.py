"""OCR + field extraction, used to cross-check the document against what the
applicant typed into the form."""
from __future__ import annotations

import io
import re
from typing import Dict, List, Optional

from PIL import Image

from .findings import Finding
from .validators import parse_date

try:
    import pytesseract
    _OK = True
except Exception:  # pragma: no cover
    _OK = False


def ocr_text(data: bytes) -> str:
    if not _OK:
        return ""
    try:
        img = Image.open(io.BytesIO(data)).convert("L")
        if min(img.size) < 900:
            s = 900 / min(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
        return pytesseract.image_to_string(img, config="--psm 6")
    except Exception:
        return ""


def extract_fields(text: str) -> Dict[str, Optional[str]]:
    flat = re.sub(r"[ \t]+", " ", text)
    aadhaar = re.search(r"\b(\d{4}\s?\d{4}\s?\d{4})\b", flat)
    pan = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", flat)
    dob = re.search(r"(?:DOB|Date of Birth|Birth)[^0-9A-Za-z]{0,6}(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{4})", flat, re.I)
    if not dob:
        dob = re.search(r"\b(\d{2}[/\-.]\d{2}[/\-.]\d{4})\b", flat)
    name = re.search(r"(?:Name)[ :]*([A-Za-z][A-Za-z .]{2,40})", flat)
    return {
        "aadhaar": re.sub(r"\s", "", aadhaar.group(1)) if aadhaar else None,
        "pan": pan.group(1) if pan else None,
        "dob": dob.group(1) if dob else None,
        "name": name.group(1).strip() if name else None,
    }


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def name_similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    ta, tb = sorted(_norm_name(a).split()), sorted(_norm_name(b).split())
    return SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()


def cross_check(extracted: Dict[str, Optional[str]], declared: Dict[str, str]) -> List[Finding]:
    """Compare OCR output with the applicant's declared data."""
    out: List[Finding] = []
    if declared.get("name") and extracted.get("name"):
        sim = name_similarity(declared["name"], extracted["name"])
        if sim < 0.6:
            out.append(Finding("consistency", "NAME_MISMATCH", "high",
                               "Name on the document does not match the declared name.",
                               {"declared": declared["name"], "document": extracted["name"], "similarity": round(sim, 2)}))
        elif sim < 0.85:
            out.append(Finding("consistency", "NAME_PARTIAL_MATCH", "low",
                               "Name on the document only partly matches the declared name.",
                               {"declared": declared["name"], "document": extracted["name"], "similarity": round(sim, 2)}))
    if declared.get("dob") and extracted.get("dob"):
        d1, d2 = parse_date(declared["dob"]), parse_date(extracted["dob"])
        if d1 and d2 and d1 != d2:
            out.append(Finding("consistency", "DOB_MISMATCH", "high",
                               "Date of birth on the document differs from the declared date of birth.",
                               {"declared": declared["dob"], "document": extracted["dob"]}))
    for key, label in (("aadhaar", "Aadhaar"), ("pan", "PAN")):
        d = re.sub(r"\s", "", declared.get(key, "") or "").upper()
        e = (extracted.get(key) or "").upper()
        if d and e and d != e:
            # one character off is most likely an OCR slip, not a different number
            near = len(d) == len(e) and sum(a != b for a, b in zip(d, e)) <= 1
            out.append(Finding("consistency", f"{key.upper()}_MISMATCH", "low" if near else "critical",
                               f"{label} number on the document {'may differ slightly from' if near else 'differs from'} the declared number.",
                               {"declared": d, "document": e, "probable_ocr_error": near}))
    return out
