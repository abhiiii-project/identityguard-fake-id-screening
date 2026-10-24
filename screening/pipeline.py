"""Orchestrates every check and produces an explainable report."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import forensics, identity, ocr, validators
from .findings import Finding, decide, risk_score
from .ml import TamperModel
from .store import HistoryStore


@dataclass
class ScreeningReport:
    decision: str
    risk_score: float
    findings: List[Finding]
    ml_probability: Optional[float] = None
    extracted: Dict = field(default_factory=dict)
    checks_run: List[str] = field(default_factory=list)
    elapsed_ms: int = 0

    def to_dict(self) -> Dict:
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        fs = sorted(self.findings, key=lambda f: order[f.severity])
        by_sev: Dict[str, int] = {}
        for f in fs:
            by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
        return {"decision": self.decision, "risk_score": self.risk_score,
                "ml_tamper_probability": self.ml_probability, "summary": by_sev,
                "findings": [f.to_dict() for f in fs], "extracted_fields": self.extracted,
                "checks_run": self.checks_run, "elapsed_ms": self.elapsed_ms}


class ScreeningPipeline:
    def __init__(self, store: Optional[HistoryStore] = None, model: Optional[TamperModel] = None):
        self.store = store
        self.model = model or TamperModel()

    def screen(self, declared: Dict[str, str], document: Optional[bytes] = None,
               mrz: Optional[tuple] = None, record: bool = True, use_ocr: bool = True) -> ScreeningReport:
        t0 = time.time()
        findings: List[Finding] = []
        checks: List[str] = []
        extracted: Dict = {}
        feats: Dict = {}
        dhash = None
        ml_p = None

        # 1. declared data -------------------------------------------------
        checks.append("declared_data_validation")
        surname = (declared.get("name", "").split() or [""])[-1]
        if declared.get("aadhaar"):
            findings += validators.validate_aadhaar(declared["aadhaar"])
        if declared.get("pan"):
            findings += validators.validate_pan(declared["pan"], surname)
        findings += validators.validate_dates(declared.get("dob"), declared.get("issue_date"), declared.get("expiry_date"))
        if not declared.get("aadhaar") and not declared.get("pan") and not mrz:
            findings.append(Finding("validation", "NO_ID_NUMBER", "medium", "No identity number was supplied to verify.", {}))

        checks.append("synthetic_identity_signals")
        findings += identity.check_declared(declared)

        # 2. passport MRZ --------------------------------------------------
        if mrz:
            checks.append("passport_mrz_checksums")
            findings += validators.validate_mrz_td3(*mrz)

        # 3. document image -----------------------------------------------
        if document:
            checks.append("image_forensics")
            try:
                feats, ffind, extra = forensics.analyse_image(document)
                findings += ffind
                dhash = extra["dhash"]
                if self.model.available:
                    checks.append("ml_tamper_classifier")
                    ml_p = self.model.predict(feats)
                    if ml_p is not None and ml_p > 0.8:
                        findings.append(Finding("ml", "ML_TAMPER", "medium",
                                                f"The learned tamper classifier rates this image {ml_p:.0%} likely to be manipulated.",
                                                {"probability": round(ml_p, 3)}))
            except Exception as exc:
                findings.append(Finding("forensics", "IMAGE_UNREADABLE", "high",
                                        "The uploaded file could not be read as an image.", {"error": str(exc)[:120]}))
            if use_ocr:
                checks.append("ocr_cross_check")
                text = ocr.ocr_text(document)
                extracted = ocr.extract_fields(text)
                if text.strip():
                    findings += ocr.cross_check(extracted, declared)
                else:
                    findings.append(Finding("consistency", "OCR_EMPTY", "low",
                                            "No readable text could be extracted from the document.", {}))
        else:
            findings.append(Finding("forensics", "NO_DOCUMENT", "info", "No document image supplied; only data checks were run.", {}))

        # 4. history -------------------------------------------------------
        if self.store:
            checks.append("duplicate_and_velocity")
            findings += self.store.check(declared, dhash)

        # drop info-only noise from scoring but keep in report
        score = risk_score(findings, ml_p)
        decision = decide(score, findings)
        if self.store and record:
            self.store.record(declared, dhash, score, decision)
        return ScreeningReport(decision, score, findings, ml_p, extracted, checks, int((time.time() - t0) * 1000))
