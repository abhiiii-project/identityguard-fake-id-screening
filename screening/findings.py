"""Common data structures and the explainable risk-scoring logic."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List

SEVERITY_WEIGHT = {"info": 0.0, "low": 0.08, "medium": 0.22, "high": 0.45, "critical": 0.75}


@dataclass
class Finding:
    category: str          # validation | forensics | identity | consistency | duplicate | ml
    code: str              # stable machine-readable id
    severity: str          # info | low | medium | high | critical
    message: str           # human readable explanation
    evidence: Dict = field(default_factory=dict)

    @property
    def weight(self) -> float:
        return SEVERITY_WEIGHT[self.severity]

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["weight"] = self.weight
        return d


def risk_score(findings: List[Finding], ml_probability: float | None = None) -> float:
    """Combine findings using a noisy-OR so that independent signals reinforce
    each other but the score always stays within 0..100.

    P(fraud) = 1 - prod(1 - w_i)
    The optional ML probability is treated as one more independent signal,
    scaled down so that it cannot single-handedly reject an application.
    """
    survive = 1.0
    for f in findings:
        survive *= 1.0 - f.weight
    if ml_probability is not None:
        survive *= 1.0 - min(0.5, max(0.0, ml_probability) * 0.5)
    return round((1.0 - survive) * 100.0, 1)


def decide(score: float, findings: List[Finding]) -> str:
    """Map score to an action. A single critical finding always forces REJECT."""
    if any(f.severity == "critical" for f in findings) or score >= 65:
        return "REJECT"
    if score >= 30:
        return "MANUAL_REVIEW"
    return "APPROVE"
