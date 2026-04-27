"""hallucination_risk_meter -- estimate LLM-answer hallucination risk.

Public surface (Python port of the JS sibling):

    from hallucination_risk_meter import score, RiskScore

* ``score(answer, context=None, citations=None, signals=None)`` -- compute
  a 0..1 risk score plus the list of triggered signals and a coarse severity.
* ``RiskScore`` -- structured result dataclass.

The score is a sum of heuristic signal weights, clamped to ``[0, 1]``. It is
not a probability -- it is a fast, dependency-free indicator suitable for
guardrail thresholds and dashboard charts.
"""

from .score import RiskScore, score

__version__ = "0.1.0"
VERSION = __version__

__all__ = [
    "VERSION",
    "RiskScore",
    "score",
]
