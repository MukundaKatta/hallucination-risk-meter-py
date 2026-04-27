"""Heuristic hallucination-risk scoring.

The score is a sum of weighted signals, clamped to [0.0, 1.0]:

| Signal                              | Weight | Trigger                                                                                                           |
|-------------------------------------|--------|-------------------------------------------------------------------------------------------------------------------|
| ``uncertainty_language``            | 0.10   | Hedge phrases like "I think", "may", "possibly", "not sure" appear in the answer.                                 |
| ``unsourced_specifics``             | 0.30   | Specific numbers, percentages, currency, years, or dates appear in an uncited sentence.                           |
| ``confident_overreach``             | 0.20   | Confident phrasing ("definitely", "guaranteed", "always") with no supporting context.                             |
| ``unsourced_named_entities``        | 0.20   | A capitalized multi-word entity (likely person/org/place) appears with no citation and is not echoed in context.  |
| ``length_disproportionate``         | 0.20   | The answer is much longer than the supplied context, suggesting fabrication beyond the source material.           |
| ``no_citations``                    | 0.15   | The answer asserts factual content but no citations were supplied alongside it.                                   |

A coarse severity is derived from the final score:

* ``score < 0.34`` -> ``"low"``
* ``0.34 <= score < 0.67`` -> ``"medium"``
* ``score >= 0.67`` -> ``"high"``

Callers can pre-compute their own signals and pass them via ``signals=`` to
bypass detection (or add custom ones); the score is simply summed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional, Sequence, Set

# Signal weights -- exposed so callers can reason about the score.
SIGNAL_WEIGHTS: Mapping[str, float] = {
    "uncertainty_language": 0.10,
    "unsourced_specifics": 0.30,
    "confident_overreach": 0.20,
    "unsourced_named_entities": 0.20,
    "length_disproportionate": 0.20,
    "no_citations": 0.15,
}

_UNCERTAINTY_RE = re.compile(
    r"\b(?:i\s+think|i\s+believe|i'?m\s+not\s+sure|might|maybe|perhaps|possibly|"
    r"probably|seems\s+to|appears\s+to|may\s+(?:be|have)|could\s+be|"
    r"i\s+guess|not\s+entirely\s+sure)\b",
    re.IGNORECASE,
)

_OVERCONFIDENT_RE = re.compile(
    r"\b(?:definitely|guaranteed|always|never|certainly|undoubtedly|"
    r"absolutely|without\s+doubt|100\s*%)\b",
    re.IGNORECASE,
)

# Specifics: percentages, money, plain integers >= 4 digits (years/big nums),
# dates of various forms. Loose on purpose -- false positives here are usually
# still worth flagging when uncited.
_SPECIFICS_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?\s*%|"  # percent
    r"\$\s?\d[\d,]*(?:\.\d+)?|"  # currency
    r"\d{4,}|"  # 4+ digit number / year
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"  # numeric date
    r"(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{1,2}(?:,\s*\d{4})?)\b",
    re.IGNORECASE,
)

# Capitalized multi-word entities, e.g. "Marie Curie", "United Nations".
# Very simple heuristic; intentionally avoids NER deps.
_NAMED_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b")

# Citation marker used by the spec: `[1]`, `[42]`, `[id:abc]`.
_CITATION_RE = re.compile(r"\[(?:\d+|id:[^\]\s]+)\]", re.IGNORECASE)

_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class RiskScore:
    """Structured result returned by :func:`score`.

    Attributes:
        score: Total risk in ``[0.0, 1.0]`` (clamped sum of triggered signal weights).
        signals: List of triggered signal names, in detection order.
        severity: Coarse bucket: ``"low"`` / ``"medium"`` / ``"high"``.
    """

    score: float
    signals: List[str] = field(default_factory=list)
    severity: str = "low"


def score(
    answer: str,
    context: Optional[str] = None,
    citations: Optional[Sequence[object]] = None,
    signals: Optional[Iterable[str]] = None,
) -> RiskScore:
    """Compute hallucination risk for ``answer``.

    Args:
        answer: The model's answer text.
        context: Optional source / RAG context the answer was supposed to draw on.
            When provided, "unsourced" checks consult it; absent, every claim
            is treated as unsourced (no support evidence available).
        citations: Optional list of citation objects (any iterable). Just
            counted for the ``no_citations`` signal; structure is not parsed.
        signals: Optional pre-computed signals to add to the score (bypassing
            detection for those names). Unknown signal names are still
            recorded but contribute zero weight.

    Returns:
        A :class:`RiskScore` with the total, the triggered signal list, and
        a coarse severity bucket.
    """
    if answer is None:
        answer = ""
    if not isinstance(answer, str):
        raise TypeError("score: answer must be a string")
    ctx = context if isinstance(context, str) else ""
    citation_count = _count(citations)
    fired: List[str] = []

    # --- detection ---
    if _UNCERTAINTY_RE.search(answer):
        fired.append("uncertainty_language")

    # Specifics that appear in a sentence with no citation marker.
    for sentence in _split_sentences(answer):
        if _CITATION_RE.search(sentence):
            continue
        if _SPECIFICS_RE.search(sentence):
            fired.append("unsourced_specifics")
            break

    if _OVERCONFIDENT_RE.search(answer) and not _confident_supported(answer, ctx):
        fired.append("confident_overreach")

    if _has_unsourced_entities(answer, ctx):
        fired.append("unsourced_named_entities")

    if _length_disproportionate(answer, ctx):
        fired.append("length_disproportionate")

    if citation_count == 0 and _looks_factual(answer):
        fired.append("no_citations")

    # --- merge in caller-supplied signals (e.g. from upstream classifiers) ---
    if signals:
        for s in signals:
            if isinstance(s, str) and s not in fired:
                fired.append(s)

    # --- score ---
    raw = sum(SIGNAL_WEIGHTS.get(s, 0.0) for s in fired)
    total = max(0.0, min(1.0, raw))
    # Round to 2 decimal places to mirror the JS sibling's reporting convention
    # (one decimal of noise is plenty given how heuristic this is).
    total = round(total * 100) / 100

    return RiskScore(score=total, signals=fired, severity=_severity(total))


def _count(it: Optional[Sequence[object]]) -> int:
    if it is None:
        return 0
    try:
        return len(it)  # type: ignore[arg-type]
    except TypeError:
        # Generic iterable -- materialize once.
        return sum(1 for _ in it)


def _split_sentences(text: str) -> List[str]:
    out = []
    for m in _SENTENCE_RE.finditer(text):
        s = m.group(0).strip()
        if s:
            out.append(s)
    return out


def _confident_supported(answer: str, context: str) -> bool:
    """A confident claim is "supported" only if at least 30% of its tokens
    are echoed in the context. Without context, we assume unsupported."""
    if not context:
        return False
    a_tokens = set(_TOKEN_RE.findall(answer.lower()))
    c_tokens = set(_TOKEN_RE.findall(context.lower()))
    if not a_tokens:
        return True
    overlap = len(a_tokens & c_tokens) / len(a_tokens)
    return overlap >= 0.3


def _has_unsourced_entities(answer: str, context: str) -> bool:
    """True if a likely person/org/place name appears in the answer with no
    citation marker AND is not echoed in the supplied context."""
    cited_sentences: Set[str] = {
        s for s in _split_sentences(answer) if _CITATION_RE.search(s)
    }
    ctx_lower = context.lower()
    for sentence in _split_sentences(answer):
        if sentence in cited_sentences:
            continue
        for m in _NAMED_ENTITY_RE.finditer(sentence):
            entity = m.group(1)
            # Skip sentence-initial "Some Capitalized Words" -- usually generic.
            if sentence.startswith(entity) and len(entity.split()) < 2:
                continue
            if entity.lower() in ctx_lower:
                continue
            return True
    return False


def _length_disproportionate(answer: str, context: str) -> bool:
    """Answer dwarfs context (3x or more words) -- room for fabrication."""
    if not context:
        # Without context we can't measure the ratio; don't fire.
        return False
    a_len = len(answer.split())
    c_len = len(context.split())
    if a_len < 50 or c_len == 0:
        return False
    return a_len / c_len >= 3


def _looks_factual(answer: str) -> bool:
    """Cheap heuristic: text contains a specific number, date, or named
    entity -> looks like a factual assertion worth citing."""
    if _SPECIFICS_RE.search(answer):
        return True
    if _NAMED_ENTITY_RE.search(answer):
        return True
    return False


def _severity(score_val: float) -> str:
    if score_val >= 0.67:
        return "high"
    if score_val >= 0.34:
        return "medium"
    return "low"
