# hallucination-risk-meter

[![PyPI](https://img.shields.io/pypi/v/hallucination-risk-meter.svg)](https://pypi.org/project/hallucination-risk-meter/)
[![Python](https://img.shields.io/pypi/pyversions/hallucination-risk-meter.svg)](https://pypi.org/project/hallucination-risk-meter/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Estimate hallucination risk in LLM answers** from uncertainty language, unsupported specifics, citations, and context coverage. Zero runtime dependencies.

Python port of [@mukundakatta/hallucination-risk-meter](https://github.com/MukundaKatta/hallucination-risk-meter). The JS sibling has the original API; this README sticks to the Python surface.

## Install

```bash
pip install hallucination-risk-meter
```

## Usage

```python
from hallucination_risk_meter import score

answer = "The company earned $42 million in 2023."
ctx    = "Acme Inc. reported revenue of $40-44 million for fiscal year 2023."

r = score(answer, context=ctx, citations=[{"id": "1"}])

r.score      # float in [0.0, 1.0]
r.signals    # list[str] -- triggered heuristics
r.severity   # 'low' | 'medium' | 'high'
```

## Signals

Each triggered signal contributes a fixed weight to the final score (clamped to `[0, 1]`):

| Signal                       | Weight | Triggers when...                                                                                   |
|------------------------------|--------|----------------------------------------------------------------------------------------------------|
| `uncertainty_language`       | 0.10   | Hedge phrases like "I think", "may", "possibly", "not sure" appear.                                 |
| `unsourced_specifics`        | 0.30   | Specific numbers, percentages, currency, years, or dates appear in an uncited sentence.             |
| `confident_overreach`        | 0.20   | "definitely" / "guaranteed" / "always" appear without supporting context overlap.                   |
| `unsourced_named_entities`   | 0.20   | A capitalized multi-word entity appears uncited and absent from the context.                        |
| `length_disproportionate`    | 0.20   | The answer is roughly 3x longer than the supplied context.                                          |
| `no_citations`               | 0.15   | The answer asserts something factual-looking but no citations were supplied.                        |

## Severity buckets

* `score < 0.34`  -> `"low"`
* `0.34 - 0.66`   -> `"medium"`
* `score >= 0.67` -> `"high"`

## Custom signals

Inject precomputed signals from your own classifier:

```python
score("...", signals=["nli_contradiction", "self_consistency_drop"])
```

Names not in the built-in weight table contribute `0.0` but still appear in `r.signals` for downstream logging.

## API differences from the JS sibling

* Returns a `RiskScore` dataclass with `score`, `signals`, and `severity` instead of the JS `{risk, reasons, likelyHallucinated}` object.
* Adds the `signals=` parameter for injecting upstream-detector signals.
* Adds `severity` bucketing (low/medium/high) for convenient guardrail thresholds.

See the JS sibling's [README](https://github.com/MukundaKatta/hallucination-risk-meter) for the full design notes.
