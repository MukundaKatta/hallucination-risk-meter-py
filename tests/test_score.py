"""Tests for ``hallucination_risk_meter.score``."""

from __future__ import annotations

import pytest

from hallucination_risk_meter import RiskScore, score


def test_low_risk_short_answer_with_context_and_citations():
    answer = "Plants make energy from light [1]."
    ctx = "Plants make energy from light through photosynthesis."
    r = score(answer, context=ctx, citations=[{"id": "1"}])
    assert isinstance(r, RiskScore)
    assert r.severity == "low"
    assert r.score < 0.34


def test_uncertainty_language_is_detected():
    r = score("I think the sky might be blue.")
    assert "uncertainty_language" in r.signals


def test_unsourced_specifics_fire_for_uncited_sentences():
    answer = "The company earned $42 million in 2023."
    r = score(answer)
    assert "unsourced_specifics" in r.signals


def test_specifics_inside_a_cited_sentence_dont_fire():
    answer = "The company earned $42 million in 2023 [1]."
    r = score(answer, citations=[{"id": "1"}])
    assert "unsourced_specifics" not in r.signals


def test_confident_overreach_fires_without_context_support():
    r = score("This is definitely the only correct answer.")
    assert "confident_overreach" in r.signals


def test_confident_phrasing_with_supporting_context_does_not_fire():
    answer = "The boiling point is always 100 degrees Celsius for water."
    ctx = "The boiling point of water is 100 degrees Celsius at sea level."
    r = score(answer, context=ctx, citations=[{"id": "1"}])
    assert "confident_overreach" not in r.signals


def test_unsourced_named_entities_flag_uncited_people():
    # "Vladimir Pomerantsev" not in context, no citation -> flagged.
    r = score("Vladimir Pomerantsev invented the modern jet engine.")
    assert "unsourced_named_entities" in r.signals


def test_named_entity_present_in_context_does_not_flag():
    answer = "Marie Curie discovered radium."
    ctx = "Marie Curie was a Polish-French physicist who discovered polonium and radium."
    # Citations not provided -> no_citations may fire, but entity should not.
    r = score(answer, context=ctx, citations=[{"id": "1"}])
    assert "unsourced_named_entities" not in r.signals


def test_no_citations_fires_for_factual_answer():
    r = score("The Eiffel Tower is in Paris.")
    assert "no_citations" in r.signals


def test_no_citations_does_not_fire_for_obvious_opinion():
    # No specifics, no entities -> _looks_factual is False -> signal silent.
    r = score("hi how are you")
    assert "no_citations" not in r.signals


def test_length_disproportionate_fires_when_answer_dwarfs_context():
    long_answer = " ".join(["claim"] * 200)
    short_ctx = "tiny"
    r = score(long_answer, context=short_ctx)
    assert "length_disproportionate" in r.signals


def test_severity_buckets_match_score_thresholds():
    # Force high severity by stacking signals.
    answer = (
        "I think Vladimir Pomerantsev definitely earned $42 million in 1987 "
        "and Marie Curie absolutely guaranteed it."
    )
    r = score(answer)
    assert r.severity in {"medium", "high"}
    assert r.score > 0.33


def test_signals_argument_can_inject_extra_signal():
    r = score("hi", signals=["custom_signal_from_classifier"])
    assert "custom_signal_from_classifier" in r.signals


def test_signals_argument_does_not_duplicate_detected_signals():
    r = score("I think so.", signals=["uncertainty_language"])
    assert r.signals.count("uncertainty_language") == 1


def test_score_is_clamped_to_one():
    # Stack many signals -- raw sum could exceed 1.
    answer = (
        "I think Vladimir Pomerantsev definitely earned $42 million in 1987. "
        "Marie Curie absolutely guaranteed 100% certainty in 1898."
    )
    r = score(
        answer,
        signals=["custom_a", "custom_b"],
    )
    assert 0.0 <= r.score <= 1.0


def test_non_string_answer_raises():
    with pytest.raises(TypeError):
        score(123)  # type: ignore[arg-type]


def test_none_answer_treated_as_empty():
    r = score(None)  # type: ignore[arg-type]
    assert r.score == 0.0
    assert r.severity == "low"
