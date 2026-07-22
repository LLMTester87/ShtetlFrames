"""Tests for Keep/Pass vs OpenAI agreement bucketing."""

from label_feedback import _bucket, _openai_tag, apply_confidence_gate


def test_openai_tag():
    assert _openai_tag("openai:keep conf=0.9 x") == "keep"
    assert _openai_tag("note\nopenai:drop conf=0.2 y") == "drop"
    assert _openai_tag("openai:uncertain conf=0.3 z") == "uncertain"
    assert _openai_tag("") == ""


def test_buckets():
    assert _bucket("accept", "keep") == "agree_keep"
    assert _bucket("reject", "drop") == "agree_drop"
    assert _bucket("reject", "keep") == "false_keep"
    assert _bucket("accept", "drop") == "false_drop"
    assert _bucket("", "keep") == "pending"
    assert _bucket("accept", "") == "labeled_no_openai"


def test_gate_skips_passthrough():
    v = apply_confidence_gate(
        {"keep": True, "confidence": 0.1, "reason": "off", "skipped": True}
    )
    assert v["skipped"] is True
    assert v["keep"] is True
