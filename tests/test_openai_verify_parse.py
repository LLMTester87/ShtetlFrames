"""Unit tests for OpenAI verdict parsing (no network)."""

from openai_verify import (
    _parse_verdict,
    filter_candidates_openai,
    format_verdict_notes,
    notes_openai_approved,
    notes_openai_dropped,
    notes_openai_uncertain,
    verdict_is_keep,
)
from label_feedback import apply_confidence_gate, min_keep_confidence


def test_parse_keep_json():
    v = _parse_verdict(
        '{"keep": true, "looks_jewish": true, "head_covered": true, "confidence": 0.9, '
        '"reason": "shtreimel and Orthodox dress"}'
    )
    assert v["keep"] is True
    assert v["looks_jewish"] is True
    assert v["head_covered"] is True
    assert v["confidence"] == 0.9
    assert "shtreimel" in v["reason"]
    assert verdict_is_keep(v) is True


def test_parse_drop_json():
    v = _parse_verdict(
        '{"keep": false, "looks_jewish": false, "head_covered": false, '
        '"confidence": 0.8, "reason": "business suit"}'
    )
    assert v["keep"] is False
    assert verdict_is_keep(v) is False


def test_bare_head_hard_reject():
    """Bare head / coat-only must not keep — Jewish head-covering gate."""
    v = _parse_verdict(
        '{"keep": true, "looks_jewish": true, "head_covered": false, "confidence": 0.95, '
        '"reason": "beard and payot, long coat"}'
    )
    assert v["keep"] is False
    assert v["head_covered"] is False
    assert verdict_is_keep(v) is False


def test_looks_jewish_false_rejects_even_with_hat():
    v = _parse_verdict(
        '{"keep": true, "looks_jewish": false, "head_covered": true, "confidence": 0.9, '
        '"reason": "secular man in black fedora"}'
    )
    assert v["keep"] is False
    assert v["looks_jewish"] is False
    assert verdict_is_keep(v) is False


def test_missing_head_covered_rejects():
    v = _parse_verdict(
        '{"keep": true, "confidence": 0.9, "reason": "looks Orthodox Jewish"}'
    )
    assert v["keep"] is False
    assert verdict_is_keep(v) is False


def test_skipped_does_not_pass():
    assert verdict_is_keep({"keep": True, "skipped": True}) is False
    assert verdict_is_keep({"keep": False, "skipped": True}) is False


def test_notes_gate():
    assert notes_openai_approved("openai:keep conf=0.90 Orthodox dress")
    assert not notes_openai_approved("openai:drop conf=0.80 suit")
    assert not notes_openai_approved("")
    assert not notes_openai_approved("human note only")
    assert notes_openai_dropped("openai:drop conf=0.80 suit")
    assert not notes_openai_dropped("openai:keep conf=0.90 Orthodox dress")
    assert notes_openai_uncertain("openai:uncertain conf=0.40 low_conf")
    assert not notes_openai_approved("openai:uncertain conf=0.40 low_conf")
    # Open VLM uses the same Review gates with a vlm: prefix.
    assert notes_openai_approved("vlm:keep conf=0.91 shtreimel")
    assert notes_openai_dropped("vlm:drop conf=0.88 fedora only")
    assert notes_openai_uncertain("vlm:uncertain conf=0.50 maybe")


def test_confidence_gate_low_keep(monkeypatch):
    monkeypatch.setenv("OPENAI_MIN_KEEP_CONF", "0.70")
    v = apply_confidence_gate(
        {
            "keep": True,
            "looks_jewish": True,
            "head_covered": True,
            "confidence": 0.4,
            "reason": "maybe",
            "skipped": False,
        }
    )
    assert v["uncertain"] is True
    assert v["keep"] is False
    assert verdict_is_keep(v) is False
    note = format_verdict_notes(v)
    assert note.startswith("openai:uncertain")


def test_confidence_gate_high_keep(monkeypatch):
    monkeypatch.setenv("OPENAI_MIN_KEEP_CONF", "0.70")
    v = apply_confidence_gate(
        {
            "keep": True,
            "looks_jewish": True,
            "head_covered": True,
            "confidence": 0.9,
            "reason": "clear",
            "skipped": False,
        }
    )
    assert not v.get("uncertain")
    assert v["keep"] is True
    assert verdict_is_keep(v) is True
    note = format_verdict_notes(v)
    assert note.startswith("openai:keep")
    assert "head=yes" in note
    assert "jewish=yes" in note


def test_filter_noop_when_disabled(monkeypatch):
    monkeypatch.setattr("openai_verify.openai_verify_enabled", lambda: False)
    rows = [{"image_url": "https://example.com/a.jpg", "peak_score": 0.2}]
    out = filter_candidates_openai(rows)
    assert len(out) == 1


def test_min_keep_confidence_bounds(monkeypatch):
    monkeypatch.setenv("OPENAI_MIN_KEEP_CONF", "1.5")
    assert min_keep_confidence() == 0.95
    monkeypatch.setenv("OPENAI_MIN_KEEP_CONF", "-1")
    assert min_keep_confidence() == 0.0


def test_format_notes_uses_vlm_provider(monkeypatch):
    monkeypatch.setenv("VERIFY_BACKEND", "open_vlm")
    note = format_verdict_notes(
        {
            "keep": True,
            "looks_jewish": True,
            "head_covered": True,
            "confidence": 0.91,
            "reason": "shtreimel",
            "provider": "vlm",
        }
    )
    assert note.startswith("vlm:keep")


def test_open_vlm_enabled_needs_base_url(monkeypatch):
    from openai_verify import (
        openai_verify_enabled,
        open_vlm_runs_on_pod,
        open_vlm_url_is_local,
    )

    # Avoid load_env() overwriting monkeypatched vars from .env / SQLite.
    monkeypatch.setattr("openai_verify._load_env", lambda: None)
    monkeypatch.setenv("OPENAI_VERIFY", "1")
    monkeypatch.setenv("VERIFY_BACKEND", "open_vlm")
    monkeypatch.setattr("openai_verify._disabled_reason", None)
    # Empty / pod → on-GPU Ollama (configured).
    monkeypatch.setenv("OPEN_VLM_BASE_URL", "pod")
    assert openai_verify_enabled() is True
    assert open_vlm_runs_on_pod() is True
    monkeypatch.setenv("OPEN_VLM_BASE_URL", "https://openrouter.ai/api/v1")
    assert openai_verify_enabled() is True
    assert open_vlm_runs_on_pod() is False
    assert open_vlm_url_is_local("http://127.0.0.1:11434/v1") is True
    assert open_vlm_url_is_local("https://openrouter.ai/api/v1") is False


def test_cascade_skips_openai_on_vlm_drop(monkeypatch):
    from openai_verify import verify_still

    monkeypatch.setattr("openai_verify._load_env", lambda: None)
    monkeypatch.setattr("openai_verify._disabled_reason", None)
    monkeypatch.setenv("OPENAI_VERIFY", "1")
    monkeypatch.setenv("VERIFY_BACKEND", "ollama_then_openai")
    monkeypatch.setenv("OPEN_VLM_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls: list[str] = []

    def fake_one(backend, **kwargs):
        calls.append(backend)
        if backend == "open_vlm":
            return {
                "keep": False,
                "looks_jewish": False,
                "head_covered": False,
                "confidence": 0.9,
                "reason": "secular fedora",
                "skipped": False,
                "provider": "vlm",
            }
        raise AssertionError("OpenAI must not run after VLM drop")

    monkeypatch.setattr("openai_verify._verify_still_one", fake_one)
    v = verify_still(image_b64="aGVsbG8=")  # unused; fake_one short-circuits
    assert calls == ["open_vlm"]
    assert v["keep"] is False
    assert v["provider"] == "vlm"


def test_cascade_calls_openai_on_vlm_keep(monkeypatch):
    from openai_verify import verify_still

    monkeypatch.setattr("openai_verify._load_env", lambda: None)
    monkeypatch.setattr("openai_verify._disabled_reason", None)
    monkeypatch.setenv("OPENAI_VERIFY", "1")
    monkeypatch.setenv("VERIFY_BACKEND", "ollama_then_openai")
    monkeypatch.setenv("OPEN_VLM_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls: list[str] = []

    def fake_one(backend, **kwargs):
        calls.append(backend)
        if backend == "open_vlm":
            return {
                "keep": True,
                "looks_jewish": True,
                "head_covered": True,
                "confidence": 0.88,
                "reason": "shtreimel",
                "skipped": False,
                "provider": "vlm",
            }
        return {
            "keep": True,
            "looks_jewish": True,
            "head_covered": True,
            "confidence": 0.95,
            "reason": "confirmed",
            "skipped": False,
            "provider": "openai",
        }

    monkeypatch.setattr("openai_verify._verify_still_one", fake_one)
    v = verify_still(image_b64="aGVsbG8=")
    assert calls == ["open_vlm", "openai"]
    assert v["keep"] is True
    assert v["provider"] == "openai"
    assert "after_vlm" in (v.get("reason") or "")
