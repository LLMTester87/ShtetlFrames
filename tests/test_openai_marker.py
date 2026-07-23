"""Orthodox marker gate for hardened vision verify."""

from openai_verify import _normalize_marker, _parse_verdict, verdict_is_keep


def test_normalize_marker_aliases():
    assert _normalize_marker("shtreimel") == "shtreimel"
    assert _normalize_marker("Yarmulke") == "kippah"
    assert _normalize_marker("none") == "none"
    assert _normalize_marker("fur streimel hat") == "shtreimel"


def test_parse_requires_marker_for_keep():
    v = _parse_verdict(
        '{"keep": true, "looks_jewish": true, "head_covered": true, '
        '"marker": "none", "confidence": 0.95, "reason": "dark hat"}'
    )
    assert v["keep"] is False
    assert not verdict_is_keep(v)


def test_parse_shtreimel_keep():
    v = _parse_verdict(
        '{"keep": true, "looks_jewish": true, "head_covered": true, '
        '"marker": "shtreimel", "confidence": 0.9, "reason": "Hasidic man shtreimel"}'
    )
    assert v["keep"] is True
    assert verdict_is_keep(v)


def test_infer_marker_from_reason_when_omitted():
    v = _parse_verdict(
        '{"keep": true, "looks_jewish": true, "head_covered": true, '
        '"confidence": 0.9, "reason": "clear shtreimel on bearded man"}'
    )
    assert v["marker"] == "shtreimel"
    assert verdict_is_keep(v)
