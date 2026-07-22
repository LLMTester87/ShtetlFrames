"""Characterization tests for shared slugify."""

from shtetl_core.textutil import slugify


def test_slugify_basic():
    assert slugify("Hello World") == "hello_world"


def test_slugify_strips_punctuation():
    assert slugify("Jewish Life in Munkatch!!!") == "jewish_life_in_munkatch"


def test_slugify_empty_fallback():
    assert slugify("") == "video"
    assert slugify("!!!") == "video"


def test_slugify_max_len():
    long = "a" * 200
    assert len(slugify(long, max_len=40)) == 40
