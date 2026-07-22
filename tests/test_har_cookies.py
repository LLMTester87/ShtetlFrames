"""HAR → Netscape YouTube cookie import."""

from yt_cookies import cookies_look_valid, har_to_netscape


def _sample_har() -> dict:
    return {
        "log": {
            "entries": [
                {
                    "request": {
                        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        "cookies": [
                            {"name": "VISITOR_INFO1_LIVE", "value": "abc", "domain": ".youtube.com", "path": "/"},
                            {"name": "PREF", "value": "f1=1", "domain": ".youtube.com", "path": "/", "secure": True},
                        ],
                        "headers": [
                            {
                                "name": "Cookie",
                                "value": "LOGIN_INFO=secret; CONSENT=YES+",
                            }
                        ],
                    },
                    "response": {
                        "cookies": [
                            {
                                "name": "SID",
                                "value": "sidval",
                                "domain": ".google.com",
                                "path": "/",
                                "secure": True,
                                "expires": "1893456000",
                            }
                        ],
                        "headers": [
                            {
                                "name": "set-cookie",
                                "value": "SAPISID=sapival; Domain=.google.com; Path=/; Secure",
                            }
                        ],
                    },
                },
                {
                    "request": {
                        "url": "https://example.com/",
                        "cookies": [{"name": "noise", "value": "1", "domain": ".example.com"}],
                        "headers": [],
                    },
                    "response": {"cookies": [], "headers": []},
                },
            ]
        }
    }


def test_har_extracts_youtube_and_google_cookies():
    text, n = har_to_netscape(_sample_har())
    assert n >= 4
    assert cookies_look_valid(text)
    assert "VISITOR_INFO1_LIVE" in text
    assert "LOGIN_INFO" in text
    assert "SID" in text
    assert "SAPISID" in text
    assert "noise" not in text
    assert ".youtube.com" in text
    assert ".google.com" in text


def test_har_from_json_string():
    import json

    text, n = har_to_netscape(json.dumps(_sample_har()))
    assert n >= 4
    assert cookies_look_valid(text)


def test_empty_har_returns_zero():
    text, n = har_to_netscape({"log": {"entries": []}})
    assert text == ""
    assert n == 0
