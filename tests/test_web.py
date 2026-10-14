"""Web search parsing, page text extraction, SSRF guard (no network except the optional live test)."""
import base64, os

import pytest

import web

DDG = """<html><body>
<div class="result results_links results_links_deep web-result">
 <h2 class="result__title"><a rel="nofollow" class="result__a"
   href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage%3Fa%3D1%26b%3D2&amp;rut=abc">Example <b>Page</b></a></h2>
 <a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">The <b>best</b> example   page.</a>
</div>
<div class="result"><h2><a class="result__a" href="https://duckduckgo.com/y.js?ad_domain=ads.com">Ad</a></h2></div>
<div class="result"><h2><a class="result__a" href="https://second.org/">Second &amp; more</a></h2>
 <div class="result__snippet">Snippet two</div></div>
</body></html>"""

BING_TARGET = "https://bing-result.net/a?b=1"
BING = f"""<ol id="b_results">
<li class="b_algo"><div class="b_tpcn"><a href="https://bing-result.net/" class="tilk"></a></div>
 <h2><a href="https://www.bing.com/ck/a?!&amp;&amp;p=xyz&amp;u=a1{base64.urlsafe_b64encode(BING_TARGET.encode()).decode().rstrip('=')}&amp;ntb=1">Bing title</a></h2>
 <div class="b_caption"><p>Bing <strong>snippet</strong></p></div></li>
<li class="b_algo"><h2><a href="https://other.io/x">Other</a></h2><ul><li>deep</li></ul><p>Other snippet</p></li>
</ol>"""

PAGE = """<!doctype html><html><head><title> My Article </title><style>.x{color:red}</style>
<script>var secret = 1;</script></head><body>
<nav><a href="/">Home</a> <a href="/about">About</a></nav>
<header>Site banner</header>
<h1>Main title</h1><p>First   paragraph
 with <a href="#">a link</a>.</p>
<svg><title>icon</title></svg>
<h2>Section</h2><ul><li>one</li><li>two</li></ul>
<footer>Copyright footer</footer></body></html>"""


def test_ddg_parsing_and_redirect_decoding():
    r = web.parse_results(DDG, "ddg")
    assert r == [{"title": "Example Page", "url": "https://example.com/page?a=1&b=2", "snippet": "The best example page."},
                 {"title": "Second & more", "url": "https://second.org/", "snippet": "Snippet two"}]


def test_bing_parsing():
    r = web.parse_results(BING, "bing")
    assert r[0]["url"] == BING_TARGET and r[0]["snippet"] == "Bing snippet"
    assert r[1] == {"title": "Other", "url": "https://other.io/x", "snippet": "Other snippet"}


def test_decode_link():
    assert web.decode_link("//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.com%2F%C3%A9t%C3%A9&rut=1") == "https://a.com/été"
    assert web.decode_link("https://plain.com/x") == "https://plain.com/x"


def test_html_text():
    title, text = web.html_text(PAGE)
    assert title == "My Article"
    assert text == "# Main title\n\nFirst paragraph with a link.\n\n## Section\n\n- one\n- two"
    for gone in ("secret", "color", "Home", "banner", "footer", "icon"):
        assert gone not in text


def test_html_text_prefers_main():
    body = "<main><p>" + "real content " * 60 + "</p></main><div>sidebar junk</div>"
    assert "sidebar" not in web.html_text(body)[1]


@pytest.mark.parametrize("url", ["http://127.0.0.1:8080/", "http://localhost:7860/api", "http://10.0.0.5/",
                                 "http://192.168.1.1/", "http://169.254.169.254/latest/meta-data/", "http://[::1]/",
                                 "http://0.0.0.0/", "file:///C:/Windows/win.ini", "ftp://example.com/"])
def test_ssrf_guard(url):
    with pytest.raises(ValueError):
        web.check_url(url)
    with pytest.raises(ValueError):
        web.fetch(url)


def test_redirect_to_private_address_is_refused(monkeypatch):
    import urllib.error, email.message

    def fake_open(req, timeout):
        h = email.message.Message()
        h["Location"] = "http://127.0.0.1:8080/secret"
        raise urllib.error.HTTPError(req.full_url, 302, "Found", h, None)

    monkeypatch.setattr(web, "check_url", lambda u: u if "127.0.0.1" not in u else (_ for _ in ()).throw(ValueError("private")))
    monkeypatch.setattr(web._opener, "open", fake_open)
    with pytest.raises(ValueError, match="private"):
        web.fetch("https://public.example/")


def test_results_block():
    res = [{"title": "A", "url": "https://a.com", "snippet": "snip a"}, {"title": "B", "url": "https://b.com", "snippet": "snip b"}]
    pages = [{"url": "https://b.com", "title": "B", "text": "full text of b"}, {"url": "https://c.com", "title": "C", "text": "c text"}]
    out = web.results_block("q", res, pages)
    assert "[1] A — https://a.com\nsnip a" in out
    assert "[2] B — https://b.com\nfull text of b" in out and "snip b" not in out
    assert "[3] C — https://c.com\nc text" in out
    assert "[n]" in out
    assert "no results" in web.results_block("q", [], [])


@pytest.mark.skipif(os.environ.get("SPARKY_LIVE") != "1", reason="network test: set SPARKY_LIVE=1")
def test_live_search_and_fetch():
    r = web.search("python programming language", 5)
    assert r and all(x["url"].startswith("http") for x in r)
    page = web.fetch("https://www.python.org/")
    assert "Python" in page["title"] and len(page["text"]) > 200
