"""Coverage for the seed web-access tools: ``seed/tools/search.py`` (keyless
Exa MCP search, primary, falling back to brave/wikipedia/duckduckgo) and
``seed/tools/fetch.py`` (direct GET -> Exa MCP fetch -> jina -> wayback
fallback chain, with arXiv/GitHub URL rewriting and soft-block detection).

No real network access is used: every test mocks ``urllib.request.urlopen``.
The scripts are loaded by file path (they are standalone tools invoked by the
agent as ``python3 mind/tools/search.py`` / ``fetch.py``, not importable
package modules), matching how they actually ship under ``app/seed/tools``.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SEED_TOOLS = ROOT / "seed" / "tools"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so dataclasses (fetch.py) can resolve their
    # defining module the way the interpreter expects of a "real" import.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


search = _load("_seed_search_under_test", SEED_TOOLS / "search.py")
fetch = _load("_seed_fetch_under_test", SEED_TOOLS / "fetch.py")


class FakeResponse:
    """Enough of an ``http.client.HTTPResponse`` for both tools' call sites."""

    def __init__(self, data: bytes, headers: dict | None = None, url: str = "http://example.test"):
        self._data = data
        self.headers = headers or {}
        self._url = url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, amount: int | None = None) -> bytes:
        return self._data if amount is None else self._data[:amount]

    def geturl(self) -> str:
        return self._url


def sse_payload(result: dict | None = None, error: dict | None = None) -> bytes:
    message: dict = {"jsonrpc": "2.0", "id": 1}
    if error is not None:
        message["error"] = error
    else:
        message["result"] = result
    return f"event: message\ndata: {json.dumps(message)}\n\n".encode("utf-8")


def exa_text_result(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def router(rules):
    """``urlopen`` replacement: first rule whose predicate matches wins."""

    def _urlopen(request, timeout=None):
        url = request.full_url
        for predicate, outcome in rules:
            if predicate(url):
                if isinstance(outcome, BaseException):
                    raise outcome
                if callable(outcome) and not isinstance(outcome, FakeResponse):
                    return outcome(request)
                return outcome
        raise AssertionError(f"no rule matched {url}")

    return _urlopen


def contains(fragment):
    return lambda url: fragment in url


@contextlib.contextmanager
def patched_urlopen(fake):
    with mock.patch.object(urllib.request, "urlopen", fake):
        yield


@contextlib.contextmanager
def captured_streams():
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield out, err


# --------------------------------------------------------------------------
# search.py: Exa MCP protocol
# --------------------------------------------------------------------------

class ExaSearchProtocol(unittest.TestCase):
    def test_parses_multiple_result_blocks(self):
        blob = (
            "Title: First Paper\nURL: https://example.com/a\nPublished: 2020-01-01\n"
            "Author: Someone\nHighlights:\nA highlight sentence.\n\n"
            "Title: Second\nURL: https://example.com/b\nText: Body text for the second result."
        )
        results = search.parse_exa_search(blob)
        self.assertEqual(results, [
            ("https://example.com/a", "First Paper", "A highlight sentence."),
            ("https://example.com/b", "Second", "Body text for the second result."),
        ])

    def test_call_exa_mcp_success_over_sse(self):
        payload = sse_payload(result=exa_text_result("Title: X\nURL: https://x\nText: hi"))
        with patched_urlopen(router([(contains("mcp.exa.ai"), FakeResponse(payload))])):
            text = search.call_exa_mcp("web_search_exa", {"query": "q", "numResults": 5})
        self.assertIn("Title: X", text)

    def test_call_exa_mcp_tool_level_error_raises_value_error(self):
        payload = sse_payload(result={"isError": True, "content": [{"type": "text", "text": "boom"}]})
        with patched_urlopen(router([(contains("mcp.exa.ai"), FakeResponse(payload))])):
            with self.assertRaises(ValueError):
                search.call_exa_mcp("web_search_exa", {"query": "q"})

    def test_call_exa_mcp_json_rpc_error_raises_value_error(self):
        payload = sse_payload(error={"message": "bad request"})
        with patched_urlopen(router([(contains("mcp.exa.ai"), FakeResponse(payload))])):
            with self.assertRaises(ValueError):
                search.call_exa_mcp("web_search_exa", {"query": "q"})

    def test_429_is_reworded_but_stays_an_http_error(self):
        error = urllib.error.HTTPError("https://mcp.exa.ai/mcp", 429, "Too Many Requests", {}, None)
        with patched_urlopen(router([(contains("mcp.exa.ai"), error)])):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                search.call_exa_mcp("web_search_exa", {"query": "q"})
        self.assertEqual(ctx.exception.code, 429)
        self.assertIn("keyless", ctx.exception.reason)

    def test_exa_engine_returns_parsed_results(self):
        payload = sse_payload(result=exa_text_result("Title: Only\nURL: https://only\nText: body"))
        with patched_urlopen(router([(contains("mcp.exa.ai"), FakeResponse(payload))])):
            self.assertEqual(search.exa("q", 3), [("https://only", "Only", "body")])


# --------------------------------------------------------------------------
# search.py: engine ordering / fallback / exit codes via main()
# --------------------------------------------------------------------------

class SearchEngineOrderingAndExitCodes(unittest.TestCase):
    def _run(self, argv):
        with mock.patch.object(sys, "argv", ["search.py", *argv]):
            with captured_streams() as (out, err):
                code = search.main()
        return code, out.getvalue(), err.getvalue()

    def test_exa_is_tried_first_and_wins(self):
        payload = sse_payload(result=exa_text_result("Title: Landau\nURL: https://l\nText: eigenvalue"))
        rules = [(contains("mcp.exa.ai"), FakeResponse(payload))]
        with patched_urlopen(router(rules)):
            code, out, err = self._run(["landau widom"])
        self.assertEqual(code, 0)
        self.assertIn("Landau", out)
        self.assertIn("[exa]", err)

    def test_exa_429_falls_back_to_brave_and_says_so(self):
        brave_html = (
            '<div class="snippet fdb" data-type="web">'
            '<a href="https://example.com/page">'
            '<div class="search-snippet-title">Example Title</div></a>'
            '<div class="content">Example snippet text</div></div>'
        )
        rate_limited = urllib.error.HTTPError("https://mcp.exa.ai/mcp", 429, "Too Many Requests", {}, None)
        rules = [
            (contains("mcp.exa.ai"), rate_limited),
            (contains("search.brave.com"), FakeResponse(brave_html.encode())),
        ]
        with patched_urlopen(router(rules)):
            code, out, err = self._run(["q"])
        self.assertEqual(code, 0)
        self.assertIn("Example Title", out)
        self.assertIn("skipped exa", err)
        self.assertIn("[brave]", err)

    def test_engine_flag_restricts_to_one_engine(self):
        wiki_json = json.dumps({"query": {"search": [{"title": "T", "snippet": "S"}]}}).encode()
        rules = [(contains("wikipedia.org"), FakeResponse(wiki_json))]
        with patched_urlopen(router(rules)):
            code, out, err = self._run(["q", "--engine", "wikipedia"])
        self.assertEqual(code, 0)
        self.assertIn("[wikipedia]", err)

    def test_all_engines_empty_is_exit_3(self):
        empty_sse = sse_payload(result=exa_text_result(""))
        rules = [
            (contains("mcp.exa.ai"), FakeResponse(empty_sse)),
            (contains("search.brave.com"), FakeResponse(b"<html>no results markup</html>")),
            (contains("wikipedia.org"), FakeResponse(json.dumps({"query": {"search": []}}).encode())),
            (contains("duckduckgo.com"), FakeResponse(b"<html></html>")),
        ]
        with patched_urlopen(router(rules)):
            code, out, err = self._run(["q"])
        self.assertEqual(code, 3)

    def test_all_engines_unreachable_is_exit_4(self):
        with patched_urlopen(router([(lambda u: True, urllib.error.URLError("no route"))])):
            code, out, err = self._run(["q"])
        self.assertEqual(code, 4)

    def test_all_engines_refused_is_exit_5(self):
        refused = urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)
        with patched_urlopen(router([(lambda u: True, refused)])):
            code, out, err = self._run(["q"])
        self.assertEqual(code, 5)


# --------------------------------------------------------------------------
# fetch.py: standalone helpers
# --------------------------------------------------------------------------

class SoftBlockDetection(unittest.TestCase):
    def test_detects_known_challenge_phrases(self):
        for phrase in [
            "Just a moment...",
            "Client Challenge",
            "Making sure you're not a bot!",
            "Please enable JavaScript and cookies to continue",
            "Attention Required! | Cloudflare",
            "cf-chl-widget",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIsNotNone(fetch.is_soft_blocked(phrase))

    def test_short_extraction_is_flagged(self):
        self.assertIsNotNone(fetch.is_soft_blocked("hi"))

    def test_real_content_passes(self):
        body = "This is a perfectly ordinary paragraph of article text. " * 10
        self.assertIsNone(fetch.is_soft_blocked(body))


class HtmlExtraction(unittest.TestCase):
    def test_drops_chrome_keeps_structure_and_links_sparingly(self):
        page = (
            "<html><head><style>.x{color:red}</style></head><body>"
            "<nav>menu</nav>"
            "<h1>Title</h1>"
            '<p>Hello <a href="https://x.com/a">world</a>.</p>'
            "<ul><li>one</li><li>two</li></ul>"
            "<footer>foot</footer></body></html>"
        )
        text = fetch.extract_html(page)
        self.assertIn("# Title", text)
        self.assertIn("Hello world (https://x.com/a).", text)
        self.assertIn("- one", text)
        self.assertIn("- two", text)
        self.assertNotIn("menu", text)
        self.assertNotIn("foot", text)
        self.assertNotIn("color:red", text)

    def test_looks_like_html_sniffs_doctype_and_tag(self):
        self.assertTrue(fetch.looks_like_html("<!DOCTYPE html><html><body>x</body></html>"))
        self.assertTrue(fetch.looks_like_html("<HTML><body>x</body></HTML>"))
        self.assertFalse(fetch.looks_like_html("plain text, no markup here"))


class UrlRewriting(unittest.TestCase):
    def test_arxiv_abs_url_yields_html_and_abs_candidates_and_hints(self):
        candidates, hints = fetch.rewrite_url("https://arxiv.org/abs/1804.01257")
        self.assertEqual(candidates, [
            "https://arxiv.org/html/1804.01257",
            "https://arxiv.org/abs/1804.01257",
        ])
        self.assertTrue(any("export.arxiv.org/api/query" in h for h in hints))
        self.assertTrue(any("e-print/1804.01257" in h for h in hints))

    def test_arxiv_pdf_url_adds_pdf_candidate_and_exa_hint(self):
        candidates, hints = fetch.rewrite_url("https://arxiv.org/pdf/1804.01257")
        self.assertIn("https://arxiv.org/pdf/1804.01257", candidates)
        self.assertTrue(any("exa" in h for h in hints))

    def test_github_blob_rewrites_to_raw_githubusercontent(self):
        candidates, hints = fetch.rewrite_url("https://github.com/foo/bar/blob/main/src/x.py")
        self.assertEqual(candidates, ["https://raw.githubusercontent.com/foo/bar/main/src/x.py"])
        self.assertTrue(any("raw content" in h for h in hints))

    def test_github_repo_url_suggests_clone_without_cloning(self):
        candidates, hints = fetch.rewrite_url("https://github.com/foo/bar")
        self.assertEqual(candidates, ["https://github.com/foo/bar"])
        self.assertTrue(any("git clone --depth 1" in h for h in hints))

    def test_ordinary_url_is_unchanged(self):
        candidates, hints = fetch.rewrite_url("https://example.com/page")
        self.assertEqual(candidates, ["https://example.com/page"])
        self.assertEqual(hints, [])


# --------------------------------------------------------------------------
# fetch.py: per-method attempts
# --------------------------------------------------------------------------

class DirectAttempt(unittest.TestCase):
    def test_plain_ua_403_retries_with_browser_ua_and_succeeds(self):
        html_ok = "<html><body><h1>Hi</h1><p>" + ("word " * 60) + "</p></body></html>"
        forbidden = urllib.error.HTTPError("u", 403, "Forbidden", {}, None)
        calls = []

        def _urlopen(request, timeout=None):
            calls.append(request.headers.get("User-agent", ""))
            if len(calls) == 1:
                raise forbidden
            return FakeResponse(html_ok.encode(), headers={"Content-Type": "text/html"})

        with patched_urlopen(_urlopen):
            ok, detail, text = fetch.attempt_direct("http://example.com")
        self.assertTrue(ok)
        self.assertIn("browser UA", detail)
        self.assertIn("Hi", text)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0], calls[1])

    def test_soft_block_on_200_is_a_failure(self):
        blocked = "<html><body>Just a moment... checking your browser</body></html>"
        with patched_urlopen(lambda req, timeout=None: FakeResponse(
                blocked.encode(), headers={"Content-Type": "text/html"})):
            ok, detail, text = fetch.attempt_direct("http://blocked.example")
        self.assertFalse(ok)
        self.assertIn("soft-blocked", detail)
        self.assertIsNone(text)

    def test_non_retryable_status_fails_immediately(self):
        with patched_urlopen(lambda req, timeout=None: (_ for _ in ()).throw(
                urllib.error.HTTPError("u", 404, "Not Found", {}, None))):
            ok, detail, text = fetch.attempt_direct("http://missing.example")
        self.assertFalse(ok)
        self.assertIn("404", detail)

    def test_plain_text_is_not_run_through_html_extractor(self):
        body = b"def f():\n    return 1\n\n" * 20
        with patched_urlopen(lambda req, timeout=None: FakeResponse(
                body, headers={"Content-Type": "text/plain"})):
            ok, detail, text = fetch.attempt_direct("http://raw.example/x.py")
        self.assertTrue(ok)
        self.assertEqual(text, body.decode())


class ExaFetchAttempt(unittest.TestCase):
    def test_success(self):
        payload = sse_payload(result=exa_text_result("# Title\nURL: u\n\n" + ("body " * 60)))
        with patched_urlopen(router([(contains("mcp.exa.ai"), FakeResponse(payload))])):
            ok, detail, text = fetch.attempt_exa_fetch("https://example.com", 4000)
        self.assertTrue(ok)
        self.assertIn("Title", text)

    def test_soft_block_detected_even_via_exa(self):
        # e.g. JSTOR returns "Client Challenge" even through Exa's fetch tool.
        payload = sse_payload(result=exa_text_result("Client Challenge"))
        with patched_urlopen(router([(contains("mcp.exa.ai"), FakeResponse(payload))])):
            ok, detail, text = fetch.attempt_exa_fetch("https://www.jstor.org/stable/x", 4000)
        self.assertFalse(ok)
        self.assertIn("soft-blocked", detail)

    def test_rate_limit_reported_not_raised(self):
        error = urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)
        with patched_urlopen(router([(contains("mcp.exa.ai"), error)])):
            ok, detail, text = fetch.attempt_exa_fetch("https://example.com", 4000)
        self.assertFalse(ok)
        self.assertIn("429", detail)


class JinaAndWaybackAttempts(unittest.TestCase):
    def test_jina_401_is_reported_clearly(self):
        with patched_urlopen(lambda req, timeout=None: (_ for _ in ()).throw(
                urllib.error.HTTPError("u", 401, "Unauthorized", {}, None))):
            ok, detail, text = fetch.attempt_jina("https://example.com")
        self.assertFalse(ok)
        self.assertIn("401", detail)

    def test_wayback_timeout_is_reported(self):
        with patched_urlopen(lambda req, timeout=None: (_ for _ in ()).throw(
                urllib.error.URLError("timed out"))):
            ok, detail, text = fetch.attempt_wayback("https://example.com")
        self.assertFalse(ok)
        self.assertIn("unreachable or timed out", detail)


# --------------------------------------------------------------------------
# fetch.py: orchestration and CLI
# --------------------------------------------------------------------------

class FetchOneOrdering(unittest.TestCase):
    def test_falls_through_direct_to_exa(self):
        payload = sse_payload(result=exa_text_result("# T\nURL: u\n\n" + ("word " * 60)))
        rules = [
            (contains("mcp.exa.ai"), FakeResponse(payload)),
            (lambda u: True, urllib.error.URLError("no route")),
        ]
        with patched_urlopen(router(rules)):
            result = fetch.fetch_one("https://example.com", 4000)
        self.assertTrue(result.ok)
        self.assertEqual(result.method, "exa")
        self.assertTrue(any(line.startswith("direct: failed") for line in result.log))

    def test_method_flag_restricts_to_one_method(self):
        with patched_urlopen(lambda req, timeout=None: (_ for _ in ()).throw(
                urllib.error.URLError("no route"))):
            result = fetch.fetch_one("https://example.com", 4000, only_method="jina")
        self.assertFalse(result.ok)
        self.assertEqual(len(result.log), 1)
        self.assertTrue(result.log[0].startswith("jina:"))

    def test_total_failure_carries_all_attempt_reasons(self):
        with patched_urlopen(lambda req, timeout=None: (_ for _ in ()).throw(
                urllib.error.URLError("no route"))):
            result = fetch.fetch_one("https://example.com", 4000)
        self.assertFalse(result.ok)
        self.assertIsNone(result.method)
        self.assertEqual({line.split(":")[0] for line in result.log},
                         {"direct", "exa", "jina", "wayback"})


class FetchCli(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._orig_cache_dir = fetch.CACHE_DIR
        fetch.CACHE_DIR = Path(self._tmp.name)
        self.addCleanup(lambda: setattr(fetch, "CACHE_DIR", self._orig_cache_dir))

    def _run(self, argv):
        with mock.patch.object(sys, "argv", ["fetch.py", *argv]):
            with captured_streams() as (out, err):
                code = fetch.main()
        return code, out.getvalue(), err.getvalue()

    def test_success_saves_file_and_prints_excerpt(self):
        html_ok = "<html><body><p>" + ("hello world " * 100) + "</p></body></html>"
        with patched_urlopen(lambda req, timeout=None: FakeResponse(
                html_ok.encode(), headers={"Content-Type": "text/html"})):
            code, out, err = self._run(["https://example.com", "--max-chars", "20"])
        self.assertEqual(code, 0)
        self.assertIn("[direct]", out)
        self.assertIn("saved:", out)
        saved_path = Path(out.splitlines()[1].split("saved: ", 1)[1])
        self.assertTrue(saved_path.exists())
        self.assertGreater(len(saved_path.read_text()), 20)

    def test_partial_failure_is_exit_1(self):
        def _urlopen(request, timeout=None):
            if "good.example" in request.full_url:
                return FakeResponse(b"<html><body><p>" + b"ok " * 100 + b"</p></body></html>",
                                    headers={"Content-Type": "text/html"})
            raise urllib.error.URLError("no route")

        with patched_urlopen(_urlopen):
            code, out, err = self._run(["https://good.example", "https://bad.example"])
        self.assertEqual(code, 1)

    def test_total_failure_is_exit_2(self):
        with patched_urlopen(lambda req, timeout=None: (_ for _ in ()).throw(
                urllib.error.URLError("no route"))):
            code, out, err = self._run(["https://a.example", "https://b.example"])
        self.assertEqual(code, 2)

    def test_out_with_multiple_urls_is_a_usage_error(self):
        code, out, err = self._run(["https://a.example", "https://b.example", "--out", "/tmp/x.txt"])
        self.assertEqual(code, 2)
        self.assertIn("--out", err)


if __name__ == "__main__":
    unittest.main()


class FetchDefaultLocationCase(unittest.TestCase):
    def test_inside_a_mind_full_texts_go_to_mind_space_fetched(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            mind = Path(temp) / "mind"
            (mind / "tools").mkdir(parents=True)
            (mind / "space").mkdir()
            copy = mind / "tools" / "fetch.py"
            copy.write_text((SEED_TOOLS / "fetch.py").read_text())
            module = _load("fetch_in_mind", copy)
            self.assertEqual(module.CACHE_DIR, (mind / "space" / "fetched").resolve())

    def test_outside_a_mind_uses_a_per_user_temp_directory(self) -> None:
        import os
        module = _load("fetch_in_seed", SEED_TOOLS / "fetch.py")
        self.assertEqual(module.CACHE_DIR.name, f"artificium-fetch-cache-{os.getuid()}")
