#!/usr/bin/env python3
"""Fetch one or more URLs to readable text. Standard library only.

Tries several methods per URL and stops at the first real success. Prints,
for each URL, one line per method attempted (success, or why it failed), the
path of the file the full extracted text was saved to, and a bounded excerpt.
Reports network trouble on stdout/stderr; it never raises a traceback for it.

    python3 mind/tools/fetch.py URL [URL ...] [--max-chars N] [--out PATH] [--method NAME]

Methods, tried in order for each URL (unless --method restricts to one):

  1. direct  - a plain urllib GET with an honest User-Agent
     ("Artificium-fetch/1.0 (+python-urllib)"). A 403/429/503 is retried once
     with a browser-like User-Agent (some sites bot-challenge browsers but
     serve plain scripted clients fine, or vice versa; both orders happen in
     the wild, so both are tried). Any 200 response, from either UA, is
     checked for soft-block/challenge pages (see is_soft_blocked()) before
     being accepted: a "page" can succeed at the HTTP level and still be a
     "prove you're human" wall instead of content.
  2. exa     - the keyless Exa MCP endpoint (web_fetch_exa; same endpoint
     search.py's "exa" engine uses for web_search_exa), which also handles
     PDFs, ScienceDirect, and some other hard sites directly.
  3. jina    - https://r.jina.ai/URL, a third-party reader proxy. Short
     timeout, best-effort: it returns HTTP 401 for anonymous callers on some
     networks.
  4. wayback - a Wayback Machine snapshot of the URL. Short timeout,
     best-effort: archive.org is sometimes slow or unresponsive.

arXiv abs/pdf/html URLs and GitHub blob/repo URLs are special-cased first
(see rewrite_url()). This only changes which URL(s) the "direct" method
tries, or adds an informational hint line; it never changes the method order
above, and a GitHub *repository* URL (as opposed to a file) is never cloned
automatically, only suggested.

Saves the full extracted text of each URL to --out (only valid for a single
URL) or otherwise to mind/space/fetched/ (printed) under a name derived from
a hash of the URL, so re-fetching the same URL reuses the same path.

Exit codes: 0 every URL fetched, 1 some URLs fetched and some failed
(partial success), 2 every URL failed, or a usage error (e.g. --out with more
than one URL).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

PLAIN_AGENT = "Artificium-fetch/1.0 (+python-urllib)"
# Some sites bot-challenge an honest UA but serve a browser-like one fine
# (the reverse also happens - see the module docstring - which is why plain
# is tried first: it is the more informative failure of the two).
BROWSER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Firefox/128.0"

DIRECT_TIMEOUT = 20
SHORT_TIMEOUT = 10  # for the optional, best-effort last resorts
MAX_BYTES = 12_000_000
DEFAULT_MAX_CHARS = 4000

def _default_cache_dir() -> Path:
    """Where full texts are saved when --out is not given.

    Inside a mind (this script at mind/tools/fetch.py) that is
    mind/space/fetched/, so fetched sources persist and can be cited later.
    Elsewhere, a per-user temporary directory: a shared /tmp path fails with
    PermissionError when another account created it first.
    """
    tools = Path(__file__).resolve().parent
    if tools.name == "tools" and (tools.parent / "space").is_dir():
        return tools.parent / "space" / "fetched"
    return Path(tempfile.gettempdir()) / f"artificium-fetch-cache-{os.getuid()}"


CACHE_DIR = _default_cache_dir()

# Keyless MCP endpoint for Exa's hosted search/fetch tools; no account or API
# key needed. Protocol ported from the reference TypeScript client "exa.ts"
# (callExaMcp / parseMcpResults, MIT licensed): a JSON-RPC tools/call POST
# answered as Server-Sent Events, one "data: {...}" line per event.
EXA_MCP_URL = "https://mcp.exa.ai/mcp"
# mcp.exa.ai answers Python's default "Python-urllib/3.x" User-Agent with a
# flat 403; any explicit one (curl's, a browser's, or this honest one) is
# accepted, so one is required here even though it is not otherwise needed.
EXA_HEADERS = {"Content-Type": "application/json",
              "Accept": "application/json, text/event-stream",
              "User-Agent": PLAIN_AGENT}

# Text that shows up on bot-challenge / consent walls rather than content.
# Matched against extracted text (lowercased), so it survives HTML markup.
SOFT_BLOCK_PATTERNS = [
    r"just a moment",
    r"client challenge",
    r"making sure you'?re not a bot",
    r"verify you are( a)? human",
    r"are you a robot",
    r"attention required",
    r"\bcaptcha\b",
    r"cf-chl",
    r"enable javascript and cookies",
    r"access denied",
    r"checking your browser",
]

METHODS = ["direct", "exa", "jina", "wayback"]


# --------------------------------------------------------------------------
# Low-level HTTP
# --------------------------------------------------------------------------

def http_get(url: str, agent: str, timeout: float, max_bytes: int = MAX_BYTES):
    """A single GET. Raises urllib.error.HTTPError/URLError/OSError as-is."""
    headers = {
        "User-Agent": agent,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    }
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        body = response.read(max_bytes + 1)
        final_url = response.geturl() or url
    return content_type, body[:max_bytes], len(body) > max_bytes, final_url


def is_soft_blocked(text_sample: str) -> str | None:
    """Return a reason string if this looks like a challenge/consent wall."""
    lowered = text_sample.lower()
    for pattern in SOFT_BLOCK_PATTERNS:
        if re.search(pattern, lowered):
            return f"looks like a bot-challenge or consent page (matched {pattern!r})"
    if len(text_sample.strip()) < 200:
        return "too little text after extraction (likely blocked, empty, or JS-only)"
    return None


# --------------------------------------------------------------------------
# HTML -> readable text
# --------------------------------------------------------------------------

_SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


class TextExtractor(HTMLParser):
    """Readable text from HTML: drop chrome, keep structure, keep links sparingly."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: list[str] = []
        self._skip_depth = 0
        self._pre_depth = 0
        self._link_href: str | None = None

    def _start(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in _HEADING_TAGS:
            self.chunks.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "p":
            self.chunks.append("\n\n")
        elif tag == "li":
            self.chunks.append("\n- ")
        elif tag == "br":
            self.chunks.append("\n")
        elif tag in ("pre", "code"):
            self._pre_depth += 1
        elif tag == "a":
            for name, value in attrs:
                if name == "href" and value and value.startswith(("http://", "https://")):
                    self._link_href = value
                    break

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._start(tag, attrs)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._start(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in ("pre", "code"):
            self._pre_depth = max(0, self._pre_depth - 1)
        elif tag == "a" and self._link_href:
            self.chunks.append(f" ({self._link_href})")
            self._link_href = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self.chunks.append(data)

    def text(self) -> str:
        raw = "".join(self.chunks)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", raw)
        return raw.strip()


def extract_html(html_text: str) -> str:
    parser = TextExtractor()
    try:
        parser.feed(html_text)
    except Exception:
        pass  # malformed markup: keep whatever was collected before the error
    return parser.text()


def looks_like_html(sample: str) -> bool:
    return "<html" in sample[:1000].lower() or "<!doctype html" in sample[:200].lower()


def extract_pdf(body: bytes) -> tuple[bool, str, str | None]:
    """pdftotext if it's on PATH; otherwise fail so the caller falls through."""
    if not shutil.which("pdftotext"):
        return False, "no pdftotext on PATH; falling through to other methods (e.g. exa)", None
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        handle.write(body)
        tmp_path = handle.name
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", tmp_path, "-"], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"pdftotext failed: {exc}", None
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[:200]
        return False, f"pdftotext exited {result.returncode}: {detail}", None
    return True, "", result.stdout.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# arXiv / GitHub URL rewriting
# --------------------------------------------------------------------------

_ARXIV_ID_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf|html|e-print)/([^?#]+?)(?:\.pdf)?/?$", re.I)


def rewrite_url(url: str) -> tuple[list[str], list[str]]:
    """Return (candidates for the 'direct' method, informational hint lines)."""
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()

    if "arxiv.org" in host:
        match = _ARXIV_ID_RE.search(url)
        if match:
            arxiv_id = match.group(1)
            hints = [
                f"arxiv hint: metadata at https://arxiv.org/abs/{arxiv_id}, full text at "
                f"https://arxiv.org/html/{arxiv_id}; the export API "
                f"(http://export.arxiv.org/api/query?id_list={arxiv_id}) gives structured "
                f"metadata directly, and https://arxiv.org/e-print/{arxiv_id} is the raw "
                "LaTeX/PS source if both of those fail."
            ]
            candidates = [f"https://arxiv.org/html/{arxiv_id}", f"https://arxiv.org/abs/{arxiv_id}"]
            if "/pdf/" in parsed.path.lower():
                hints.append(
                    "arxiv hint: this was a PDF URL; the 'exa' method usually handles "
                    "arXiv PDFs better than a direct fetch does.")
                candidates.append(f"https://arxiv.org/pdf/{arxiv_id}")
            return candidates, hints

    if host in ("github.com", "www.github.com"):
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 5 and parts[2] == "blob":
            owner, repo, _blob, ref, *rest = parts
            raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{'/'.join(rest)}"
            return [raw], [f"github hint: rewrote blob URL to raw content: {raw}"]
        if len(parts) == 2:
            owner, repo = parts
            return [url], [
                f"github hint: this is a repository, not a single file; "
                f"`git clone --depth 1 https://github.com/{owner}/{repo}.git` gets the "
                "whole thing without scraping the HTML page (not run automatically)."]

    return [url], []


# --------------------------------------------------------------------------
# Exa MCP (web_fetch_exa)
# --------------------------------------------------------------------------

def call_exa_mcp(tool: str, arguments: dict, timeout: float = 30) -> str:
    """Call a keyless Exa MCP tool, return content[0].text. See module docstring."""
    url = EXA_MCP_URL + "?" + urllib.parse.urlencode({"tools": tool})
    body = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=EXA_HEADERS, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise urllib.error.HTTPError(
                exc.url, exc.code, "keyless Exa rate limit; retry later or set EXA_API_KEY",
                exc.headers, exc.fp) from exc
        raise
    payload = None
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        candidate = line[len("data:"):].strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if "result" in parsed or "error" in parsed:
            payload = parsed
    if payload is None:
        raise ValueError("exa: no JSON-RPC payload found in the SSE response")
    if "error" in payload:
        raise ValueError(f"exa: {payload['error'].get('message', payload['error'])}")
    result = payload.get("result", {})
    if result.get("isError"):
        content = result.get("content") or [{}]
        raise ValueError(f"exa: {content[0].get('text', 'tool reported an error')}")
    content = result.get("content") or []
    if not content or "text" not in content[0]:
        raise ValueError("exa: response had no content text")
    return content[0]["text"]


# --------------------------------------------------------------------------
# Per-method attempts
# --------------------------------------------------------------------------

def attempt_direct(url: str) -> tuple[bool, str, str | None]:
    """Plain UA, then browser UA on 403/429/503. Returns (ok, detail, text)."""
    for attempt, agent in ((1, PLAIN_AGENT), (2, BROWSER_AGENT)):
        ua_label = "plain UA" if attempt == 1 else "browser UA"
        try:
            content_type, body, truncated, final_url = http_get(url, agent, DIRECT_TIMEOUT)
        except urllib.error.HTTPError as exc:
            if attempt == 1 and exc.code in (403, 429, 503):
                continue
            return False, f"HTTP {exc.code} {exc.reason} ({ua_label})", None
        except (urllib.error.URLError, OSError) as exc:
            return False, f"unreachable or timed out ({ua_label}): {exc}", None

        is_pdf = "pdf" in content_type.lower() or url.lower().split("?")[0].endswith(".pdf")
        if is_pdf:
            ok, detail, extracted = extract_pdf(body)
            if not ok:
                return False, detail, None
        else:
            raw_text = body.decode("utf-8", errors="replace")
            extracted = extract_html(raw_text) if looks_like_html(raw_text) else raw_text

        block = is_soft_blocked(extracted)
        if block:
            return False, f"soft-blocked ({ua_label}): {block}", None
        note = f"direct GET ({ua_label})" + (", truncated" if truncated else "")
        return True, note, extracted
    return False, "refused by both plain and browser User-Agent", None


def attempt_exa_fetch(url: str, max_chars: int) -> tuple[bool, str, str | None]:
    try:
        text_blob = call_exa_mcp(
            "web_fetch_exa", {"urls": [url], "maxCharacters": max(max_chars, 20000)})
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code} {exc.reason}", None
    except (urllib.error.URLError, OSError) as exc:
        return False, f"unreachable or timed out: {exc}", None
    except (ValueError, KeyError) as exc:
        return False, str(exc), None
    block = is_soft_blocked(text_blob)
    if block:
        return False, f"soft-blocked: {block}", None
    return True, "exa fetch", text_blob


def attempt_jina(url: str) -> tuple[bool, str, str | None]:
    target = "https://r.jina.ai/" + url
    try:
        content_type, body, truncated, final_url = http_get(target, PLAIN_AGENT, SHORT_TIMEOUT)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return False, "401 unauthorized for anonymous callers (needs a Jina API key)", None
        return False, f"HTTP {exc.code} {exc.reason}", None
    except (urllib.error.URLError, OSError) as exc:
        return False, f"unreachable or timed out: {exc}", None
    text_body = body.decode("utf-8", errors="replace")
    block = is_soft_blocked(text_body)
    if block:
        return False, f"soft-blocked: {block}", None
    return True, "jina reader", text_body


def attempt_wayback(url: str) -> tuple[bool, str, str | None]:
    target = f"https://web.archive.org/web/2id_/{url}"
    try:
        content_type, body, truncated, final_url = http_get(target, PLAIN_AGENT, SHORT_TIMEOUT)
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code} {exc.reason}", None
    except (urllib.error.URLError, OSError) as exc:
        return False, f"unreachable or timed out: {exc}", None
    raw_text = body.decode("utf-8", errors="replace")
    extracted = extract_html(raw_text) if looks_like_html(raw_text) else raw_text
    block = is_soft_blocked(extracted)
    if block:
        return False, f"soft-blocked: {block}", None
    return True, "wayback snapshot", extracted


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

@dataclass
class FetchResult:
    ok: bool
    method: str | None
    text: str | None
    final_url: str
    log: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)


def fetch_one(url: str, max_chars: int, only_method: str | None = None) -> FetchResult:
    candidates, hints = rewrite_url(url)
    methods = [only_method] if only_method else METHODS
    log: list[str] = []

    if "direct" in methods:
        for candidate in candidates:
            ok, detail, extracted = attempt_direct(candidate)
            label = "direct" if candidate == url else f"direct ({candidate})"
            log.append(f"{label}: {'ok' if ok else 'failed'} - {detail}")
            if ok:
                return FetchResult(True, "direct", extracted, candidate, log, hints)

    if "exa" in methods:
        ok, detail, extracted = attempt_exa_fetch(candidates[0], max_chars)
        log.append(f"exa: {'ok' if ok else 'failed'} - {detail}")
        if ok:
            return FetchResult(True, "exa", extracted, candidates[0], log, hints)

    if "jina" in methods:
        ok, detail, extracted = attempt_jina(url)
        log.append(f"jina: {'ok' if ok else 'failed'} - {detail}")
        if ok:
            return FetchResult(True, "jina", extracted, url, log, hints)

    if "wayback" in methods:
        ok, detail, extracted = attempt_wayback(url)
        log.append(f"wayback: {'ok' if ok else 'failed'} - {detail}")
        if ok:
            return FetchResult(True, "wayback", extracted, url, log, hints)

    return FetchResult(False, None, None, url, log, hints)


def cache_path_for(url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.txt"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("urls", nargs="+", metavar="URL")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                        help=f"excerpt length printed to stdout (default {DEFAULT_MAX_CHARS})")
    parser.add_argument("--out", type=str, default=None,
                        help="save the full text here instead of the cache directory "
                             "(only valid with a single URL)")
    parser.add_argument("--method", choices=METHODS,
                        help="use only this method instead of trying them in order")
    args = parser.parse_args()

    if args.out and len(args.urls) > 1:
        print("--out only works with a single URL; fetch them one at a time, or drop "
              "--out and use the printed cache-directory paths.", file=sys.stderr)
        return 2

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    successes = failures = 0
    for url in args.urls:
        result = fetch_one(url, args.max_chars, args.method)
        for line in result.log:
            print(f"{url}: {line}", file=sys.stderr)
        for hint in result.hints:
            print(hint, file=sys.stderr)

        if result.ok:
            successes += 1
            out_path = Path(args.out) if args.out else cache_path_for(url)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result.text or "", encoding="utf-8")
            print(f"[{result.method}] {url} -> {result.final_url}")
            print(f"saved: {out_path}")
            excerpt = (result.text or "")[: args.max_chars]
            print(excerpt)
            remaining = len(result.text or "") - len(excerpt)
            if remaining > 0:
                print(f"... [{remaining} more characters in {out_path}]")
        else:
            failures += 1
            print(f"FAILED: {url} - every attempted method failed; see stderr for why.")
        print()

    if failures == 0:
        return 0
    if successes == 0:
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
