#!/usr/bin/env python3
"""Web search over several engines, first answer wins. Standard library only.

Deliberately low level: it prints results as plain text on stdout, names the
engine that answered, and says plainly why it found nothing, so a failure is
diagnosable instead of silent. Reading a page found here is a separate step;
`lynx -dump URL` already does it.

    python3 mind/tools/search.py "query" [-n COUNT] [--engine NAME]

Engines, tried in order: exa (general web, via the keyless Exa MCP endpoint;
primary engine), brave (general web), wikipedia (encyclopedia only, but
reliable), duckduckgo (unreachable from some hosts; kept as a fallback).
Exa needs no API key but is rate-limited for anonymous callers; a 429 there
falls back to brave and says so on stderr. Brave refuses after a handful of
queries in quick succession, then the search falls back further and says so
on stderr. Space queries out when the results must come from the open web.

An engine that answers with a page of results that have nothing to do with
the query is worse than one that refuses: Bing does exactly that to scripted
clients, which is why it is not on this list.

Exit codes: 0 results, 3 no results, 4 no route to any engine, 5 all refused.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

# Without a browser-like agent these endpoints answer with empty or blocked pages.
AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Firefox/128.0"
HEADERS = {"User-Agent": AGENT, "Accept-Language": "en-US,en;q=0.9"}

Result = tuple[str, str, str]  # url, title, snippet

# Keyless MCP endpoint for Exa's hosted search/fetch tools. No account or API
# key is required, but anonymous callers share a modest rate limit (HTTP 429).
# Setting EXA_API_KEY is optional and not needed for this to work.
EXA_MCP_URL = "https://mcp.exa.ai/mcp"
# mcp.exa.ai answers Python's default "Python-urllib/3.x" User-Agent with a
# flat 403; any explicit one (curl's, a browser's, or this honest one) is
# accepted, so one is required here even though it is not otherwise needed.
EXA_HEADERS = {"Content-Type": "application/json",
              "Accept": "application/json, text/event-stream",
              "User-Agent": "Artificium-search/1.0 (+python-urllib)"}


def fetch(url: str, data: bytes | None = None) -> str:
    request = urllib.request.Request(url, data=data, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8", errors="replace")


def text(fragment: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", fragment))).strip()


def call_exa_mcp(tool: str, arguments: dict, timeout: float = 25) -> str:
    """Call a keyless Exa MCP tool and return its ``content[0].text``.

    Protocol ported from the reference TypeScript client ``exa.ts``
    (``callExaMcp`` / ``parseMcpResults``, MIT licensed): a JSON-RPC
    ``tools/call`` POST answered as Server-Sent Events, one ``data: {...}``
    line per event. Raises ``urllib.error.HTTPError`` (429 reworded for
    clarity) on the keyless rate limit, or ``ValueError`` if Exa reports a
    tool-level error.
    """
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


def parse_exa_search(blob: str) -> list[Result]:
    """Split Exa's ``web_search_exa`` text into (url, title, snippet) blocks."""
    blocks = re.split(r"\n(?=Title:\s)", blob.strip())
    results = []
    for block in blocks:
        title = re.search(r"^Title:\s*(.*)$", block, re.M)
        url = re.search(r"^URL:\s*(.*)$", block, re.M)
        if not (title and url):
            continue
        body = re.search(r"^(?:Highlights|Text):\s*(.*)", block, re.M | re.S)
        snippet = re.sub(r"\s+", " ", body.group(1)).strip() if body else ""
        results.append((url.group(1).strip(), title.group(1).strip(), snippet[:300]))
    return results


def exa(query: str, count: int = 10) -> list[Result]:
    text_blob = call_exa_mcp("web_search_exa", {"query": query, "numResults": count})
    return parse_exa_search(text_blob)


def brave(query: str) -> list[Result]:
    page = fetch("https://search.brave.com/search?" + urllib.parse.urlencode({"q": query}))
    # Each web result is a snippet block; split on the next block's opening tag.
    blocks = re.findall(
        r'<div[^>]*class="snippet[^"]*"[^>]*data-type="web".*?'
        r'(?=<div[^>]*class="snippet[^"]*"[^>]*data-type=|$)', page, re.S)
    results = []
    for block in blocks:
        link = re.search(r'<a[^>]*href="(https?://[^"]+)"', block)
        title = re.search(r'class="[^"]*search-snippet-title[^"]*"[^>]*>(.*?)</div>', block, re.S)
        snippet = re.search(r'class="content[^"]*"[^>]*>(.*?)</div>', block, re.S)
        if link and title:
            results.append((html.unescape(link.group(1)), text(title.group(1)),
                            text(snippet.group(1)) if snippet else ""))
    return results


def wikipedia(query: str) -> list[Result]:
    page = fetch("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "list": "search", "format": "json",
        "srlimit": "10", "srsearch": query}))
    results = []
    for hit in json.loads(page).get("query", {}).get("search", []):
        title = hit.get("title", "")
        url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        results.append((url, title, text(hit.get("snippet", ""))))
    return results


class DuckResults(HTMLParser):
    """Collect DuckDuckGo lite result links and snippets in document order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[list[str]] = []
        self.snippets: list[str] = []
        self._in_link = False
        self._in_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: (value or "") for name, value in attrs}
        classes = values.get("class", "").split()
        if tag == "a" and "result-link" in classes:
            self._in_link = True
            target = urllib.parse.parse_qs(
                urllib.parse.urlparse(values.get("href", "")).query).get("uddg")
            self.links.append([target[0] if target else values.get("href", ""), ""])
        elif tag == "td" and "result-snippet" in classes:
            self._in_snippet = True
            self.snippets.append("")

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._in_link = False
        elif tag == "td":
            self._in_snippet = False

    def handle_data(self, data: str) -> None:
        if self._in_link and self.links:
            self.links[-1][1] += data
        elif self._in_snippet and self.snippets:
            self.snippets[-1] += data


def duckduckgo(query: str) -> list[Result]:
    page = fetch("https://lite.duckduckgo.com/lite/",
                 data=urllib.parse.urlencode({"q": query}).encode())
    parsed = DuckResults()
    parsed.feed(page)
    return [(url, title.strip(), parsed.snippets[i].strip() if i < len(parsed.snippets) else "")
            for i, (url, title) in enumerate(parsed.links) if title.strip()]


ENGINES = {"exa": exa, "brave": brave, "wikipedia": wikipedia, "duckduckgo": duckduckgo}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("-n", type=int, default=10, help="maximum results (default 10)")
    parser.add_argument("--engine", choices=sorted(ENGINES),
                        help="use only this engine instead of trying them in order")
    args = parser.parse_args()

    order = [args.engine] if args.engine else list(ENGINES)
    refused, unreachable, empty = [], [], []
    for name in order:
        try:
            found = ENGINES[name](args.query, args.n) if name == "exa" else ENGINES[name](args.query)
        except urllib.error.HTTPError as exc:
            refused.append(f"{name}: HTTP {exc.code} {exc.reason}")
            continue
        except (urllib.error.URLError, OSError) as exc:
            unreachable.append(f"{name}: {exc}")
            continue
        except (ValueError, KeyError) as exc:
            empty.append(f"{name}: unreadable answer ({exc})")
            continue
        if not found:
            empty.append(f"{name}: answered, no results parsed")
            continue
        # A fallback changes what the results can be: say so, and say why.
        for line in refused + unreachable + empty:
            print(f"skipped {line}", file=sys.stderr)
        print(f"[{name}]" + (" (encyclopedia only)" if name == "wikipedia" else ""),
              file=sys.stderr)
        for index, (url, title, snippet) in enumerate(found[: args.n]):
            print(f"{index + 1}. {title}")
            print(f"   {url}")
            if snippet:
                print(f"   {snippet}")
            print()
        return 0

    for line in refused + unreachable + empty:
        print(line, file=sys.stderr)
    if empty:
        print("An engine answered but nothing was parsed. Either the query genuinely has "
              "no results, or the engine changed its markup. Check with a query you know "
              "to be common before assuming the former; if the markup moved, this script "
              "is yours to fix.", file=sys.stderr)
        return 3
    if refused:
        print("Every reachable engine refused. Rate limiting is the usual cause. Retry "
              "later, use --engine to try another one, or read a known URL directly.",
              file=sys.stderr)
        return 5
    print("No engine was reachable. Check connectivity before concluding anything "
          "about the query itself.", file=sys.stderr)
    return 4


if __name__ == "__main__":
    sys.exit(main())
