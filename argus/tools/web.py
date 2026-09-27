"""The web: fetch a page as text, and search (Brave or a SearXNG instance).

Both run in the argus process, not in the command sandbox, so the approval gate
decides whether they may reach the network (``sandbox.network``, and the ``ask``
policy confirms each host). Over SSH they still run on the machine running argus.
"""

from __future__ import annotations

import html
import json
import os
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from argus.tools.base import Tool, ToolContext, ToolError, ToolResult, obj

MAX_BYTES = 3_000_000
USER_AGENT = "argus (coding agent; +https://github.com/SnoozeWalknn/argus-harness)"
SKIP = {"script", "style", "noscript", "svg", "template", "iframe", "head"}
BLOCK = {
    "p", "div", "section", "article", "main", "header", "footer", "nav", "aside", "ul", "ol",
    "table", "tr", "blockquote", "figure", "figcaption", "dl", "dt", "dd", "form", "hr", "br",
}  # fmt: skip


class _Text(HTMLParser):
    """HTML → readable text with Markdown-ish headings, lists, links and code blocks."""

    def __init__(self, base: str):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.out: list[str] = []
        self.skip = 0
        self.pre = 0
        self.title = ""
        self.in_title = False
        self.href: str | None = None
        self.link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self.in_title = True
        if tag in SKIP:
            self.skip += 1
            return
        if self.skip:
            return
        if tag in BLOCK:
            self.out.append("\n")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag == "pre":
            self.pre += 1
            self.out.append("\n```\n")
        elif tag == "code" and not self.pre:
            self.out.append("`")
        elif tag in ("td", "th"):
            self.out.append(" | ")
        elif tag == "a":
            href = dict(attrs).get("href") or ""
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                self.href = urljoin(self.base, href)
                self.link_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if tag in SKIP:
            self.skip = max(self.skip - 1, 0)
            return
        if self.skip:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6", "p") or tag in BLOCK:
            self.out.append("\n")
        elif tag == "pre":
            self.pre = max(self.pre - 1, 0)
            self.out.append("\n```\n")
        elif tag == "code" and not self.pre:
            self.out.append("`")
        elif tag == "a" and self.href is not None:
            text = "".join(self.link_text).strip()
            if text and text != self.href:
                self.out.append(f" ({self.href})")
            self.href = None

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title += data
        if self.skip:
            return
        if not self.pre:
            data = re.sub(r"\s+", " ", data)
        if self.href is not None:
            self.link_text.append(data)
        self.out.append(data)

    def text(self) -> str:
        raw = "".join(self.out)
        parts = re.split(r"(\n```\n.*?\n```\n)", raw, flags=re.S)
        cleaned = []
        for i, part in enumerate(parts):
            if i % 2:  # code block: keep as is
                cleaned.append(part)
                continue
            lines = [ln.strip() for ln in part.split("\n")]
            cleaned.append("\n".join(lines))
        return re.sub(r"\n{3,}", "\n\n", "".join(cleaned)).strip()


def html_to_text(page: str, base: str = "") -> tuple[str, str]:
    """(title, text) of an HTML page."""
    p = _Text(base)
    p.feed(page)
    p.close()
    return html.unescape(p.title.strip()), p.text()


def _client(timeout: float) -> httpx.Client:
    # trust_env: honours HTTPS_PROXY / SSL_CERT_FILE like every other tool on the machine
    return httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT})


class FetchTool(Tool):
    name = "fetch"
    description = (
        "Fetch a web page or file over HTTP(S) and return it as text (HTML is converted "
        "to Markdown-like text; JSON is pretty-printed). Long pages come in parts: pass "
        "`start` to continue. Works for local servers too, e.g. http://localhost:3000."
    )
    parameters = obj(
        {
            "url": {"type": "string"},
            "start": {"type": "integer", "description": "character offset to continue from"},
        },
        ["url"],
    )
    summary = "fetch a URL as text (start=offset for long pages)"

    def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        url = str(args["url"]).strip()
        if not urlsplit(url).scheme:
            url = "https://" + url
        if urlsplit(url).scheme not in ("http", "https"):
            raise ToolError("only http and https URLs can be fetched")
        start = max(int(args.get("start") or 0), 0)
        limit = ctx.cfg.fetch_max_chars
        try:
            with _client(ctx.cfg.fetch_timeout) as client, client.stream("GET", url) as r:
                body = b""
                for chunk in r.iter_bytes():
                    body += chunk
                    if len(body) > MAX_BYTES:
                        break
                status, ctype, final_url = (
                    r.status_code,
                    r.headers.get("content-type", ""),
                    str(r.url),
                )
                encoding = r.encoding or "utf-8"
        except httpx.HTTPError as e:
            raise ToolError(f"could not fetch {url}: {type(e).__name__}: {e}") from None
        kind = ctype.split(";")[0].strip().lower()
        title = ""
        if "html" in kind or (
            not kind and body.lstrip()[:15].lower().startswith(b"<!doctype html")
        ):
            title, text = html_to_text(body.decode(encoding, "replace"), final_url)
        elif "json" in kind:
            try:
                text = json.dumps(json.loads(body), indent=2, ensure_ascii=False)
            except ValueError:
                text = body.decode(encoding, "replace")
        elif kind.startswith("text/") or kind in ("application/xml", "application/javascript", ""):
            text = body.decode(encoding, "replace")
            if "\0" in text[:4096]:
                text = ""
        else:
            text = ""
        head = f"{final_url}  [{status} {kind or 'unknown type'}]" + (f"  {title}" if title else "")
        if not text:
            return ToolResult(
                f"{head}\n({len(body)} bytes of {kind or 'binary'} content, not shown)",
                ok=status < 400,
                error=f"HTTP {status}" if status >= 400 else None,
                meta={"status": status, "bytes": len(body)},
            )
        part = text[start : start + limit]
        more = len(text) - start - len(part)
        note = (
            f"\n[{more} more characters; fetch again with start={start + len(part)}]"
            if more > 0
            else ""
        )
        return ToolResult(
            f"{head}\n\n{part}{note}",
            ok=status < 400,
            error=f"HTTP {status}" if status >= 400 else None,
            truncated=more > 0,
            meta={"status": status, "chars": len(text), "url": final_url},
        )


# -- search ---------------------------------------------------------------------------------------

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


def search_backend(cfg: Any) -> tuple[str, str, str] | None:
    """(backend, url, key) for ``tools.search``: auto picks Brave with BRAVE_API_KEY, else
    SearXNG at tools.search_url / SEARXNG_URL; None = no search available."""
    choice = cfg.search
    url = cfg.search_url or ""
    if choice == "off":
        return None
    if choice in ("auto", "brave") and os.environ.get("BRAVE_API_KEY"):
        return "brave", url or BRAVE_URL, os.environ["BRAVE_API_KEY"]
    searx = url or os.environ.get("SEARXNG_URL", "")
    if choice in ("auto", "searxng") and searx:
        return "searxng", searx, ""
    return None


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web. Returns titles, URLs and snippets; fetch a URL to read it."
    parameters = obj(
        {
            "query": {"type": "string"},
            "count": {"type": "integer", "description": "results (default 8)"},
        },
        ["query"],
    )
    summary = "search the web; returns titles, urls, snippets"

    def __init__(self, backend: str, url: str, key: str):
        self.backend, self.url, self.key = backend, url, key

    def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        query = str(args["query"]).strip()
        if not query:
            raise ToolError("empty query")
        count = min(max(int(args.get("count") or 8), 1), 20)
        try:
            with _client(ctx.cfg.fetch_timeout) as client:
                if self.backend == "brave":
                    r = client.get(
                        self.url,
                        params={"q": query, "count": count},
                        headers={"X-Subscription-Token": self.key, "Accept": "application/json"},
                    )
                    r.raise_for_status()
                    rows = [
                        (x.get("title", ""), x.get("url", ""), x.get("description", ""))
                        for x in (r.json().get("web") or {}).get("results") or []
                    ]
                else:
                    r = client.get(
                        self.url.rstrip("/") + "/search", params={"q": query, "format": "json"}
                    )
                    r.raise_for_status()
                    rows = [
                        (x.get("title", ""), x.get("url", ""), x.get("content", ""))
                        for x in r.json().get("results") or []
                    ]
        except (httpx.HTTPError, ValueError) as e:
            raise ToolError(f"{self.backend} search failed: {type(e).__name__}: {e}") from None
        rows = rows[:count]
        if not rows:
            return ToolResult("No results.", meta={"count": 0, "backend": self.backend})
        lines = []
        for i, (title, url, snip) in enumerate(rows, 1):
            snip = re.sub(r"<[^>]+>", "", html.unescape(snip or "")).strip()
            lines.append(f"{i}. {title.strip()}\n   {url}\n   {snip[:300]}")
        return ToolResult("\n".join(lines), meta={"count": len(rows), "backend": self.backend})
