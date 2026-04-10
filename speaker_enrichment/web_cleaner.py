"""
web_cleaner.py
==============
Fetches a URL and returns the cleanest possible plain text, stripping
everything that is not potentially relevant to a politician's biography:
navigation, ads, cookie banners, scripts, styles, footers, social-share
widgets, etc.

Strategy (in order):
  1. Wikipedia URLs → Wikipedia w/api.php (plain-text extract, no bot blocking)
  2. Fetch with requests (rotating user agent, timeout)
  3. Try trafilatura  — best at extracting main article body
  4. Fall back to BeautifulSoup — removes boilerplate tags, then converts
     remaining text
  5. Truncate to MAX_CLEANED_TEXT_CHARS

Stores raw HTML to disk under RAW_HTML_DIR for traceability.
Wikipedia API responses are stored as .txt files (no raw HTML to save).
"""

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup

from config import (
    FETCH_TIMEOUT_SECONDS,
    MAX_CLEANED_TEXT_CHARS,
    RAW_HTML_DIR,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
    "Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

# Tags whose entire subtree is discarded before text extraction
BOILERPLATE_TAGS = {
    "script", "style", "noscript", "meta", "link", "head",
    "nav", "header", "footer", "aside",
    "form", "button", "input", "select", "textarea",
    "iframe", "object", "embed", "canvas", "svg",
    "figure",         # often just captions under images
}

# CSS class / id fragments that signal boilerplate containers
BOILERPLATE_PATTERNS = re.compile(
    r"(cookie|consent|gdpr|banner|popup|modal|overlay|"
    r"social|share|comment|sidebar|widget|newsletter|"
    r"adverti|sponsor|related|recommend|promo|breadcrumb|"
    r"pagination|pager|skip-to|site-header|site-footer|"
    r"search-form|language-switch|nav-menu)",
    re.IGNORECASE,
)

# Collapse excess whitespace
_WS = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _url_to_filename(url: str) -> str:
    """Stable filename for a URL based on its SHA-1."""
    return hashlib.sha1(url.encode()).hexdigest() + ".html"


def _ua(idx: int = 0) -> str:
    return USER_AGENTS[idx % len(USER_AGENTS)]


def _is_wikipedia_url(url: str) -> bool:
    """Return True for Wikipedia article URLs (any language)."""
    host = urlparse(url).hostname or ""
    # matches xx.wikipedia.org but not xx.wikimedia.org or wikisource etc.
    return bool(re.match(r"^[a-z]{2,3}\.wikipedia\.org$", host))


def _fetch_wikipedia_api(url: str, timeout: int) -> "FetchResult":
    """
    Fetch a Wikipedia article via the w/api.php plain-text extract endpoint.
    Avoids the HTML endpoint entirely (which gets 403-blocked on server IPs).
    Returns a populated FetchResult on success, or one with .error set.
    """
    result = FetchResult()
    result.url = url

    parsed = urlparse(url)
    lang   = (parsed.hostname or "en").split(".")[0]   # "en", "de", "fr", …
    # Title is the last path component, URL-decoded
    title  = unquote(parsed.path.lstrip("/").split("/")[-1])

    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    params  = {
        "action":      "query",
        "prop":        "extracts",
        "titles":      title,
        "explaintext": "1",    # plain text, no HTML markup
        "redirects":   "1",    # follow redirects automatically
        "format":      "json",
        "formatversion": "2",
    }

    try:
        resp = requests.get(
            api_url, params=params,
            headers={"User-Agent": "ParlSpeakerEnrichment/1.0 (research bot)"},
            timeout=timeout,
        )
        result.http_status = resp.status_code
        if resp.status_code >= 400:
            result.error = f"HTTP {resp.status_code}"
            return result

        data  = resp.json()
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            result.error = "wikipedia_api_empty"
            return result

        page  = pages[0]
        if page.get("missing"):
            result.error = "wikipedia_page_missing"
            return result

        text = (page.get("extract") or "").strip()
        if not text or len(text) < 20:
            result.error = "no_text_extracted"
            return result

        if len(text) > MAX_CLEANED_TEXT_CHARS:
            text = text[:MAX_CLEANED_TEXT_CHARS] + "\n[TRUNCATED]"

        # Store the text as a .txt file for traceability
        raw_dir  = Path(RAW_HTML_DIR)
        raw_dir.mkdir(parents=True, exist_ok=True)
        filename = hashlib.sha1(url.encode()).hexdigest() + ".txt"
        (raw_dir / filename).write_text(text, encoding="utf-8")
        result.raw_html_path   = filename
        result.cleaned_text    = text
        result.cleaned_text_len = len(text)
        return result

    except Exception as e:
        result.error = f"fetch_error: {type(e).__name__}: {e}"
        return result


def _bs_extract(html: str) -> str:
    """BeautifulSoup fallback extractor."""
    soup = BeautifulSoup(html, "lxml")

    # Remove boilerplate tags entirely
    for tag in soup.find_all(BOILERPLATE_TAGS):
        tag.decompose()

    # Remove elements whose class or id smell like boilerplate
    for el in soup.find_all(True):
        classes = " ".join(el.get("class", []))
        el_id   = el.get("id", "")
        if BOILERPLATE_PATTERNS.search(classes) or BOILERPLATE_PATTERNS.search(el_id):
            el.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return _WS.sub("\n\n", text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FetchResult:
    __slots__ = ("url", "http_status", "raw_html_path", "cleaned_text",
                 "cleaned_text_len", "error")

    def __init__(self):
        self.url             = ""
        self.http_status     = None
        self.raw_html_path   = None   # relative path string
        self.cleaned_text    = None
        self.cleaned_text_len = 0
        self.error           = None


def fetch_and_clean(url: str, ua_index: int = 0,
                    store_raw: bool = True) -> FetchResult:
    """
    Fetch `url`, store raw HTML, extract clean text.
    Returns a FetchResult with all fields populated.
    """
    result = FetchResult()
    result.url = url

    # ---- 0. Wikipedia shortcut — use the API to avoid HTML bot-blocking ----
    if _is_wikipedia_url(url):
        return _fetch_wikipedia_api(url, timeout=FETCH_TIMEOUT_SECONDS)

    # ---- 1. HTTP fetch (with one 403 retry using a different User-Agent) ----
    html = None
    for attempt in range(2):
        try:
            resp = requests.get(
                url,
                headers={
                    "User-Agent":      _ua(ua_index + attempt),
                    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en,*;q=0.5",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Cache-Control":   "no-cache",
                },
                timeout=FETCH_TIMEOUT_SECONDS,
                allow_redirects=True,
            )
            result.http_status = resp.status_code
            if resp.status_code == 403 and attempt == 0:
                # Rotate UA and wait briefly before retry
                time.sleep(2)
                continue
            if resp.status_code >= 400:
                result.error = f"HTTP {resp.status_code}"
                return result
            html = resp.text
            break
        except requests.exceptions.Timeout:
            result.error = "timeout"
            return result
        except requests.exceptions.TooManyRedirects:
            result.error = "too_many_redirects"
            return result
        except Exception as e:
            result.error = f"fetch_error: {type(e).__name__}: {e}"
            return result

    if html is None:
        result.error = f"HTTP {result.http_status}"
        return result

    # ---- 2. Store raw HTML ----
    if store_raw:
        raw_dir = Path(RAW_HTML_DIR)
        raw_dir.mkdir(parents=True, exist_ok=True)
        filename = _url_to_filename(url)
        (raw_dir / filename).write_text(html, encoding="utf-8", errors="replace")
        result.raw_html_path = filename   # relative path stored in DB

    # ---- 3. Extract clean text ----
    # Try trafilatura first (best at article bodies)
    cleaned = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        favor_recall=True,   # we want more coverage, not less
    )

    if not cleaned or len(cleaned.strip()) < 100:
        # Fall back to BS4
        cleaned = _bs_extract(html)

    if not cleaned or len(cleaned.strip()) < 20:
        result.error = "no_text_extracted"
        return result

    # Truncate to ceiling (we want quality, not quantity for the LLM)
    if len(cleaned) > MAX_CLEANED_TEXT_CHARS:
        cleaned = cleaned[:MAX_CLEANED_TEXT_CHARS] + "\n[TRUNCATED]"

    result.cleaned_text     = cleaned
    result.cleaned_text_len = len(cleaned)
    return result
