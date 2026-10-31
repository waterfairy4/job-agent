"""Web access: read one job page from a link, and search the web.

fetch_page: downloads one page the user gives (like opening it in a browser) and extracts the
main text. Pages that build their content with JavaScript come back almost empty; then, if
Playwright is installed, the page is rendered in a headless browser; otherwise the user is
asked to paste the text.

search: one web search through Baidu (百度 via Qianfan 千帆), Bocha (博查) or Tavily.
"""
import time

import httpx

from jobagent import config

# Transient failures (network drops, timeouts, rate limits, server errors) are retried with exponential
# backoff. Wrong keys or missing pages (other 4xx) fail at once: retrying cannot fix them.
RETRY_STATUS = {429, 500, 502, 503, 504}


def request(c: httpx.Client, method: str, url: str, **kwargs) -> httpx.Response:
    """One HTTP call with up to config.HTTP_RETRIES attempts. Returns the last response, or raises the
    last network error."""
    attempts = max(1, config.HTTP_RETRIES)
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            resp = c.request(method, url, **kwargs)
        except httpx.TransportError:  # ConnectError, timeouts, connection reset
            if last:
                raise
        else:
            if resp.status_code not in RETRY_STATUS or last:
                return resp
        time.sleep(config.HTTP_BACKOFF * 2 ** attempt)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
MIN_TEXT = 300  # fewer characters than this = the page did not really load
# Anti-bot / login walls: the page text is a check, not the job
BLOCKED = ("访问验证", "安全验证", "验证码", "请完成验证", "人机验证", "滑动验证", "拖动滑块", "请先登录",
           "verify you are human", "captcha", "access denied", "are you a robot")


def looks_blocked(text: str) -> bool:
    head = text[:1500].lower()
    return any(b in head for b in BLOCKED)


class WebError(RuntimeError):
    pass


def extract_text(html: str) -> str:
    import trafilatura

    text = trafilatura.extract(html, include_comments=False, include_tables=True, favor_recall=True) or ""
    if len(text) < MIN_TEXT:  # fall back to all visible text
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.decompose()
        alt = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
        text = alt if len(alt) > len(text) else text
    return text.strip()


def has_login() -> bool:
    """True after `python login.py` saved a browser session."""
    from pathlib import Path

    folder = Path(config.BROWSER_PROFILE)
    return folder.is_dir() and any(folder.iterdir())


def _render_with_browser(url: str) -> str | None:
    """Render a page with Playwright, if it is installed. Uses your saved login (login.py) when there is one.
    Returns HTML or None."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    try:
        with sync_playwright() as p:
            if has_login():
                ctx = p.chromium.launch_persistent_context(config.BROWSER_PROFILE, headless=True, user_agent=UA)
                page = ctx.new_page()
            else:
                ctx = p.chromium.launch()
                page = ctx.new_page(user_agent=UA)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=20000)
                try:
                    page.wait_for_load_state("networkidle", timeout=8000)
                except Exception:
                    pass  # some sites never go idle; what has loaded is usually enough
                return page.content()
            finally:
                ctx.close()
    except Exception:
        return None


def fetch_page(url: str, client: httpx.Client | None = None) -> str:
    if not url.startswith(("http://", "https://")):
        raise WebError("That does not look like a web link (it should start with http:// or https://).")
    try:
        c = client or httpx.Client(follow_redirects=True, timeout=20, headers={"User-Agent": UA})
        resp = request(c, "GET", url)
        resp.raise_for_status()
        text = extract_text(resp.text)
    except httpx.HTTPError as e:
        raise WebError(f"Could not open the page ({e.__class__.__name__}).") from e
    if len(text) < MIN_TEXT or looks_blocked(text):  # JavaScript page or a check page: try a real browser
        html = _render_with_browser(url)
        if html:
            rendered = extract_text(html)
            if len(rendered) >= MIN_TEXT or len(rendered) > len(text):
                text = rendered
    if looks_blocked(text):
        hint = "" if has_login() else " Or run `python login.py` once to let the app use your login."
        raise WebError("The site showed a verification or login page instead of the job. "
                       "Please open it in your browser and paste the job description here." + hint)
    if len(text) < MIN_TEXT:
        raise WebError("The page loads its content with JavaScript, so the text could not be read. "
                       "Please copy the job description from the page and paste it here.")
    return text[:20000]


BOCHA_FRESHNESS = {"week": "oneWeek", "month": "oneMonth", "semiyear": "oneYear", "year": "oneYear"}
TAVILY_DAYS = {"week": 7, "month": 30, "semiyear": 183, "year": 365}


def search(query: str, count: int = 5, client: httpx.Client | None = None,
           sites: list[str] | None = None, recency: str = "year") -> list[dict]:
    """Returns [{"title", "url", "snippet", "date"}].

    sites: only results from these domains (e.g. ["www.nowcoder.com"]).
    recency: "week", "month", "semiyear" or "year".
    """
    if not config.SEARCH_API_KEY:
        raise WebError("Web search is not set up: add SEARCH_PROVIDER and SEARCH_API_KEY to your .env file.")
    c = client or httpx.Client(timeout=30)
    headers = {"Authorization": f"Bearer {config.SEARCH_API_KEY}", "Content-Type": "application/json"}
    try:
        if config.SEARCH_PROVIDER == "baidu":
            body = {"messages": [{"role": "user", "content": query[:72]}], "search_source": "baidu_search_v2",
                    "resource_type_filter": [{"type": "web", "top_k": count}],
                    "search_recency_filter": recency}
            if sites:
                body["search_filter"] = {"match": {"site": list(sites)[:20]}}
            resp = request(c, "POST", "https://qianfan.baidubce.com/v2/ai_search/web_search", headers=headers, json=body)
            if resp.status_code == 404:  # older accounts/docs use the shorter path
                resp = request(c, "POST", "https://qianfan.baidubce.com/v2/ai_search", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") and not data.get("references"):
                raise WebError(f"Baidu search error: {data.get('message') or data.get('code')}")
            return [{"title": r.get("title", ""), "url": r.get("url", ""),
                     "snippet": r.get("content", "") or r.get("snippet", ""), "date": r.get("date", "")}
                    for r in data.get("references", [])][:count]
        if config.SEARCH_PROVIDER == "tavily":
            body = {"query": query, "max_results": count, "search_depth": "basic",
                    "days": TAVILY_DAYS.get(recency, 365)}
            if sites:
                body["include_domains"] = list(sites)
            resp = request(c, "POST", "https://api.tavily.com/search", headers=headers, json=body)
            resp.raise_for_status()
            return [{"title": r.get("title", ""), "url": r.get("url", ""),
                     "snippet": r.get("content", ""), "date": r.get("published_date", "")}
                    for r in resp.json().get("results", [])]
        body = {"query": query, "count": count, "summary": True,
                "freshness": BOCHA_FRESHNESS.get(recency, "oneYear")}
        if sites:
            body["include"] = "|".join(sites)
        resp = request(c, "POST", "https://api.bochaai.com/v1/web-search", headers=headers, json=body)
        resp.raise_for_status()
        pages = ((resp.json().get("data") or {}).get("webPages") or {}).get("value") or []
        return [{"title": p.get("name", ""), "url": p.get("url", ""),
                 "snippet": p.get("summary") or p.get("snippet", ""), "date": p.get("datePublished", "")}
                for p in pages]
    except httpx.HTTPStatusError as e:
        raise WebError(f"Web search failed (HTTP {e.response.status_code}). Check SEARCH_API_KEY "
                       f"and SEARCH_PROVIDER in .env.") from e
    except httpx.HTTPError as e:
        raise WebError(f"Web search failed ({e.__class__.__name__}).") from e
