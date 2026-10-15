"""Web retrieval, HTML interpretation, and tool presentation.

Inference is supplied by the runtime; this module owns no conversation or
credential authority. The caller supplies the process-lifetime fetch cache.
"""

import asyncio
import time
import urllib.parse
from html.parser import HTMLParser

from . import formats, http_client


WEBFETCH_TIMEOUT_S = 30
WEBFETCH_MAX_BYTES = 10_485_760  # 10 MiB
WEBFETCH_MAX_PROMPT_CHARS = 200_000
WEBFETCH_CACHE_TTL = 15 * 60    # 15 minutes
WEBFETCH_CACHE_MAX_ENTRIES = 128
WEBSEARCH_TIMEOUT_S = 20
WEBSEARCH_MAX_RESPONSE_BYTES = 2_000_000
WEBSEARCH_MAX_RESULTS = 8
DUCKDUCKGO_HTML_SEARCH_URL = 'https://html.duckduckgo.com/html/'


HTML_TEXT_BLOCK_TAGS = {
    'address', 'article', 'aside', 'blockquote', 'br', 'dd', 'details', 'div',
    'dl', 'dt', 'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3',
    'h4', 'h5', 'h6', 'header', 'hr', 'li', 'main', 'nav', 'ol', 'p', 'pre',
    'section', 'table', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr', 'ul',
}


HTML_TEXT_SKIP_TAGS = {'script', 'style', 'template', 'noscript', 'svg'}


class HtmlTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip_depth = 0

    def _newline(self):
        if self.parts and self.parts[-1] != '\n':
            self.parts.append('\n')

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in HTML_TEXT_SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in HTML_TEXT_BLOCK_TAGS:
            self._newline()

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in HTML_TEXT_SKIP_TAGS:
            if self.skip_depth:
                self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag in HTML_TEXT_BLOCK_TAGS:
            self._newline()

    def handle_data(self, data):
        if self.skip_depth:
            return
        text = ' '.join(data.split())
        if text:
            self.parts.append(text)

    def text(self) -> str:
        lines = []
        current = []
        for part in self.parts:
            if part == '\n':
                if current:
                    lines.append(' '.join(current))
                    current = []
            else:
                current.append(part)
        if current:
            lines.append(' '.join(current))
        return '\n'.join(lines).strip()


def _html_to_text(html: str) -> str:
    parser = HtmlTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def _content_media_type(content_type: str) -> str:
    return content_type.split(';', 1)[0].strip().lower()


def _is_html_content_type(content_type: str) -> bool:
    return _content_media_type(content_type) in {'text/html', 'application/xhtml+xml'}


def _decode_duckduckgo_result_url(raw_href: str) -> str:
    parsed = urllib.parse.urlparse(raw_href)
    query = urllib.parse.parse_qs(parsed.query)
    return query.get('uddg', [raw_href])[0]


class DuckDuckGoResultParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._href = None
        self._text_parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != 'a' or self._href is not None:
            return
        attr_map = {name.lower(): value or '' for name, value in attrs}
        classes = attr_map.get('class', '').split()
        href = attr_map.get('href')
        if href and 'result__a' in classes:
            self._href = href
            self._text_parts = []

    def handle_data(self, data):
        if self._href is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag):
        if tag.lower() != 'a' or self._href is None:
            return
        title = ' '.join(''.join(self._text_parts).split())
        if title:
            self.results.append({
                'title': title,
                'url': _decode_duckduckgo_result_url(self._href),
            })
        self._href = None
        self._text_parts = []


def _parse_duckduckgo_results(html: str) -> list[dict]:
    parser = DuckDuckGoResultParser()
    parser.feed(html)
    parser.close()
    return parser.results


async def _fetch_url_async(url: str, cancel_check=None) -> dict:
    """GET a URL with redirect tracking, return dict with content/contentType/status/finalUrl/redirects.
    HTTP is upgraded to HTTPS. Cross-host redirects are surfaced, not followed."""
    if url.startswith('http://'):
        url = 'https://' + url[len('http://'):]
    elif not url.startswith('https://'):
        url = 'https://' + url

    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return {'error': f'invalid URL: {url}'}

    request_headers = {
        'Accept': 'text/markdown;q=1.0, text/html;q=0.9, text/plain;q=0.8, application/json;q=0.7, */*;q=0.1',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    try:
        response = await http_client.async_http_request_follow_same_host(
            'GET', url, headers_in=request_headers,
            timeout=WEBFETCH_TIMEOUT_S, max_bytes=WEBFETCH_MAX_BYTES,
            retry_max_attempts=http_client.HTTP_RETRY_MAX_ATTEMPTS,
            retry_base_delay_s=http_client.HTTP_RETRY_BASE_DELAY_S,
            retry_max_jitter_s=http_client.HTTP_RETRY_MAX_JITTER_S,
            retry_backoff_factor=http_client.HTTP_RETRY_BACKOFF_FACTOR,
            cancel_check=cancel_check)
        if response.redirect_url:
            return {'redirectUrl': response.redirect_url, 'status': response.status,
                    'finalUrl': response.url, 'error': None}
        content_type = response.header('content-type')
        body = http_client.decode_text(response.body, response.headers)
        return {'content': body, 'contentType': content_type, 'status': response.status,
                'finalUrl': response.url, 'truncated': response.truncated, 'error': None}
    except Exception as e:
        return {'error': f'fetch failed: {e}', 'finalUrl': url}


async def run_webfetch_async(
        url: str, prompt: str,
        *, cache, complete,
        cancel_event: asyncio.Event | None = None) -> str:
    if not url:
        return "Error: url is required"
    if not prompt:
        return "Error: prompt is required"
    now = time.time()
    cached = cache.get(url)
    if cached and now - cached[0] < WEBFETCH_CACHE_TTL:
        content_text, content_type, final_url, status = cached[1], cached[2], cached[3], cached[4]
        cache_hit = True
    else:
        response = await _fetch_url_async(
            url,
            cancel_check=(
                cancel_event.is_set if cancel_event is not None else None))
        if response.get('error'):
            return f"Error: {response['error']}"
        if response.get('redirectUrl'):
            return "\n".join([
                f"WebFetch redirect: HTTP {response['status']}",
                f"requested_url: {response['finalUrl']}",
                f"redirect_url: {response['redirectUrl']}",
                "Call WebFetch again with redirect_url if you want to fetch that page.",
            ])
        content_type = response['contentType']
        if _is_html_content_type(content_type):
            content_text = _html_to_text(response['content'])
        else:
            content_text = response['content']
        if response.get('truncated'):
            content_text += "\n[... page truncated at fetch limit]"
        # cap to a sane size before sending to the model
        if len(content_text) > WEBFETCH_MAX_PROMPT_CHARS:
            content_text = content_text[:WEBFETCH_MAX_PROMPT_CHARS] + "\n[... content truncated for prompt processing]"
        final_url = response['finalUrl']
        status = response['status']
        cache[url] = (now, content_text, content_type, final_url, status)
        cache_hit = False

    msgs = [
        formats.instruction_item(
            "You are processing content fetched by the WebFetch tool. "
            "Answer only from the fetched page content. "
            "If the content does not contain the answer, say so plainly. "
            "Keep quotes short and do not reproduce large copyrighted passages."),
        formats.message_item(
            "user",
            f"URL: {final_url}\nContent-Type: {content_type}\nPrompt: {prompt}\n\n--- Page content ---\n{content_text}"),
    ]
    answer = await complete(
        msgs,
        cancel_check=(
            cancel_event.is_set if cancel_event is not None else None),
    ) or "(no answer returned)"
    header = f"[WebFetch status={status} cache_hit={cache_hit} bytes~={len(content_text)} url={final_url}]"
    return f"{header}\n{answer}"


async def run_websearch_async(query: str, allowed_domains: list | None = None,
                              blocked_domains: list | None = None,
                              cancel_event: asyncio.Event | None = None) -> str:
    if not query or len(query) < 2:
        return "Error: query must be at least 2 characters"
    if allowed_domains and blocked_domains:
        return "Error: allowed_domains and blocked_domains cannot both be specified"
    if allowed_domains and len(allowed_domains) > 20:
        return "Error: allowed_domains max 20 entries"
    if blocked_domains and len(blocked_domains) > 20:
        return "Error: blocked_domains max 20 entries"

    form = urllib.parse.urlencode({'q': query, 'b': '', 'kl': 'us-en'})
    try:
        response = await http_client.async_http_request(
            'POST',
            DUCKDUCKGO_HTML_SEARCH_URL,
            body=form.encode('utf-8'),
            headers_in={'Content-Type': 'application/x-www-form-urlencoded',
                        'Accept': 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.1'},
            timeout=WEBSEARCH_TIMEOUT_S,
            max_bytes=WEBSEARCH_MAX_RESPONSE_BYTES,
            retry_max_attempts=http_client.HTTP_RETRY_MAX_ATTEMPTS,
            retry_base_delay_s=http_client.HTTP_RETRY_BASE_DELAY_S,
            retry_max_jitter_s=http_client.HTTP_RETRY_MAX_JITTER_S,
            retry_backoff_factor=http_client.HTTP_RETRY_BACKOFF_FACTOR,
            cancel_check=(
                cancel_event.is_set if cancel_event is not None else None),
        )
        if response.status >= 400:
            return f"Error: web search request failed: HTTP {response.status} {response.reason}"
        html = http_client.decode_text(response.body, response.headers)
    except Exception as e:
        return f"Error: web search request failed: {e}"

    results = []
    for result in _parse_duckduckgo_results(html):
        target = result['url']
        host = urllib.parse.urlparse(target).netloc.lower()
        if allowed_domains and not any(host == d.lower() or host.endswith('.' + d.lower())
                                       for d in allowed_domains):
            continue
        if blocked_domains and (any(host == d.lower() or host.endswith('.' + d.lower())
                                    for d in blocked_domains)):
            continue
        results.append(result)
        if len(results) >= WEBSEARCH_MAX_RESULTS:
            break

    if not results:
        return f"No search results for query {query!r}"

    out_lines = [f"WebSearch results for {query!r} ({len(results)} results):"]
    for i, r in enumerate(results, start=1):
        out_lines.append(f"{i}. {r['title']}\n   {r['url']}")

    return "\n".join(out_lines)
