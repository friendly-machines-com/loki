"""Web backends retain their parsing, cache, and inference boundary."""

import asyncio
import unittest
from unittest import mock

from loki_agent import formats, http_client, web_tools


class WebToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_parses_and_caches_page_but_delegates_each_completion(self):
        response = http_client.HttpResponse(
            'https://example.test/page', 200, 'OK',
            {'content-type': 'text/html; charset=utf-8'},
            b'<h1>Heading</h1><script>hidden script</script><p>Visible page</p>')
        fetch = mock.AsyncMock(return_value=response)
        complete = mock.AsyncMock(return_value='Page answer')
        cache = {}
        cancel = asyncio.Event()
        with mock.patch.object(http_client, 'async_http_request_follow_same_host', fetch):
            first = await web_tools.run_webfetch_async(
                'http://example.test/page', 'Summarize', cache=cache,
                complete=complete, cancel_event=cancel)
            second = await web_tools.run_webfetch_async(
                'http://example.test/page', 'Another question', cache=cache,
                complete=complete, cancel_event=cancel)
        fetch.assert_awaited_once()
        self.assertEqual(fetch.await_args.args[0], 'GET')
        self.assertEqual(fetch.await_args.args[1], 'https://example.test/page')
        self.assertIn('cache_hit=False', first)
        self.assertIn('cache_hit=True', second)
        self.assertTrue(first.endswith('Page answer'))
        self.assertEqual(complete.await_count, 2)
        messages = complete.await_args.args[0]
        prompt = formats.item_text(messages[-1])
        self.assertIn('Another question', prompt)
        self.assertIn('Heading', prompt)
        self.assertIn('Visible page', prompt)
        self.assertNotIn('hidden script', prompt)
        self.assertIs(complete.await_args.kwargs['cancel_check'].__self__, cancel)

    async def test_search_decodes_result_links_and_filters_domains(self):
        response = http_client.HttpResponse(
            web_tools.DUCKDUCKGO_HTML_SEARCH_URL, 200, 'OK',
            {'content-type': 'text/html; charset=utf-8'},
            b'<a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.test%2Fa">First</a>'
            b'<a class="result__a" href="https://other.test/b">Second</a>')
        request = mock.AsyncMock(return_value=response)
        with mock.patch.object(http_client, 'async_http_request', request):
            result = await web_tools.run_websearch_async(
                'query', allowed_domains=['example.test'])
        self.assertIn('First', result)
        self.assertIn('https://example.test/a', result)
        self.assertNotIn('Second', result)
        self.assertEqual(request.await_args.args[0], 'POST')
        self.assertEqual(request.await_args.args[1], web_tools.DUCKDUCKGO_HTML_SEARCH_URL)

    def test_transport_decodes_declared_charset_without_page_interpretation(self):
        self.assertEqual(http_client.decode_text(
            b'H\x00i\x00', {'content-type': 'text/plain; charset=utf-16le'}), 'Hi')
        self.assertEqual(http_client.decode_text(
            b'plain', {'content-type': 'text/plain; charset=unknown-encoding'}), 'plain')
