import unittest

from loki_agent import sse


class SseDecoderTests(unittest.TestCase):
    def test_fragmented_utf8_multiline_data_and_comments(self):
        encoded = (
            "\ufeff: keepalive\r\n"
            "event: update\r\n"
            "id: 7\r\n"
            "data: hello\r\n"
            "data: w\u00f6rld\r\n"
            "\r\n"
        ).encode("utf-8")
        expected = [
            sse.SseEvent("update", "hello\nw\u00f6rld", "7"),
        ]

        # Every possible two-chunk boundary includes boundaries inside the
        # UTF-8 character and both bytes of CRLF.
        for split in range(len(encoded) + 1):
            with self.subTest(split=split):
                decoder = sse.SseDecoder()
                events = decoder.feed(encoded[:split])
                events.extend(decoder.feed(encoded[split:]))
                events.extend(decoder.finish())
                self.assertEqual(events, expected)

        decoder = sse.SseDecoder()
        events = []
        for byte in encoded:
            events.extend(decoder.feed(bytes([byte])))
        events.extend(decoder.finish())
        self.assertEqual(events, expected)

    def test_lf_crlf_and_cr_are_valid_line_endings(self):
        decoder = sse.SseDecoder()

        events = decoder.feed(
            b"data: one\n\ndata: two\r\rdata: three\r\n\r\n")
        events.extend(decoder.finish())

        self.assertEqual(
            [event.data for event in events],
            ["one", "two", "three"],
        )

    def test_final_event_is_dispatched_at_eof(self):
        decoder = sse.SseDecoder()

        events = decoder.feed(b"data: final")
        events.extend(decoder.finish())

        self.assertEqual(events, [sse.SseEvent("message", "final")])

    def test_comments_without_data_do_not_dispatch(self):
        decoder = sse.SseDecoder()

        events = decoder.feed(b": ping\n\n")
        events.extend(decoder.finish())

        self.assertEqual(events, [])

    def test_event_size_limit_is_enforced(self):
        decoder = sse.SseDecoder(max_event_bytes=8)

        with self.assertRaisesRegex(ValueError, "size limit"):
            decoder.feed(b"data: too long\n")


if __name__ == "__main__":
    unittest.main()
