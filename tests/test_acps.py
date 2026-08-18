"""ACP transport and front/worker tests.

The end-to-end test runs the real front process and its spawned worker
over pipes with the dummy provider (no network): initialize, session/new
(real subprocess spawn), session/prompt, and the reply's stopReason.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loki_agent import acps  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FramingTests(unittest.TestCase):
    def test_response_and_notification_shapes(self):
        self.assertEqual(
            acps.response(7, result={"a": 1}),
            {"jsonrpc": "2.0", "id": 7, "result": {"a": 1}})
        self.assertEqual(
            acps.response(7, error={"code": -32601, "message": "x"})["error"],
            {"code": -32601, "message": "x"})
        self.assertNotIn("result", acps.response(
            7, error={"code": -1, "message": "x"}))
        note = acps.notification("session/update", {"sessionId": "s"})
        self.assertNotIn("id", note)
        self.assertEqual(note["method"], "session/update")

    def test_read_messages_parses_lines_and_skips_blanks(self):
        import io
        fin = io.StringIO('\n{"id": 1}\n\n{"id": 2}\n')
        messages = list(acps.read_messages(fin))
        self.assertEqual([m["id"] for m in messages], [1, 2])

    def test_read_messages_rejects_non_json(self):
        import io
        with self.assertRaises(acps.TransportError):
            list(acps.read_messages(io.StringIO("not json\n")))


class QuarantineTests(unittest.TestCase):
    def test_stdout_is_reserved_for_protocol(self):
        code = (
            "import sys, os, json\n"
            "sys.path.insert(0, %r)\n"
            "from loki_agent import acps\n"
            "saved = os.dup(1)\n"
            "acps.quarantine_stdout()\n"
            "write = acps.make_writer(saved)\n"
            "write(acps.response(1, result={'ok': True}))\n"
            "print('stray output')\n"
        ) % ROOT
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=ROOT)
        lines = [l for l in proc.stdout.splitlines() if l.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]),
                         {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})


@unittest.skipUnless(hasattr(os, "fork"), "needs subprocess")
class FrontWorkerTests(unittest.TestCase):
    def _front_env(self, tmpdir):
        env = dict(os.environ)
        env.update({
            "PYTHONPATH": ROOT,
            "HOME": tmpdir,
            "XDG_CONFIG_HOME": os.path.join(tmpdir, "config"),
            "XDG_STATE_HOME": os.path.join(tmpdir, "state"),
            "TERM": "dumb",
            "LOKI_PROVIDER": "dummy",
            "LOKI_API_BASE": "http://dummy.invalid/v1",
            "LOKI_MODEL": "dummy-model",
            "LOKI_DUMMY_REPLY": "acp reply text",
        })
        return env

    def test_initialize_new_session_prompt_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = self._front_env(tmpdir)
            front = subprocess.Popen(
                [sys.executable, "-m", "loki_agent.acp_main"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, env=env, cwd=tmpdir)
            try:
                def send(message):
                    front.stdin.write(json.dumps(message) + "\n")
                    front.stdin.flush()

                def recv():
                    line = front.stdout.readline()
                    self.assertTrue(line, "front produced no message")
                    return json.loads(line)

                send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": 1}})
                reply = recv()
                self.assertEqual(reply["id"], 1)
                self.assertEqual(reply["result"]["protocolVersion"], 1)
                self.assertEqual(
                    reply["result"]["agentInfo"]["name"], "loki")

                send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
                      "params": {"cwd": tmpdir}})
                reply = recv()
                self.assertEqual(reply["id"], 2)
                session_id = reply["result"]["sessionId"]
                self.assertTrue(session_id)

                send({"jsonrpc": "2.0", "id": 3,
                      "method": "session/prompt",
                      "params": {
                          "sessionId": session_id,
                          "prompt": [{"type": "text",
                                      "text": "hello acp"}]}})
                reply = recv()
                self.assertEqual(reply["id"], 3)
                self.assertEqual(reply["result"]["stopReason"], "end_turn")
            finally:
                front.stdin.close()
                try:
                    front.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    front.kill()
                    front.wait()

    def test_unknown_session_is_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = self._front_env(tmpdir)
            front = subprocess.Popen(
                [sys.executable, "-m", "loki_agent.acp_main"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, env=env, cwd=tmpdir)
            try:
                front.stdin.write(json.dumps({
                    "jsonrpc": "2.0", "id": 9,
                    "method": "session/prompt",
                    "params": {"sessionId": "nope",
                               "prompt": [{"type": "text", "text": "x"}]},
                }) + "\n")
                front.stdin.flush()
                line = front.stdout.readline()
                reply = json.loads(line)
                self.assertEqual(reply["id"], 9)
                self.assertIn("error", reply)
            finally:
                front.stdin.close()
                try:
                    front.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    front.kill()
                    front.wait()


if __name__ == "__main__":
    unittest.main()
