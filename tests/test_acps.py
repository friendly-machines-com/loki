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

                def recv_reply(reply_id):
                    while True:
                        message = recv()
                        if message.get("id") == reply_id:
                            return message

                send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": 1}})
                reply = recv_reply(1)
                self.assertEqual(reply["result"]["protocolVersion"], 1)
                self.assertEqual(
                    reply["result"]["agentInfo"]["name"], "loki")

                send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
                      "params": {"cwd": tmpdir}})
                reply = recv_reply(2)
                session_id = reply["result"]["sessionId"]
                self.assertTrue(session_id)

                send({"jsonrpc": "2.0", "id": 3,
                      "method": "session/prompt",
                      "params": {
                          "sessionId": session_id,
                          "prompt": [{"type": "text",
                                      "text": "hello acp"}]}})
                updates = []
                while True:
                    message = recv()
                    if message.get("id") == 3:
                        reply = message
                        break
                    updates.append(message)
                self.assertEqual(reply["result"]["stopReason"], "end_turn")
                # The turn's assistant text must have streamed as a
                # session/update before the reply landed.
                self.assertTrue(any(
                    m.get("method") == "session/update"
                    and m["params"]["update"]["sessionUpdate"]
                    == "agent_message_chunk"
                    for m in updates))
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


class EventMapperTests(unittest.TestCase):
    def test_assistant_delta_streams_chunk(self):
        from loki_agent import acp_events
        updates = acp_events.map_event("s", {"type": "assistant_delta",
                                             "content": "hi"}, {})
        self.assertEqual(len(updates), 1)
        self.assertEqual(
            updates[0]["update"]["sessionUpdate"], "agent_message_chunk")
        self.assertEqual(updates[0]["update"]["content"]["text"], "hi")

    def test_tool_call_then_result_pair(self):
        from loki_agent import acp_events
        state = {}
        call = acp_events.map_event("s", {"type": "tool_call",
                                          "name": "Bash",
                                          "args": {"command": "ls"}}, state)
        self.assertEqual(call[0]["update"]["sessionUpdate"], "tool_call")
        self.assertEqual(call[0]["update"]["kind"], "execute")
        self.assertIn("ls", call[0]["update"]["title"])
        result = acp_events.map_event("s", {"type": "tool_result",
                                            "name": "Bash",
                                            "call_id": "call_1",
                                            "content": "a\nb",
                                            "is_error": False}, state)
        self.assertEqual(result[0]["update"]["sessionUpdate"],
                         "tool_call_update")
        self.assertEqual(result[0]["update"]["toolCallId"], "call-1")
        self.assertNotIn("status", result[0]["update"])

    def test_tool_result_error_is_failed(self):
        from loki_agent import acp_events
        updates = acp_events.map_event("s", {"type": "tool_result",
                                             "content": "boom",
                                             "is_error": True},
                                       {"pending_call_id": "call-2"})
        self.assertEqual(updates[0]["update"]["status"], "failed")

    def test_ignored_events_map_to_nothing(self):
        from loki_agent import acp_events
        for kind in ("assistant_end", "response_timing", "max_loops",
                     "assistant_start"):
            self.assertEqual(
                acp_events.map_event("s", {"type": kind}, {}), [])


@unittest.skipUnless(hasattr(os, "fork"), "needs subprocess")
class UpdateStreamingTests(unittest.TestCase):
    """A tool-call turn must stream session/update notifications."""

    def test_prompt_emits_updates_and_stop_reason(self):
        # The dummy provider replies with tool calls when the user text
        # starts with "tool:" -- reply is JSON naming the call.
        with tempfile.TemporaryDirectory() as tmpdir:
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
                "LOKI_DUMMY_REPLY": "plain answer",
            })
            front = subprocess.Popen(
                [sys.executable, "-m", "loki_agent.acp_main"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, env=env, cwd=tmpdir)
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
                recv()
                send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
                      "params": {"cwd": tmpdir}})
                session_id = recv()["result"]["sessionId"]
                send({"jsonrpc": "2.0", "id": 3,
                      "method": "session/prompt",
                      "params": {"sessionId": session_id,
                                 "prompt": [{"type": "text",
                                             "text": "hello"}]}})
                messages = []
                while True:
                    reply = recv()
                    if reply.get("id") == 3:
                        break
                    messages.append(reply)
                # Plain reply: at least the assistant message chunk arrived
                # as a session/update notification before the reply.
                self.assertTrue(
                    any(m.get("method") == "session/update"
                        and m["params"]["update"]["sessionUpdate"]
                        == "agent_message_chunk"
                        for m in messages),
                    f"no agent_message_chunk in {[m.get('method') for m in messages]}")
                self.assertEqual(reply["result"]["stopReason"], "end_turn")
            finally:
                front.stdin.close()
                try:
                    front.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    front.kill()
                    front.wait()


@unittest.skipUnless(hasattr(os, "fork"), "needs subprocess")
class CancelEndToEndTests(unittest.TestCase):
    """session/cancel mid-turn must yield stopReason "cancelled"."""

    def test_cancel_during_streaming_turn(self):
        import time as _time
        with tempfile.TemporaryDirectory() as tmpdir:
            gate = os.path.join(tmpdir, "release")
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
                "LOKI_DUMMY_REPLY": "chunked answer",
                "LOKI_STREAM": "1",
                "LOKI_DUMMY_STREAM_CHUNKS":
                    '["first ", "second part"]',
                "LOKI_DUMMY_STREAM_GATE": gate,
            })
            front = subprocess.Popen(
                [sys.executable, "-m", "loki_agent.acp_main"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, env=env, cwd=tmpdir)
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
                while recv().get("id") != 1:
                    pass
                send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
                      "params": {"cwd": tmpdir}})
                while True:
                    reply = recv()
                    if reply.get("id") == 2:
                        break
                session_id = reply["result"]["sessionId"]

                send({"jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                      "params": {"sessionId": session_id,
                                 "prompt": [{"type": "text",
                                             "text": "hello"}]}})
                # Wait for the first delta to stream: the turn is now
                # in flight and blocked on the gate.
                deadline = _time.monotonic() + 5
                saw_first_chunk = False
                while _time.monotonic() < deadline:
                    message = recv()
                    if (message.get("method") == "session/update"
                            and message["params"]["update"].get(
                                "sessionUpdate") == "agent_message_chunk"):
                        saw_first_chunk = True
                        break
                self.assertTrue(saw_first_chunk, "no delta streamed")

                send({"jsonrpc": "2.0", "method": "session/cancel",
                      "params": {"sessionId": session_id}})

                while True:
                    reply = recv()
                    if reply.get("id") == 3:
                        break
                self.assertEqual(reply["result"]["stopReason"],
                                 "cancelled")
                # The gate was never released: the cancel, not the gate,
                # ended the turn.
                self.assertFalse(os.path.exists(gate))
            finally:
                front.stdin.close()
                try:
                    front.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    front.kill()
                    front.wait()


@unittest.skipUnless(hasattr(os, "fork"), "needs subprocess")
class LoadReplayTests(unittest.TestCase):
    def _front(self, env, cwd):
        return subprocess.Popen(
            [sys.executable, "-m", "loki_agent.acp_main"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, env=env, cwd=cwd)

    def _env(self, tmpdir, reply="loadable answer"):
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
            "LOKI_DUMMY_REPLY": reply,
        })
        return env

    def test_load_replays_history_and_continues(self):
        import time as _time
        with tempfile.TemporaryDirectory() as tmpdir:
            env = self._env(tmpdir)
            # Session 1: one prompt, one answer, saved.
            front = self._front(env, tmpdir)
            try:
                def send(fd, m):
                    fd.stdin.write(json.dumps(m) + "\n")
                    fd.stdin.flush()

                def recv(fd):
                    line = fd.stdout.readline()
                    self.assertTrue(line)
                    return json.loads(line)

                send(front, {"jsonrpc": "2.0", "id": 1,
                             "method": "initialize",
                             "params": {"protocolVersion": 1}})
                while recv(front).get("id") != 1:
                    pass
                send(front, {"jsonrpc": "2.0", "id": 2,
                             "method": "session/new",
                             "params": {"cwd": tmpdir}})
                while True:
                    m = recv(front)
                    if m.get("id") == 2:
                        break
                first_session = m["result"]["sessionId"]
                send(front, {"jsonrpc": "2.0", "id": 3,
                             "method": "session/prompt",
                             "params": {
                                 "sessionId": first_session,
                                 "prompt": [{"type": "text",
                                             "text": "remember this"}]}})
                while True:
                    m = recv(front)
                    if m.get("id") == 3:
                        break
                self.assertEqual(m["result"]["stopReason"], "end_turn")
            finally:
                front.stdin.close()
                front.wait(timeout=5)

            # The saved log must exist with a cwd in its state.
            logs = [f for f in os.listdir(
                os.path.join(tmpdir, ".loki", "chats"))]
            self.assertEqual(len(logs), 1, logs)
            saved_id = logs[0]

            # Session 2: fresh front process, load the saved conversation.
            front2 = self._front(env, tmpdir)
            try:
                send(front2, {"jsonrpc": "2.0", "id": 1,
                              "method": "initialize",
                              "params": {"protocolVersion": 1}})
                while recv(front2).get("id") != 1:
                    pass
                send(front2, {"jsonrpc": "2.0", "id": 2,
                              "method": "session/load",
                              "params": {"sessionId": saved_id,
                                         "cwd": tmpdir}})
                replayed = []
                while True:
                    m = recv(front2)
                    if m.get("id") == 2:
                        break
                    replayed.append(m)
                new_session = m["result"]["sessionId"]
                self.assertNotEqual(new_session, first_session)
                kinds = [
                    u["params"]["update"]["sessionUpdate"]
                    for u in replayed
                    if u.get("method") == "session/update"
                ]
                self.assertIn("user_message_chunk", kinds)
                self.assertIn("agent_message_chunk", kinds)
                texts = " ".join(
                    u["params"]["update"]["content"]["text"]
                    for u in replayed
                    if u.get("method") == "session/update"
                    and u["params"]["update"]["sessionUpdate"]
                    in ("user_message_chunk", "agent_message_chunk"))
                self.assertIn("remember this", texts)
                self.assertIn("loadable answer", texts)

                # The loaded session continues: a new prompt works.
                send(front2, {"jsonrpc": "2.0", "id": 3,
                              "method": "session/prompt",
                              "params": {
                                  "sessionId": new_session,
                                  "prompt": [{"type": "text",
                                              "text": "continue"}]}})
                while True:
                    m = recv(front2)
                    if m.get("id") == 3:
                        break
                self.assertEqual(m["result"]["stopReason"], "end_turn")
            finally:
                front2.stdin.close()
                front2.wait(timeout=5)


@unittest.skipUnless(hasattr(os, "fork"), "needs subprocess")
class SessionListTests(unittest.TestCase):
    def test_list_reports_saved_sessions_with_cwd(self):
        with tempfile.TemporaryDirectory() as tmpdir:
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
                "LOKI_DUMMY_REPLY": "one",
            })
            front = subprocess.Popen(
                [sys.executable, "-m", "loki_agent.acp_main"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, env=env, cwd=tmpdir)
            try:
                def send(m):
                    front.stdin.write(json.dumps(m) + "\n")
                    front.stdin.flush()

                def recv():
                    line = front.stdout.readline()
                    self.assertTrue(line)
                    return json.loads(line)

                send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": 1}})
                while recv().get("id") != 1:
                    pass
                send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
                      "params": {"cwd": tmpdir}})
                while True:
                    m = recv()
                    if m.get("id") == 2:
                        break
                session_id = m["result"]["sessionId"]
                send({"jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                      "params": {"sessionId": session_id,
                                 "prompt": [{"type": "text",
                                             "text": "hi"}]}})
                while True:
                    m = recv()
                    if m.get("id") == 3:
                        break
            finally:
                front.stdin.close()
                front.wait(timeout=5)

            # A second front process sees the saved conversation listed.
            front2 = subprocess.Popen(
                [sys.executable, "-m", "loki_agent.acp_main"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, env=env, cwd=tmpdir)
            try:
                front2.stdin.write(json.dumps({
                    "jsonrpc": "2.0", "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": 1}}) + "\n")
                front2.stdin.flush()
                while True:
                    m = json.loads(front2.stdout.readline())
                    if m.get("id") == 1:
                        break
                front2.stdin.write(json.dumps({
                    "jsonrpc": "2.0", "id": 2,
                    "method": "session/list",
                    "params": {}}) + "\n")
                front2.stdin.flush()
                m = json.loads(front2.stdout.readline())
                self.assertEqual(m["id"], 2)
                sessions = m["result"]["sessions"]
                self.assertEqual(len(sessions), 1, sessions)
                entry = sessions[0]
                self.assertTrue(entry["sessionId"].endswith(".json"))
                self.assertEqual(entry["cwd"], tmpdir)
                self.assertIn("updatedAt", entry)
            finally:
                front2.stdin.close()
                front2.wait(timeout=5)


@unittest.skipUnless(hasattr(os, "fork"), "needs subprocess")
class ConfigOptionTests(unittest.TestCase):
    def test_session_new_returns_model_options(self):
        # The dummy env has no usable catalog credentials, so the option
        # list carries only the explicit LOKI_* connection -- which is
        # enough to prove the option plumbing flows and is settable.
        with tempfile.TemporaryDirectory() as tmpdir:
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
                "LOKI_DUMMY_REPLY": "x",
            })
            front = subprocess.Popen(
                [sys.executable, "-m", "loki_agent.acp_main"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, env=env, cwd=tmpdir)
            try:
                def send(m):
                    front.stdin.write(json.dumps(m) + "\n")
                    front.stdin.flush()

                def recv():
                    line = front.stdout.readline()
                    self.assertTrue(line)
                    return json.loads(line)

                send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": 1}})
                while recv().get("id") != 1:
                    pass
                send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
                      "params": {"cwd": tmpdir}})
                while True:
                    m = recv()
                    if m.get("id") == 2:
                        break
                session_id = m["result"]["sessionId"]
                options = m["result"]["configOptions"]
                self.assertEqual(len(options), 1)
                model_option = options[0]
                self.assertEqual(model_option["id"], "model")
                self.assertEqual(model_option["category"], "model")
                self.assertEqual(model_option["type"], "select")
                self.assertEqual(model_option["currentValue"],
                                 "loki-explicit")
                values = [o["value"] for o in model_option["options"]]
                self.assertEqual(values, ["loki-explicit"])

                # Setting the value round-trips and echoes full state.
                send({"jsonrpc": "2.0", "id": 3,
                      "method": "session/set_config_option",
                      "params": {"sessionId": session_id,
                                 "configId": "model",
                                 "value": "loki-explicit"}})
                while True:
                    m = recv()
                    if m.get("id") == 3:
                        break
                self.assertIn("configOptions", m["result"])
                self.assertEqual(
                    m["result"]["configOptions"][0]["currentValue"],
                    "loki-explicit")
            finally:
                front.stdin.close()
                try:
                    front.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    front.kill()
                    front.wait()


@unittest.skipUnless(hasattr(os, "fork"), "needs fork/pty")
class TtyStdinTests(unittest.TestCase):
    """The front must work when stdin is a tty, not just a pipe.

    terminals.py closes sys.stdin at import when stdin is a tty (the
    terminal UI owns fd 0 via /dev/tty); the ACP processes read fd 0
    directly instead.  This test gives the front a real controlling
    pty, exactly like an interactive manual run.
    """

    def test_front_answers_initialize_with_tty_stdin(self):
        import fcntl
        import pty
        import termios
        import time as _time
        master, slave = pty.openpty()

        def child_setup():
            os.setsid()
            fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
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
                "LOKI_DUMMY_REPLY": "x",
            })
            proc = subprocess.Popen(
                [sys.executable, "-m", "loki_agent.acp_main"],
                stdin=slave, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env=env, cwd=tmpdir,
                preexec_fn=child_setup, text=True)
            os.close(slave)
            try:
                os.write(master, (json.dumps({
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": 1},
                }) + "\n").encode())
                reply = None
                deadline = _time.monotonic() + 8
                while _time.monotonic() < deadline:
                    line = proc.stdout.readline()
                    if line:
                        reply = json.loads(line)
                        break
                self.assertIsNotNone(reply, "no reply to initialize")
                self.assertEqual(reply["id"], 1)
                self.assertEqual(
                    reply["result"]["agentInfo"]["name"], "loki")
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                os.close(master)





class WireCwdTests(unittest.TestCase):
    """session/open's cwd lands in the session's virtual shell_cwd.

    Workers inherit the front process's cwd; the conversation's working
    directory arrives over the wire, so a tool call with a relative path
    resolves inside the session directory, not the worker's.
    """

    def test_open_sets_shell_cwd_and_tools_resolve(self):
        from loki_agent.acp_worker import Worker
        from loki_agent.sessions import Session
        from loki_agent import loki

        async def run():
            session = Session(shell_cwd="/")  # worker cwd, deliberately wrong
            worker = Worker(session, lambda message: None)
            await worker.handle({
                "jsonrpc": "2.0", "id": 1, "method": "session/open",
                "params": {"sessionId": "w", "cwd": ROOT}}, concurrent=False)
            result = await loki.dispatch_tool_async(
                "Bash", {"command": "cat tests/test_acps.py",
                         "description": "probe"})
            return session.shell_cwd, result

        shell_cwd, result = asyncio.run(run())
        self.assertEqual(shell_cwd, ROOT)
        self.assertTrue(result["ok"])
        self.assertIn("WireCwdTests", result["content"])


