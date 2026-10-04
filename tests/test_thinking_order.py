"""Thinking controls are order-independent; projection owns validity.

Setting properties in different orders must reach the same state
(/thinking budget N then mode manual == mode manual then budget N ==
one atomic statement). A joint state that no order can validate is
rejected once, at projection -- never at assignment, never silently.
"""

import unittest
from unittest import mock

from loki_agent import loki, models as modelsdev, protocols
from loki_agent.sessions import Session


def fresh_session(model="claude-opus-4-5", url="https://api.anthropic.com/v1",
                  kind=protocols.ANTHROPIC_MESSAGES, provider_id="anthropic",
                  effort=True):
    session = Session()
    session.runtime_config = loki.make_runtime_config(
        url, kind, model=model, provider_id=provider_id,
        reasoning_effort_profile=(
            modelsdev.ReasoningEffortProfile(["low", "medium", "high"])
            if effort else None))
    return session


class OrderIndependenceTests(unittest.TestCase):
    """The diamond: any order == the atomic statement."""

    def setUp(self):
        self.session = fresh_session()
        patch = mock.patch.object(loki, "_DEFAULT_SESSION", self.session)
        patch.start()
        self.addCleanup(patch.stop)

    def apply(self, session, commands):
        with mock.patch.object(loki, "_DEFAULT_SESSION", session):
            for command in commands:
                loki.thinking_command(command)

    def saved(self, session):
        return (session.reasoning_effort_preference, session.thinking_mode,
                session.thinking_budget, session.reasoning_retention)

    def test_budget_mode_and_effort_are_order_independent(self):
        atomic = fresh_session()
        self.apply(atomic, ["effort high mode manual budget 2048"])
        expected = self.saved(atomic)

        for order in [
                ["effort high", "budget 2048", "mode manual"],
                ["mode manual", "budget 2048", "effort high"],
                ["budget 2048", "effort high", "mode manual"],
                ["mode manual", "effort high", "budget 2048"]]:
            session = fresh_session()
            self.apply(session, order)
            self.assertEqual(
                self.saved(session), expected,
                msg=f"order {order} diverged from the atomic statement")

    def test_assignment_never_validates_the_joint_state(self):
        session = fresh_session()
        with mock.patch.object(loki, "_DEFAULT_SESSION", session):
            # A lone budget is a legal assignment; validity is projection's.
            loki.thinking_command("budget 2048")
            self.assertEqual(session.thinking_budget, 2048)
            loki.thinking_command("mode manual")
            self.assertEqual(session.thinking_mode, "manual")

    def test_projection_drops_but_keeps_an_unpaired_budget(self):
        # A lone budget is a legal assignment; it applies only once a
        # manual mode joins it, so projection drops it from this turn
        # (dormant, not erased) rather than rejecting -- rejection here
        # would reintroduce order dependence.
        loki.thinking_command("budget 2048")
        self.assertEqual(self.session.thinking_budget, 2048)
        captured = loki.capture_turn_settings()
        self.assertIsNone(captured.budget)
        # The atomic order reaches the same valid state.
        other = fresh_session()
        with mock.patch.object(loki, "_DEFAULT_SESSION", other):
            loki.thinking_command("mode manual budget 2048")
            captured = loki.capture_turn_settings()
            self.assertEqual(captured.budget, 2048)
            self.assertEqual(captured.mode, "manual")
        # And completing the pair from the other order reaches it too.
        loki.thinking_command("mode manual")
        captured = loki.capture_turn_settings()
        self.assertEqual(captured.budget, 2048)
        self.assertEqual(captured.mode, "manual")

    def test_no_trial_state_exists(self):
        loki.thinking_command("mode manual budget 2048")
        self.assertFalse(
            hasattr(self.session, "thinking_request"))
        captured = loki.capture_turn_settings()
        self.assertFalse(hasattr(captured, "trial"))


class ProjectionSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.session = fresh_session()
        patch = mock.patch.object(loki, "_DEFAULT_SESSION", self.session)
        patch.start()
        self.addCleanup(patch.stop)

    def test_status_shows_state_not_machinery(self):
        loki.thinking_command("mode manual budget 2048")
        text = loki.thinking_status_text()
        self.assertIn("Mode: manual", text)
        self.assertIn("2048", text)
        self.assertNotIn("Unverified model acceptance", text)
        self.assertNotIn("unknown", text)

    def test_effort_renders_the_model_default_when_unset(self):
        text = loki.thinking_status_text()
        # The profile's own default, as state -- not "unknown".
        self.assertIn("high", text)
        self.assertNotIn("Effort: unknown", text)

    def test_allowance_is_state_not_a_lecture(self):
        fresh = fresh_session()
        with mock.patch.object(loki, "_DEFAULT_SESSION", fresh):
            text = loki.thinking_status_text()
            self.assertNotIn("manual mode requires an explicit value", text)

    def test_traces_hint_is_a_fact_not_a_tutorial(self):
        self.session.reasoning_traces = "on"
        # Fresh Claude defaults to thinking off: say that fact once.
        text = loki.reasoning_traces_status_text()
        self.assertIn("thinking off", text)
        self.assertNotIn("use /thinking to change them", text)


if __name__ == "__main__":
    unittest.main()
