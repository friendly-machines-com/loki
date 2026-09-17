"""The resume approval must expose every security-relevant connection field.

A field that silently drops out of this prompt is approved without being
seen, so the partition between displayed and deliberately-undisplayed fields
is asserted against the dataclass rather than described in a docstring.
"""

import dataclasses
import unittest

from loki_agent import protocols
from loki_agent.authentications import CredentialRef
from loki_agent.connections import (
    DISPLAYED_CONNECTION_FIELDS,
    UNDISPLAYED_CONNECTION_FIELDS,
    ConnectionDescriptor,
    connection_display_fields,
)


def rendered(descriptor):
    return "\n".join(
        f"{label}: {value}"
        for label, value in connection_display_fields(descriptor))


class ApprovalCoverageTests(unittest.TestCase):
    def test_every_descriptor_field_is_classified(self):
        fields = {
            field.name for field in dataclasses.fields(ConnectionDescriptor)}
        classified = (
            DISPLAYED_CONNECTION_FIELDS
            | set(UNDISPLAYED_CONNECTION_FIELDS))
        self.assertEqual(
            fields, classified,
            "classify the new ConnectionDescriptor field as displayed or "
            "undisplayed-with-a-reason")

    def test_displayed_and_undisplayed_do_not_overlap(self):
        self.assertEqual(
            DISPLAYED_CONNECTION_FIELDS
            & set(UNDISPLAYED_CONNECTION_FIELDS),
            frozenset())

    def test_every_displayed_field_actually_renders(self):
        cases = [
            ("provider_id", {"provider_id": "MARK-provider-id"},
             "MARK-provider-id"),
            ("provider_name", {"provider_name": "MARK-provider-name"},
             "MARK-provider-name"),
            ("model", {"model": "MARK-model"}, "MARK-model"),
            ("chat_url", {"chat_url": "https://MARK-chat.invalid/v1"},
             "MARK-chat.invalid"),
            ("models_url", {"models_url": "https://MARK-models.invalid/v1"},
             "MARK-models.invalid"),
            ("protocol", {"protocol": "MARK-protocol"}, "MARK-protocol"),
            ("credential_ref",
             {"credential_ref": CredentialRef.environment("MARK_KEY")},
             "MARK_KEY"),
            ("auth_header", {"auth_header": "MARK-header"}, "MARK-header"),
            ("auth_scheme", {"auth_scheme": "MARK-scheme"}, "MARK-scheme"),
        ]
        # A new displayed field needs a rendering case here; the set below is
        # asserted, not maintained by hand.
        self.assertEqual(
            {field for field, _overrides, _marker in cases}
            | {"stream", "prompt_cache"},
            set(DISPLAYED_CONNECTION_FIELDS))
        for field, overrides, marker in cases:
            with self.subTest(field=field):
                self.assertIn(marker, rendered(self._descriptor(**overrides)))
        self.assertIn(
            "Streaming: yes", rendered(self._descriptor(stream=True)))
        self.assertIn(
            "Anthropic prompt cache: yes",
            rendered(self._descriptor(
                protocol=protocols.ANTHROPIC_MESSAGES, prompt_cache=True)))

    def _descriptor(self, **overrides):
        values = dict(
            provider_id=None,
            provider_name=None,
            model="base-model",
            chat_url="https://base.invalid/v1",
            models_url=None,
            protocol=protocols.OPENAI_CHAT,
        )
        values.update(overrides)
        return ConnectionDescriptor(**values)

    def test_credential_scheme_and_header_are_shown(self):
        descriptor = ConnectionDescriptor(
            provider_id="custom",
            provider_name=None,
            model="m",
            chat_url="https://chat.invalid/v1",
            models_url=None,
            protocol=protocols.OPENAI_CHAT,
            credential_ref=CredentialRef.environment("EXAMPLE_KEY"),
            auth_header="X-Api-Key",
            auth_scheme="custom",
        )

        text = rendered(descriptor)

        self.assertIn("Credential: EXAMPLE_KEY", text)
        self.assertIn("Credential scheme: custom", text)
        self.assertIn("Credential header: X-Api-Key", text)

    def test_credentialless_connection_says_none(self):
        descriptor = ConnectionDescriptor(
            provider_id="custom",
            provider_name=None,
            model="m",
            chat_url="http://localhost:8000/v1/chat/completions",
            models_url=None,
            protocol=protocols.OPENAI_CHAT,
        )

        self.assertIn("Authentication: none", rendered(descriptor))


if __name__ == "__main__":
    unittest.main()
