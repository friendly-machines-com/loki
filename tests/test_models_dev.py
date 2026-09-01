import asyncio
import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from loki_agent import http_client, models, protocols, terminals
from loki_agent.credentials import CredentialStore

# Minimal synthetic models.dev dataset (provider-keyed, like the real API).
DATA = {
    "zhipuai": {
        "id": "zhipuai", "name": "Zhipu AI", "npm": "@ai-sdk/openai-compatible",
        "env": ["ZHIPU_API_KEY"], "api": "https://open.bigmodel.cn/api/paas/v4",
        "models": {
            "glm-5.2": {
                "id": "glm-5.2", "name": "GLM-5.2", "family": "glm",
                "reasoning": True, "tool_call": True, "temperature": True,
                "attachment": False, "structured_output": True, "open_weights": False,
                "cost": {"input": 1.4, "output": 4.4},
            },
        },
    },
    "openrouter": {
        "id": "openrouter", "name": "OpenRouter", "npm": "@openrouter/ai-sdk-provider",
        "env": ["OPENROUTER_API_KEY"], "api": "https://openrouter.ai/api/v1",
        "models": {
            "z-ai/glm-5.2": {
                "id": "z-ai/glm-5.2", "name": "GLM-5.2", "family": "glm",
                "reasoning": True, "tool_call": True, "temperature": True,
                "attachment": True, "structured_output": True, "open_weights": False,
                "cost": {"input": 2, "output": 8},
            },
        },
    },
    "anthropic": {
        "id": "anthropic", "name": "Anthropic", "npm": "@ai-sdk/anthropic",
        "env": ["ANTHROPIC_API_KEY"], "api": "https://api.anthropic.com/v1/messages",
        "models": {
            "claude-sonnet-4-6": {
                "id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "family": "claude-sonnet",
                "reasoning": True, "tool_call": True, "temperature": True,
                "attachment": True, "structured_output": True, "open_weights": False,
                "cost": {"input": 3, "output": 15},
            },
        },
    },
}


def _groups():
    return models.build_groups(DATA)


def _credentials(**overrides):
    values = {
        "ZHIPU_API_KEY": "zhipu-key",
        "OPENROUTER_API_KEY": "openrouter-key",
        "ANTHROPIC_API_KEY": "anthropic-key",
    }
    values.update(overrides)
    return CredentialStore(values)


def _input_script(inputs):
    """Async input_fn returning scripted strings; EOFError when exhausted."""
    iterator = iter(inputs)

    async def fake(prompt=None, history=None):
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError

    return fake


def _write_text(text):
    print(text, end="")


class CatalogFetchTests(unittest.TestCase):
    def test_fetch_uses_loki_http_transport_with_catalog_bounds(self):
        response = http_client.HttpResponse(
            models.MODELS_DEV_URL,
            200,
            "OK",
            {"content-type": "application/json"},
            json.dumps(DATA).encode("utf-8"),
        )
        transport = mock.AsyncMock(return_value=response)

        with mock.patch.object(
                http_client, "async_http_request", new=transport):
            result = asyncio.run(models.fetch_models_dev())

        self.assertEqual(result, DATA)
        self.assertEqual(transport.await_args.args, (
            "GET", models.MODELS_DEV_URL))
        self.assertEqual(
            transport.await_args.kwargs["headers_in"],
            {
                "User-Agent": models.USER_AGENT,
                "Accept": "application/json",
            },
        )
        self.assertEqual(
            transport.await_args.kwargs["timeout"],
            models.MODELS_DEV_TIMEOUT_S,
        )
        self.assertEqual(
            transport.await_args.kwargs["max_bytes"],
            models.MODELS_DEV_MAX_BYTES,
        )
        self.assertEqual(
            transport.await_args.kwargs["retry_max_attempts"],
            models.MODELS_DEV_RETRY_MAX_ATTEMPTS,
        )

    def test_fetch_reads_cache_file_without_network_io(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = pathlib.Path(directory, "models.json")
            cache_path.write_text(json.dumps(DATA), encoding="utf-8")
            transport = mock.AsyncMock(
                side_effect=AssertionError("cache hit performed network I/O"))
            with mock.patch.object(
                    http_client, "async_http_request", new=transport
            ), mock.patch.object(
                    asyncio, "to_thread",
                    side_effect=AssertionError("cache read used an executor")
            ):
                result = asyncio.run(models.fetch_models_dev(
                    cache_path=str(cache_path)))

        self.assertEqual(result, DATA)
        transport.assert_not_awaited()

    def test_fetch_writes_cache_file_without_an_executor(self):
        response = http_client.HttpResponse(
            models.MODELS_DEV_URL,
            200,
            "OK",
            {"content-type": "application/json"},
            json.dumps(DATA).encode("utf-8"),
        )
        with tempfile.TemporaryDirectory() as directory:
            cache_path = pathlib.Path(directory, "models.json")
            with mock.patch.object(
                    http_client, "async_http_request",
                    new=mock.AsyncMock(return_value=response)
            ), mock.patch.object(
                    asyncio, "to_thread",
                    side_effect=AssertionError("cache write used an executor")
            ):
                result = asyncio.run(models.fetch_models_dev(
                    cache_path=str(cache_path)))
            cached = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertEqual(result, DATA)
        self.assertEqual(cached, DATA)

    def test_fetch_leaves_the_event_loop_responsive(self):
        async def scenario():
            request_started = asyncio.Event()
            release_response = asyncio.Event()
            sibling_ran = asyncio.Event()

            async def transport(*_args, **_kwargs):
                request_started.set()
                await release_response.wait()
                return http_client.HttpResponse(
                    models.MODELS_DEV_URL,
                    200,
                    "OK",
                    {"content-type": "application/json"},
                    b"{}",
                )

            with mock.patch.object(
                    http_client, "async_http_request", new=transport):
                fetch = asyncio.create_task(models.fetch_models_dev())
                await request_started.wait()
                asyncio.get_running_loop().call_soon(sibling_ran.set)
                await asyncio.wait_for(sibling_ran.wait(), timeout=0.2)
                release_response.set()
                return await fetch

        self.assertEqual(asyncio.run(scenario()), {})

    def test_fetch_rejects_http_errors_and_oversized_catalogs(self):
        responses = [
            (
                http_client.HttpResponse(
                    models.MODELS_DEV_URL,
                    503,
                    "Unavailable",
                    {},
                    b"failure",
                ),
                "HTTP 503",
            ),
            (
                http_client.HttpResponse(
                    models.MODELS_DEV_URL,
                    200,
                    "OK",
                    {},
                    b"{}",
                    truncated=True,
                ),
                "exceeds",
            ),
        ]

        for response, message in responses:
            with self.subTest(message=message), mock.patch.object(
                    http_client,
                    "async_http_request",
                    new=mock.AsyncMock(return_value=response),
            ):
                with self.assertRaisesRegex(OSError, message):
                    asyncio.run(models.fetch_models_dev())


class CatalogNormalizationTests(unittest.TestCase):
    @staticmethod
    def openai_provider(**overrides):
        provider = {
            "id": "openai",
            "name": "OpenAI",
            "npm": "@ai-sdk/openai",
            "env": ["OPENAI_API_KEY"],
            "models": {
                "gpt-test": {
                    "id": "gpt-test",
                    "name": "GPT Test",
                    "tool_call": True,
                },
            },
        }
        provider.update(overrides)
        return provider

    def test_canonical_openai_entry_gets_marked_platform_endpoint(self):
        raw = {"openai": self.openai_provider()}

        normalized = models.normalize_catalog(raw)

        self.assertNotIn("api", raw["openai"])
        self.assertEqual(
            normalized["openai"]["api"],
            "https://api.openai.com/v1",
        )
        self.assertEqual(
            models.provider_display_name("openai", normalized["openai"]),
            "OpenAI Platform API [endpoint supplied by Loki]",
        )
        self.assertIn(
            "endpoint supplied by Loki",
            models.provider_description(normalized["openai"]),
        )

    def test_openai_repair_requires_complete_exact_signature(self):
        variants = {
            "provider id": {"id": "not-openai"},
            "native package": {"npm": "@ai-sdk/openai-compatible"},
            "credential declaration": {
                "env": ["OPENAI_API_KEY", "OPENAI_ORG_ID"],
            },
        }
        for label, overrides in variants.items():
            with self.subTest(label):
                raw = {"openai": self.openai_provider(**overrides)}
                normalized = models.normalize_catalog(raw)
                self.assertNotIn("api", normalized["openai"])
                self.assertFalse(
                    models.provider_supported(normalized["openai"]))

    def test_openai_repair_requires_canonical_provider_map_key(self):
        raw = {"openai-compatible": self.openai_provider()}

        normalized = models.normalize_catalog(raw)

        self.assertIs(normalized, raw)
        self.assertNotIn("api", normalized["openai-compatible"])

    def test_noncanonical_openai_endpoint_is_preserved_but_rejected(self):
        raw = {
            "openai": self.openai_provider(
                api="https://credentials.example/v1"),
        }

        normalized = models.normalize_catalog(raw)

        self.assertEqual(
            normalized["openai"]["api"],
            "https://credentials.example/v1",
        )
        self.assertFalse(models.provider_supported(normalized["openai"]))

    def test_catalog_supplied_canonical_openai_endpoint_is_not_repaired(self):
        raw = {
            "openai": self.openai_provider(
                api="https://api.openai.com/v1"),
        }

        normalized = models.normalize_catalog(raw)

        self.assertTrue(models.provider_supported(normalized["openai"]))
        self.assertEqual(
            models.provider_display_name("openai", normalized["openai"]),
            "OpenAI",
        )

    def test_openai_access_uses_responses_and_canonical_endpoints(self):
        normalized = models.normalize_catalog({
            "openai": self.openai_provider(),
        })
        access = models.provider_access(
            normalized["openai"],
            CredentialStore({"OPENAI_API_KEY": "secret"}),
        )

        self.assertIsNotNone(access)
        self.assertEqual(access.credential_env, "OPENAI_API_KEY")
        self.assertEqual(access.protocol, protocols.OPENAI_RESPONSES)
        provider = protocols.make_provider(
            access.api_url,
            provider=access.protocol,
        )
        self.assertEqual(
            provider.chat_url,
            "https://api.openai.com/v1/responses",
        )
        self.assertEqual(
            provider.models_url,
            "https://api.openai.com/v1/models",
        )

    def test_acp_choice_marks_loki_supplied_openai_endpoint(self):
        normalized = models.normalize_catalog({
            "openai": self.openai_provider(),
        })
        groups = models.build_groups(normalized)

        choices = models.flattened_config_option_choices(
            CredentialStore({"OPENAI_API_KEY": "secret"}),
            groups=groups,
        )

        self.assertEqual(len(choices), 1)
        option, _leaf = choices[0]
        self.assertEqual(
            option["name"],
            "gpt-test (OpenAI Platform API [endpoint supplied by Loki])",
        )
        self.assertIn("endpoint supplied by Loki", option["description"])

    def test_ensure_index_normalizes_without_mutating_fetched_catalog(self):
        raw = {"openai": self.openai_provider()}
        saved = models._index_cache
        models._index_cache = None
        try:
            with mock.patch.object(
                    models, "fetch_models_dev",
                    new=mock.AsyncMock(return_value=raw)):
                data, groups = asyncio.run(models.ensure_index())
        finally:
            models._index_cache = saved

        self.assertNotIn("api", raw["openai"])
        self.assertEqual(
            data["openai"]["api"],
            "https://api.openai.com/v1",
        )
        self.assertEqual(
            [provider_id for provider_id, _, _ in groups["GPT Test"]],
            ["openai"],
        )


class GroupingTests(unittest.TestCase):
    def test_build_groups_conflates_provider_specific_ids_by_name(self):
        groups = _groups()
        self.assertEqual(len(groups), 2)
        members = groups["GLM-5.2"]
        self.assertEqual(len(members), 2)
        # Provider-specific ids are kept per member, grouped under one name.
        ids = {m["id"] for _, _, m in members}
        self.assertEqual(ids, {"glm-5.2", "z-ai/glm-5.2"})

    def test_minimal_features_intersect_across_providers(self):
        members = _groups()["GLM-5.2"]
        bits = models.minimal_feature_bits(members)
        # reasoning/tool_call/struct/temp on both; attachment on only openrouter.
        self.assertEqual(bits, (True, True, True, True, False, False))
        self.assertEqual(models.feature_names(bits), "reasoning, tools, struct, temp")

    def test_union_features_or_across_providers(self):
        members = _groups()["GLM-5.2"]
        bits = models.union_feature_bits(members)
        # attachment is on openrouter but not zhipuai, so the union includes it.
        self.assertEqual(bits, (True, True, True, True, True, False))

    def test_feature_names_empty_bits(self):
        self.assertEqual(models.feature_names((False, False, False, False, False, False)), "")


class ProtocolAndKeyTests(unittest.TestCase):
    def test_provider_supported_accepts_endpoint_and_v1_base(self):
        self.assertTrue(models.provider_supported({"api": "https://x.test/v1/chat/completions"}))
        self.assertTrue(models.provider_supported({"api": "https://x.test/v1/messages"}))
        self.assertTrue(models.provider_supported({"api": "https://x.test/v1/responses"}))
        self.assertTrue(models.provider_supported({"api": "https://x.test/v1"}))
        self.assertTrue(models.provider_supported({"api": "https://x.test/compatible-mode/v1"}))

    def test_provider_supported_rejects_non_v1_and_missing_api(self):
        self.assertFalse(models.provider_supported({"api": "https://x.test/api/paas/v4"}))
        self.assertFalse(models.provider_supported({"api": "https://x.test"}))
        self.assertFalse(models.provider_supported({}))
        self.assertFalse(models.provider_supported({"api": None}))

    def test_provider_supported_accepts_known_npm_package(self):
        # The Z.AI / Zhipu AI / Vivgrid cases: non-v1 URL but a package that
        # names a protocol directly.
        self.assertTrue(models.provider_supported({
            "api": "https://api.z.ai/api/paas/v4",
            "npm": "@ai-sdk/openai-compatible",
        }))
        self.assertTrue(models.provider_supported({
            "api": "https://open.bigmodel.cn/api/paas/v4",
            "npm": "@ai-sdk/openai-compatible",
        }))
        self.assertTrue(models.provider_supported({
            "api": "https://api.anthropic.com/v1/messages",
            "npm": "@ai-sdk/anthropic",
        }))
        self.assertTrue(models.provider_supported({
            "api": "https://api.vivgrid.com/v1",
            "npm": "@ai-sdk/openai",
        }))

    def test_provider_supported_rejects_vendor_specific_npm_without_api(self):
        # Vendor SDKs that don't name a protocol and have no usable URL are
        # still dropped.
        self.assertFalse(models.provider_supported({"npm": "@ai-sdk/togetherai"}))
        self.assertFalse(models.provider_supported({"npm": "@ai-sdk/deepinfra"}))
        self.assertFalse(models.provider_supported({
            "npm": "@openrouter/ai-sdk-provider",
        }))

    def test_filter_supported_groups_drops_unsupported_providers_and_models(self):
        # In the synthetic DATA, all three providers are now supported
        # (zhipuai via npm, openrouter via /v1 fallback, anthropic via URL).
        # Add a fourth provider that's genuinely unsupported to verify the
        # filter still drops things.
        data = dict(DATA, togetherai={
            "id": "togetherai", "name": "Together AI",
            "npm": "@ai-sdk/togetherai", "api": None,
            "models": {"glm-5.2": {"id": "glm-5.2", "name": "GLM-5.2"}},
        })
        filtered = models.filter_supported_groups(models.build_groups(data))
        self.assertEqual(set(filtered), {"GLM-5.2", "Claude Sonnet 4.6"})
        # GLM-5.2 keeps the supported providers (zhipuai, openrouter);
        # togetherai is dropped because its vendor SDK has no api URL.
        pids = sorted(pid for pid, _, _ in filtered["GLM-5.2"])
        self.assertEqual(pids, ["openrouter", "zhipuai"])
        self.assertEqual([pid for pid, _, _ in filtered["Claude Sonnet 4.6"]], ["anthropic"])

    def test_filter_supported_groups_retains_deprecated_models(self):
        data = {
            "p1": {"api": "https://x.test/v1", "models": {
                "m1": {"id": "m1", "name": "Old Model", "status": "deprecated"},
                "m2": {"id": "m2", "name": "Alive Model"},
            }},
            "p2": {"api": "https://x.test/v1", "models": {
                "m1": {"id": "m1", "name": "Old Model", "status": "deprecated"},
                "m3": {"id": "m3", "name": "Mixed Model", "status": "deprecated"},
            }},
            "p3": {"api": "https://x.test/v1", "models": {
                "m3": {"id": "m3", "name": "Mixed Model"},
            }},
        }
        groups = models.filter_supported_groups(models.build_groups(data))
        self.assertEqual(
            set(groups), {"Old Model", "Alive Model", "Mixed Model"})
        self.assertEqual(
            [pid for pid, _, _ in groups["Old Model"]], ["p1", "p2"])
        self.assertEqual(
            [pid for pid, _, _ in groups["Mixed Model"]], ["p2", "p3"])

    def test_filter_groups_uses_credentials_after_protocol_filtering(self):
        groups = models.filter_supported_groups(
            _groups(), CredentialStore({"OPENROUTER_API_KEY": "key"}))
        self.assertEqual(set(groups), {"GLM-5.2"})
        self.assertEqual(
            [pid for pid, _, _ in groups["GLM-5.2"]],
            ["openrouter"],
        )

    def test_shared_credential_keeps_every_matching_provider(self):
        data = {
            pid: {
                "name": pid,
                "env": ["SHARED_API_KEY"],
                "api": f"https://{pid}.example.test/v1",
                "models": {
                    "m": {"id": "m", "name": "Shared Model"},
                },
            }
            for pid in ("regional-a", "regional-b")
        }
        groups = models.filter_supported_groups(
            models.build_groups(data),
            CredentialStore({"SHARED_API_KEY": "key"}),
        )
        self.assertEqual(
            [pid for pid, _, _ in groups["Shared Model"]],
            ["regional-a", "regional-b"],
        )

    def test_pat_is_a_catalog_credential_candidate(self):
        entry = {
            "env": ["EXAMPLE_ACCOUNT", "EXAMPLE_PAT"],
            "api": "https://example.test/v1",
        }
        access = models.provider_access(
            entry, CredentialStore({"EXAMPLE_PAT": "pat"}))
        self.assertEqual(access.credential_env, "EXAMPLE_PAT")

    def test_protocol_label_detects_from_api_url(self):
        openai_chat = {"api": "https://x.test/v1/chat/completions", "npm": "@ai-sdk/openai"}
        anthropic = {"api": "https://x.test/v1/messages", "npm": "@ai-sdk/anthropic"}
        responses = {"api": "https://x.test/v1/responses", "npm": "@ai-sdk/openai"}
        self.assertEqual(models.protocol_label(openai_chat), "openai-chat")
        self.assertEqual(models.protocol_label(anthropic), "anthropic-messages")
        self.assertEqual(models.protocol_label(responses), "openai-responses")

    def test_protocol_label_falls_back_to_npm_then_no_api(self):
        bespoke = {"api": "https://x.test/api/paas/v4", "npm": "@ai-sdk/perplexity"}
        noapi = {"npm": "@ai-sdk/mistral"}
        neither = {}
        self.assertEqual(models.protocol_label(bespoke), "@ai-sdk/perplexity")
        self.assertEqual(models.protocol_label(noapi), "@ai-sdk/mistral")
        self.assertEqual(models.protocol_label(neither), "no-api")

    def test_provider_access_resolves_declared_credential(self):
        access = models.provider_access(
            DATA["zhipuai"], CredentialStore({"ZHIPU_API_KEY": "zk"}))
        self.assertEqual(access.credential_env, "ZHIPU_API_KEY")
        self.assertEqual(access.api_url, "https://open.bigmodel.cn/api/paas/v4")
        self.assertEqual(access.protocol, "openai_chat")

    def test_provider_access_requires_credentials_and_template_values(self):
        entry = {
            "env": ["EXAMPLE_API_KEY", "EXAMPLE_ACCOUNT"],
            "api": "https://${EXAMPLE_ACCOUNT}.example.test/v1",
            "npm": "@ai-sdk/openai-compatible",
        }
        self.assertIsNone(models.provider_access(
            entry, CredentialStore({"EXAMPLE_ACCOUNT": "acct"})))
        self.assertIsNone(models.provider_access(
            entry, CredentialStore({"EXAMPLE_API_KEY": "key"})))
        access = models.provider_access(
            entry,
            CredentialStore({
                "EXAMPLE_API_KEY": "key",
                "EXAMPLE_ACCOUNT": "acct",
            }),
        )
        self.assertEqual(access.api_url, "https://acct.example.test/v1")

        secret_url = dict(
            entry, api="https://example.test/${EXAMPLE_API_KEY}/v1")
        self.assertIsNone(models.provider_access(
            secret_url,
            CredentialStore({
                "EXAMPLE_API_KEY": "key",
                "EXAMPLE_ACCOUNT": "acct",
            }),
        ))


class MenuTests(unittest.TestCase):
    def test_menu_selects_by_number(self):
        rows = [("a", "Alpha"), ("b", "Beta")]
        result = asyncio.run(models._numbered_menu_async(
            rows, "Choice: ", _input_script(["2"]),
            text_writer=_write_text))
        self.assertEqual(result, "b")

    def test_menu_filter_narrows_then_selects(self):
        rows = [("a", "Alpha beta"), ("b", "Beta gamma")]
        result = asyncio.run(models._numbered_menu_async(
            rows, "Choice: ", _input_script(["filter alpha", "1"]),
            text_writer=_write_text))
        self.assertEqual(result, "a")

    def test_menu_empty_cancels(self):
        rows = [("a", "Alpha")]
        result = asyncio.run(models._numbered_menu_async(
            rows, "Choice: ", _input_script([""]),
            text_writer=_write_text))
        self.assertIsNone(result)

    def test_terminal_writer_neutralizes_remote_menu_row_controls(self):
        rows = [(
            "a",
            "Model \x1b]0;owned\x07\n"
            "\u6a21\u578b \U0001f469\u200d\U0001f4bb",
        )]
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = asyncio.run(models._numbered_menu_async(
                rows,
                "Choice: ",
                _input_script(["1"]),
                text_writer=terminals.terminal.write_text,
            ))

        self.assertEqual(result, "a")
        self.assertEqual(
            output.getvalue(),
            "1. Model ^[]0;owned^G^J"
            "\u6a21\u578b \U0001f469\u200d\U0001f4bb\n",
        )

    def test_menu_header_separates_every_rendered_block(self):
        rows = [("a", "Alpha"), ("b", "Beta")]
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = asyncio.run(models._numbered_menu_async(
                rows,
                "Choice: ",
                _input_script(["filter beta", "1"]),
                text_writer=_write_text,
                header="Usable things:",
            ))

        self.assertEqual(result, "b")
        rendered = output.getvalue()
        self.assertTrue(rendered.startswith("\nUsable things:\n"))
        self.assertEqual(rendered.count("\nUsable things:\n"), 2)

    def test_model_rows_show_provider_snippet(self):
        rows = models._model_rows(_groups())
        glm = next(t for t in (r[1] for r in rows) if t.startswith("GLM-5.2"))
        self.assertIn("[2 providers: zhipuai, openrouter]", glm)

    def test_model_rows_show_all_providers(self):
        data = {f"p{i}": {"api": "https://x.test/v1", "models": {
            "m": {"id": f"m{i}", "name": "Big Model"}}} for i in range(8)}
        rows = models._model_rows(models.build_groups(data))
        label = rows[0][1]
        self.assertIn(
            "[8 providers: p0, p1, p2, p3, p4, p5, p6, p7]", label)
        self.assertNotIn("...", label)

    def test_explicit_connection_merges_with_catalog_model_id(self):
        explicit = models.ExplicitConnectionOption(
            model="glm-5.2",
            api_url="http://localhost:8000/v1",
            protocol="openai_chat",
        )
        groups = models._add_explicit_connection(_groups(), explicit)

        self.assertIn(explicit, groups["GLM-5.2"])
        glm_label = next(
            row[1] for row in models._model_rows(groups)
            if row[1].startswith("GLM-5.2"))
        self.assertIn(
            "[3 providers: zhipuai, openrouter, explicit LOKI_*]",
            glm_label,
        )
        provider_labels = [
            row[1] for row in models._provider_rows(groups["GLM-5.2"])]
        self.assertTrue(any(
            "Explicit LOKI_* connection id=glm-5.2" in label
            and "api=http://localhost:8000/v1" in label
            for label in provider_labels))

    def test_model_menu_filter_matches_provider_names(self):
        rows = models._model_rows(_groups())
        # "filter openrouter" must narrow to GLM-5.2 even though "openrouter"
        # appears only among its providers, not in the model name.
        result = asyncio.run(models._numbered_menu_async(
            rows, "Choice: ", _input_script(["filter openrouter", "1"]),
            text_writer=_write_text))
        self.assertEqual([pid for pid, _, _ in result], ["zhipuai", "openrouter"])

    def test_model_menu_filter_matches_provider_display_name(self):
        rows = models._model_rows(_groups())
        # "Zhipu AI" is the provider's display name, not its id (zhipuai).
        result = asyncio.run(models._numbered_menu_async(
            rows, "Choice: ", _input_script(["filter zhipu ai", "1"]),
            text_writer=_write_text))
        self.assertEqual([pid for pid, _, _ in result], ["zhipuai", "openrouter"])

    def test_provider_rows_show_catalog_api_url(self):
        rows = models._provider_rows(_groups()["GLM-5.2"])
        self.assertTrue(all(" api=https://" in row[1] for row in rows))

    def test_picker_rows_label_deprecation_at_the_right_level(self):
        groups = {
            "Old Model": [
                ("p1", {"name": "Provider One", "api": "https://p1.test/v1"},
                 {"id": "old-1", "name": "Old Model",
                  "status": "deprecated"}),
                ("p2", {"name": "Provider Two", "api": "https://p2.test/v1"},
                 {"id": "old-2", "name": "Old Model",
                  "status": "deprecated"}),
            ],
            "Mixed Model": [
                ("p1", {"name": "Provider One", "api": "https://p1.test/v1"},
                 {"id": "mixed-old", "name": "Mixed Model",
                  "status": "deprecated"}),
                ("p2", {"name": "Provider Two", "api": "https://p2.test/v1"},
                 {"id": "mixed-live", "name": "Mixed Model"}),
            ],
        }

        old_label = next(
            row[1] for row in models._model_rows(groups)
            if row[1].startswith("Old Model"))
        mixed_label = next(
            row[1] for row in models._model_rows(groups)
            if row[1].startswith("Mixed Model"))
        provider_labels = [
            row[1] for row in models._provider_rows(groups["Mixed Model"])]

        self.assertIn("Old Model (deprecated)", old_label)
        self.assertNotIn("Mixed Model (deprecated)", mixed_label)
        self.assertIn("(deprecated)", provider_labels[0])
        self.assertNotIn("(deprecated)", provider_labels[1])


class PickerTests(unittest.TestCase):
    def test_explicit_connection_is_selectable_without_catalog_models(self):
        explicit = models.ExplicitConnectionOption(
            model="private-model",
            api_url="http://localhost:8000/v1",
            protocol="openai_chat",
        )
        saved = models._index_cache
        models._index_cache = ({}, {})
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                result = asyncio.run(models.run_model_picker_async(
                    _input_script(["1", "1"]),
                    CredentialStore({}),
                    explicit_connection=explicit,
                    text_writer=_write_text,
                ))
        finally:
            models._index_cache = saved

        self.assertIs(result, explicit)
        rendered = output.getvalue()
        self.assertIn("private-model", rendered)
        self.assertIn("Explicit LOKI_* connection", rendered)
        self.assertIn("api=http://localhost:8000/v1", rendered)

    def test_outage_picker_can_select_explicit_connection(self):
        explicit = models.ExplicitConnectionOption(
            model="private-model",
            api_url="http://localhost:8000/v1",
            protocol="openai_chat",
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = asyncio.run(models.run_flat_model_picker_async(
                _input_script(["1"]),
                [],
                explicit_connection=explicit,
                text_writer=_write_text,
            ))

        self.assertIs(result, explicit)
        self.assertIn("Explicit LOKI_* connection", output.getvalue())

    def test_deprecated_model_is_selectable(self):
        data = {
            "provider": {
                "name": "Provider",
                "env": ["PROVIDER_API_KEY"],
                "api": "https://provider.test/v1",
                "models": {
                    "old": {
                        "id": "old",
                        "name": "Old Model",
                        "status": "deprecated",
                    },
                },
            },
        }
        saved = models._index_cache
        models._index_cache = (data, models.build_groups(data))
        try:
            result = asyncio.run(models.run_model_picker_async(
                _input_script(["1", "1"]),
                CredentialStore({"PROVIDER_API_KEY": "key"}),
                text_writer=_write_text,
            ))
        finally:
            models._index_cache = saved

        provider_id, _, model_entry = result
        self.assertEqual(provider_id, "provider")
        self.assertEqual(model_entry["status"], "deprecated")

    def test_run_model_picker_two_level_flow(self):
        saved = models._index_cache
        models._index_cache = (DATA, models.build_groups(DATA))
        output = io.StringIO()
        try:
            # Model menu (sorted): 1. Claude Sonnet 4.6, 2. GLM-5.2.
            # Provider menu (sorted): 1. OpenRouter, 2. Zhipu AI.
            with contextlib.redirect_stdout(output):
                result = asyncio.run(models.run_model_picker_async(
                    _input_script(["2", "1"]), _credentials(),
                    text_writer=_write_text))
        finally:
            models._index_cache = saved

        provider_id, provider_entry, model_entry = result
        self.assertEqual(provider_id, "openrouter")
        self.assertEqual(model_entry["id"], "z-ai/glm-5.2")
        rendered = output.getvalue()
        self.assertTrue(rendered.startswith("\nUsable models:\n"))
        self.assertIn("\nUsable providers:\n", rendered)
        self.assertLess(
            rendered.index("\nUsable models:\n"),
            rendered.index("\nUsable providers:\n"),
        )

    def test_run_model_picker_cancel_returns_none(self):
        saved = models._index_cache
        models._index_cache = (DATA, models.build_groups(DATA))
        try:
            result = asyncio.run(models.run_model_picker_async(
                _input_script([""]), _credentials(),
                text_writer=_write_text))
        finally:
            models._index_cache = saved
        self.assertIsNone(result)

    def test_run_model_picker_provider_cancel_returns_none(self):
        # Pick a model, then cancel the provider menu -> None, not a fallback.
        saved = models._index_cache
        models._index_cache = (DATA, models.build_groups(DATA))
        try:
            result = asyncio.run(models.run_model_picker_async(
                _input_script(["1", ""]), _credentials(),
                text_writer=_write_text))
        finally:
            models._index_cache = saved
        self.assertIsNone(result)

    def test_run_flat_model_picker_selects_by_number(self):
        result = asyncio.run(models.run_flat_model_picker_async(
            _input_script(["2"]), ["alpha", "beta"],
            text_writer=_write_text))
        self.assertEqual(result, "beta")

    def test_run_flat_model_picker_empty_list_returns_none(self):
        result = asyncio.run(models.run_flat_model_picker_async(
            _input_script([]), [], text_writer=_write_text))
        self.assertIsNone(result)

    def test_run_flat_model_picker_cancel_returns_none(self):
        result = asyncio.run(models.run_flat_model_picker_async(
            _input_script([""]), ["alpha"],
            text_writer=_write_text))
        self.assertIsNone(result)

    def test_run_model_picker_fetch_failure_propagates(self):
        # When models.dev is unreachable, the fetch exception propagates out
        # of the picker (nothing is swallowed); /model catches it and falls
        # back to the provider's own model list.
        with mock.patch("loki_agent.models.ensure_index",
                        side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                asyncio.run(models.run_model_picker_async(
                    _input_script([]), _credentials(),
                    text_writer=_write_text))

    def test_picker_prompts_advertise_filter_gesture(self):
        saved = models._index_cache
        models._index_cache = (DATA, models.build_groups(DATA))
        prompts = []

        async def capture(prompt=None, history=None):
            prompts.append(prompt or "")
            raise EOFError

        try:
            asyncio.run(models.run_model_picker_async(
                capture, _credentials(), text_writer=_write_text))
        except EOFError:
            pass
        finally:
            models._index_cache = saved

        self.assertTrue(prompts)
        self.assertIn("Model choice", prompts[0])
        self.assertIn("filter WORDS", prompts[0])
        self.assertIn("empty cancels", prompts[0])

    def test_model_rows_show_minimal_features_in_parentheses(self):
        rows = models._model_rows(_groups())
        labels = [row[1] for row in rows]
        glm = next(t for t in labels if t.startswith("GLM-5.2"))
        # Intersection shown; union (adds attachment) differs -> "[and more]".
        self.assertIn("(reasoning, tools, struct, temp)", glm)
        self.assertIn("[and more]", glm)
        self.assertIn("[2 providers: zhipuai, openrouter]", glm)
        # Cost: min/max input and output across the two providers.
        self.assertIn("cost: in $1.4-$2 per 1M tokens, out $4.4-$8 per 1M tokens", glm)
        # Single-provider group: intersection == union, no "[and more]".
        claude = next(t for t in labels if t.startswith("Claude Sonnet 4.6"))
        self.assertNotIn("[and more]", claude)

    def test_provider_rows_show_per_provider_cost(self):
        members = _groups()["GLM-5.2"]
        rows = models._provider_rows(members)
        text = {pid: t for (pid, _, _), t in rows}
        self.assertIn("cost: in $1.4 per 1M tokens, out $4.4 per 1M tokens", text["zhipuai"])
        self.assertIn("cost: in $2 per 1M tokens, out $8 per 1M tokens", text["openrouter"])

    def test_cost_helpers(self):
        with_cost = {"cost": {"input": 1.5, "output": 3.25}}
        no_cost = {"name": "free"}
        self.assertEqual(models.cost_pair(with_cost), (1.5, 3.25))
        self.assertIsNone(models.cost_pair(no_cost))
        self.assertEqual(models.cost_text(with_cost),
                         " cost: in $1.5 per 1M tokens, out $3.25 per 1M tokens")
        self.assertEqual(models.cost_text(no_cost), "")
        self.assertEqual(models.cost_range_text([("p", {}, no_cost)]), "")


class NpmProtocolDetectionTests(unittest.TestCase):
    def test_openai_compatible_npm_maps_to_openai_chat(self):
        from loki_agent import protocols
        self.assertEqual(protocols.detect_protocol_from_npm("@ai-sdk/openai-compatible"),
                         protocols.OPENAI_CHAT)

    def test_anthropic_npm_maps_to_anthropic_messages(self):
        from loki_agent import protocols
        self.assertEqual(protocols.detect_protocol_from_npm("@ai-sdk/anthropic"),
                         protocols.ANTHROPIC_MESSAGES)

    def test_openai_npm_maps_to_openai_responses(self):
        from loki_agent import protocols
        self.assertEqual(protocols.detect_protocol_from_npm("@ai-sdk/openai"),
                         protocols.OPENAI_RESPONSES)

    def test_vendor_specific_npm_returns_none(self):
        from loki_agent import protocols
        self.assertIsNone(protocols.detect_protocol_from_npm("@ai-sdk/togetherai"))
        self.assertIsNone(protocols.detect_protocol_from_npm("@openrouter/ai-sdk-provider"))
        self.assertIsNone(protocols.detect_protocol_from_npm("no-api"))
        self.assertIsNone(protocols.detect_protocol_from_npm(None))


class ConfigOptionAvailabilityTests(unittest.TestCase):
    def test_explicit_connection_survives_catalog_outage(self):
        from loki_agent import protocols

        explicit = models.ExplicitConnectionOption(
            model="local-model",
            api_url="http://localhost:8000/v1",
            protocol=protocols.OPENAI_CHAT,
        )
        with mock.patch.object(
                models, "ensure_index", side_effect=OSError("offline")):
            choices = models.flattened_config_option_choices(
                CredentialStore({}), explicit_connection=explicit)

        self.assertEqual(
            [option["value"] for option, _leaf in choices],
            ["loki-explicit"],
        )
        self.assertIs(choices[0][1], explicit)


if __name__ == "__main__":
    unittest.main()
