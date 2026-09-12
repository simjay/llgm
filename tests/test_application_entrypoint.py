"""Simple application inputs preserve the configured lifecycle and exact source records."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llgm import LLGM, Conversation, MaintenancePolicy, Settings, Turn, Workspace
from llgm.core.errors import ConfigurationError, ConflictError, SchemaError
from llgm.models import ScriptedModelClient


class OwnedClient(ScriptedModelClient):
    """Expose resource closure without opening a hosted provider connection."""

    def __init__(self):
        """Create a client with no generated responses and an observable lifetime."""
        super().__init__([])
        self.closed = False

    async def aclose(self):
        """Record release by the application context."""
        self.closed = True


class ApplicationEntryPointTests(unittest.IsolatedAsyncioTestCase):
    """Exercise convenient public inputs against real local persistence."""

    async def asyncSetUp(self):
        """Open independent storage and response-free model clients for each contract."""
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.workspace = await Workspace.open(self.path / "inputs").__aenter__()
        self.model = ScriptedModelClient([])
        self.app = LLGM(
            self.workspace,
            self.model,
            self.model,
            graph_model=None,
            maintenance_policy=MaintenancePolicy(mode="disabled"),
        )

    async def asyncTearDown(self):
        """Close storage before removing temporary source files."""
        await self.workspace.close()
        self.temp.cleanup()

    async def test_text_preserves_empty_whitespace_and_unicode_as_one_user_turn(self):
        """Plain text retains exactly the same bytes and role as an explicit user turn."""
        for text in ("", " \n\t ", "  e\u0301 / é 👩🏽‍💻\r\n終  "):
            with self.subTest(text=text):
                outcome = await self.app.ingest(text, organize=False)
                source = await self.workspace.source(outcome.source.node_id)
                self.assertEqual((source.turns[-1].role, source.turns[-1].text), ("user", text))
                self.assertEqual(source.metadata, {})
                self.assertIsNone(source.timestamp_ms)
        self.assertEqual(self.model.requests, [])

    async def test_chat_sequences_preserve_order_roles_content_and_explicit_turn_ids(self):
        """Lists and tuples accept standard chat mappings together with typed turns."""
        turns = [
            {"role": "user", "content": "  Original question?\n"},
            {"turn_id": "reply", "role": "assistant", "text": "Exact answer."},
            Turn("followup", "user", "Another question?"),
        ]
        expected = Conversation.from_turns(turns)
        for sequence in (turns, tuple(turns)):
            with self.subTest(container=type(sequence).__name__):
                outcome = await self.app.ingest(sequence, organize=False)
                source = await self.workspace.source(outcome.source.node_id)
                self.assertEqual(
                    [(t.role, t.text) for t in source.turns[-3:]],
                    [(t.role, t.text) for t in expected.turns],
                )

    async def test_conversation_preserves_source_identity_metadata_and_timestamp(self):
        """Existing Conversation inputs retain every explicit source field."""
        original = Conversation.from_turns(
            [{"role": "assistant", "content": "The original source."}],
            node_id="identified-source",
            metadata={"origin": "application", "tags": ["original"]},
            timestamp_ms=1_700_000_000_000,
        )
        outcome = await self.app.ingest(original, organize=False)
        source = await self.workspace.source(outcome.source.node_id)
        self.assertEqual(source.node_id, original.node_id)
        self.assertEqual(source.turns, original.turns)
        self.assertEqual(source.metadata, original.metadata)
        self.assertEqual(source.timestamp_ms, original.timestamp_ms)

    async def test_equivalent_input_forms_share_idempotency_and_reject_changed_text(self):
        """Normalization retains retry identity across text, chat turns, and Conversation."""
        text = "  Four days in Chicago.\n"
        turns = [{"role": "user", "content": text}]
        first = await self.app.ingest(text, idempotency_key="trip", organize=False)
        for retry in (turns, tuple(turns), Conversation.from_turns(turns)):
            outcome = await self.app.ingest(retry, idempotency_key="trip", organize=False)
            self.assertEqual(outcome.source.node_id, first.source.node_id)
            self.assertFalse(outcome.source.created)
        with self.assertRaises(ConflictError):
            await self.app.ingest("Five days in Chicago.", idempotency_key="trip", organize=False)
        self.assertEqual(await self.workspace.source_ids(), [first.source.node_id])

    async def test_invalid_inputs_fail_before_publishing_sources_or_calling_models(self):
        """Unsupported containers and malformed chat turns cannot partially ingest evidence."""
        invalid = (
            None,
            42,
            b"source text",
            bytearray(b"source text"),
            {"role": "user", "content": "A bare mapping is not a conversation."},
            [],
            ["A string is not a chat turn."],
            [Turn("valid", "user", "Valid first turn"), None],
            [{"role": "user", "content": 42}],
            [{"content": "Missing role"}],
            [{"role": "user", "text": "One value", "content": "A different value"}],
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(SchemaError):
                    await self.app.ingest(value)
                self.assertEqual(await self.workspace.source_ids(), [])
                self.assertEqual(self.model.requests, [])

    async def test_default_settings_capture_environment_on_context_entry(self):
        """Deferred context entry selects current settings once and closes owned resources."""
        clients = [OwnedClient(), OwnedClient(), OwnedClient()]
        environment = {
            "LLGM_WORKSPACE_PATH": str(self.path / "configured"),
            "LLGM_MAIN_MODEL": "before-entry",
            "LLGM_READER_MODEL": "before-entry",
            "LLGM_GRAPH_MODEL": "maintainer-at-entry",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("llgm.models.create_model", side_effect=clients) as create,
        ):
            context = LLGM.from_settings()
            self.assertFalse((self.path / "configured").exists())
            os.environ["LLGM_MAIN_MODEL"] = "main-at-entry"
            os.environ["LLGM_READER_MODEL"] = "reader-at-entry"
            async with context as memory:
                os.environ["LLGM_MAIN_MODEL"] = "changed-after-entry"
                await memory.ingest("Persisted through the simple entry point.", organize=False)
                self.assertEqual(create.call_args_list[0].args, ("openai", "main-at-entry"))
                self.assertEqual(create.call_args_list[1].args, ("openai", "reader-at-entry"))
        self.assertTrue(all(client.closed for client in clients))
        with self.assertRaises(ConfigurationError):
            await memory.workspace.sources()
        async with Workspace.open(self.path / "configured") as reopened:
            self.assertEqual(len(await reopened.source_ids()), 1)

    async def test_explicit_settings_bypass_process_environment(self):
        """Explicit settings remain usable even when unrelated environment settings are invalid."""
        clients = [OwnedClient(), OwnedClient(), OwnedClient()]
        settings = Settings(
            workspace_path=str(self.path / "explicit"),
            main_model="explicit-main",
            reader_model="explicit-reader",
            graph_model="explicit-maintainer",
            graph_provider="openai_compatible",
            graph_base_url="https://maintenance.example/v1",
            graph_api_key_env="MAINTENANCE_KEY",
        )
        with (
            patch.dict(os.environ, {"LLGM_UNKNOWN_SETTING": "invalid"}, clear=True),
            patch("llgm.models.create_model", side_effect=clients) as create,
        ):
            async with LLGM.from_settings(settings) as memory:
                await memory.ingest("Explicit configuration.", organize=False)
                self.assertEqual(create.call_args_list[0].args, ("openai", "explicit-main"))
                self.assertEqual(create.call_args_list[1].args, ("openai", "explicit-reader"))
                self.assertEqual(
                    create.call_args_list[2].args, ("openai_compatible", "explicit-maintainer")
                )
                self.assertEqual(
                    create.call_args_list[2].kwargs["base_url"], "https://maintenance.example/v1"
                )
                self.assertEqual(create.call_args_list[2].kwargs["api_key_env"], "MAINTENANCE_KEY")
                self.assertIs(memory.graph_model, clients[2])

    async def test_default_settings_still_require_model_ids_before_io(self):
        """Omitting settings never invents model choices or opens unconfigured resources."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("llgm.llgm.Workspace.open") as opening,
            patch("llgm.models.create_model") as creating,
        ):
            with self.assertRaises(ConfigurationError):
                async with LLGM.from_settings():
                    self.fail("Missing model IDs must fail before context entry")
            opening.assert_not_called()
            creating.assert_not_called()

    async def test_missing_graph_model_fails_before_opening_resources(self):
        """A configured reader cannot silently substitute for an unspecified maintainer."""
        settings = Settings(main_model="main", reader_model="reader")
        with patch("llgm.llgm.Workspace.open") as opening:
            with self.assertRaisesRegex(ConfigurationError, "graph_model"):
                async with LLGM.from_settings(settings):
                    self.fail("Missing maintenance configuration must be rejected")
            opening.assert_not_called()
