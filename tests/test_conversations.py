"""Persistent topic conversations preserve evidence while bounding graph growth."""

import asyncio
import json
import sqlite3

import pytest

from llgm import LLGM, Conversation, MaintenancePolicy, SourceSpan, Workspace
from llgm.core.errors import ConfigurationError, SchemaError
from llgm.memory.evidence import Evidence
from llgm.memory.migration import copy_schema3_workspace
from llgm.models import CallableModelClient, ModelResponse
from tests.node_support import Models, ReplayFactory, finish


def router(decisions=()):
    """Control topic choices at the model boundary while using real storage and retrieval."""
    choices = iter(decisions)

    async def respond(request):
        """Continue the active topic unless the fixture explicitly asks for a split."""
        payload = json.loads(request.messages[-1].content)
        if "source_node_id" in payload:
            return ModelResponse('{"links": []}')
        choice = next(choices, "active")
        selected = payload["active_node_id"] if choice == "active" else choice
        return ModelResponse(
            json.dumps(
                {
                    "node_id": selected,
                    "reason": "Same topic" if selected else "Clearly unrelated topic",
                }
            )
        )

    return CallableModelClient(respond)


def application(workspace, *, routing=None, models=None):
    """Construct the actual application using finite local interpreter callbacks."""
    models = models or Models()
    return LLGM(
        workspace=workspace,
        main_model=models.main,
        reader_model=models.reader,
        graph_model=routing or router(),
        repl_factory=ReplayFactory(),
    )


def test_answer_persists_followups_in_one_topic_across_restart(tmp_path):
    """Unmatched follow-ups still read the active topic and preserve the original source span."""

    async def scenario():
        """Run full routing, delegate reads, synthesis and persistence without hosted calls."""
        async with Workspace.open(tmp_path) as workspace:
            app = application(workspace)
            first = await app.answer("My project uses PostgreSQL.", conversation_id="project")
            span = first.references[0]
            original = (await workspace.resolve(span)).text
            assert first.conversation_id == "project" and first.node_id
            second = await app.answer("What about backups?", conversation_id="project")
            assert second.usage["graph_calls"] == 1
            assert second.usage["reader_calls"] >= 1
            assert (
                second.usage["model_calls"]
                == second.usage["graph_calls"] + second.usage["reader_calls"] + 1
            )
            assert any(event.get("role") == "graph" for event in second.trace)
            assert second.node_id == first.node_id
            assert len(await workspace.source_ids()) == 1
            assert [t.role for t in (await workspace.source(first.node_id)).turns] == [
                "user",
                "assistant",
            ] * 2
        async with Workspace.open(tmp_path) as workspace:
            app = application(workspace)
            third = await app.answer("And that?", conversation_id="project")
            assert third.node_id == first.node_id
            assert (await workspace.resolve(span)).text == original
            assert len((await workspace.source(first.node_id)).turns) == 6
            assert third.usage["model_calls"] >= 4

    asyncio.run(scenario())


def test_clear_topic_change_creates_one_node_and_return_reuses_prior_topic(tmp_path):
    """A routing split creates one topic, while a later return reuses existing evidence."""

    async def scenario():
        """Exercise explicit model decisions through ordinary answer persistence."""
        async with Workspace.open(tmp_path) as workspace:
            app = application(workspace, routing=router([None]))
            first = await app.answer("PostgreSQL database design")
            second = await app.answer("Unrelated topic: sourdough baking")
            assert first.node_id != second.node_id
            app.graph_model = router([first.node_id])
            third = await app.answer("Return to PostgreSQL database design")
            assert third.node_id == first.node_id
            assert len(await workspace.source_ids()) == 2

    asyncio.run(scenario())


def test_large_topic_appends_do_not_rewrite_history_or_load_it_for_a_span(tmp_path):
    """A multi-megabyte topic stays one graph node with bounded metadata and turn reads."""

    async def scenario():
        """Inspect actual blob I/O around metadata paging and one canonical span read."""
        async with Workspace.open(tmp_path) as workspace:
            content = "large history " * 1500
            source = await workspace.append_conversation(
                "chat",
                Conversation.from_turns([{"role": "user", "content": content} for _ in range(100)]),
            )
            original_get = workspace.blob_store.get
            reads = []

            def recording_get(digest):
                """Measure returned blob bytes rather than assuming lazy access."""
                data = original_get(digest)
                reads.append(len(data))
                return data

            workspace.blob_store.get = recording_get
            page = await workspace.source_info(source.node_id, offset=95, limit=2)
            assert page["total_turns"] == 100 and max(reads) < 1024
            turn_id = page["turns"][0]["turn_id"]
            reads.clear()
            assert (
                await workspace.resolve(SourceSpan(source.node_id, turn_id, 0, 5))
            ).text == "large"
            assert reads == [len(content.encode())]
            await workspace.append_conversation(
                "chat",
                Conversation.from_turns([{"role": "user", "content": "uniquenewkeyword"}]),
                node_id=source.node_id,
            )
            async with await Evidence.open(workspace) as evidence:
                hits = await evidence.search("uniquenewkeyword", 10)
                assert len(hits) == 1 and hits[0].passage.refs[0].node_id == source.node_id
            assert len(await workspace.source_ids()) == 1

    asyncio.run(scenario())


def test_new_chat_messages_and_general_reply_need_no_fabricated_citations(tmp_path):
    """Role/content inputs store their exact bytes and general replies may be uncited."""

    async def reply(request):
        """Return a greeting with no claim about stored personal facts."""
        return ModelResponse(finish("Hello!"))

    async def scenario():
        """Run the real final parser in conversational mode."""
        async with Workspace.open(tmp_path) as workspace:
            models = Models()
            models.main = CallableModelClient(reply)
            app = application(workspace, models=models)
            result = await app.answer([{"role": "user", "content": "  Hello!\n"}])
            assert result.status == "completed" and result.references == ()
            turns = (await workspace.source(result.node_id)).turns
            assert [(t.role, t.text) for t in turns] == [
                ("user", "  Hello!\n"),
                ("assistant", "Hello!"),
            ]

    asyncio.run(scenario())


def test_readonly_answer_and_invalid_input_do_not_write(tmp_path):
    """Read-only and rejected requests leave conversation storage untouched."""

    async def scenario():
        """Validate public boundaries before model or storage mutation."""
        async with Workspace.open(tmp_path) as workspace:
            app = application(workspace)
            result = await app.answer("Unknown memory", remember=False)
            assert result.status == "partial" and result.node_id is None
            for value in (
                "",
                [],
                [{"role": "system", "content": "instructions"}],
                [{"role": "assistant", "content": "reply"}],
            ):
                with pytest.raises((SchemaError, ConfigurationError)):
                    await app.answer(value)
            assert await workspace.source_ids() == []

    asyncio.run(scenario())


def test_existing_schema3_requires_explicit_copy_and_preserves_citations(tmp_path):
    """A schema-3 copy adds append support without mutating original records."""

    async def scenario():
        """Construct earlier metadata and reopen only the migrated destination."""
        source, destination = tmp_path / "old", tmp_path / "new"
        async with Workspace.open(source) as workspace:
            await workspace.ingest(
                Conversation.from_turns(
                    [{"role": "user", "turn_id": "original", "content": "Exact evidence"}],
                    node_id="source",
                )
            )
        connection = sqlite3.connect(source / "metadata.sqlite3")
        connection.execute("DROP TABLE conversations")
        connection.execute("DROP TABLE source_turns")
        connection.execute("PRAGMA user_version=3")
        connection.close()
        original = (source / "metadata.sqlite3").read_bytes()
        report = await copy_schema3_workspace(source, destination)
        assert report["workspace_schema"] == 5
        assert (source / "metadata.sqlite3").read_bytes() == original
        async with Workspace.open(destination) as workspace:
            assert (await workspace.resolve(SourceSpan("source", "original", 0, 5))).text == "Exact"
            await workspace.append_conversation(
                "chat",
                Conversation.from_turns([{"role": "user", "content": "More evidence"}]),
                node_id="source",
            )
            assert len((await workspace.source("source")).turns) == 2

    asyncio.run(scenario())


def test_automatic_connections_reject_semantic_categories():
    """Automatic organization cannot reintroduce writer-assigned relationship types."""
    with pytest.raises(TypeError):
        MaintenancePolicy(allowed_relations=("contradicts",))


def test_generation_failure_keeps_user_turn_without_inventing_assistant_reply(tmp_path):
    """A failed main leaves its saved input available to a later continuation."""

    async def fail(request):
        """Fail after real delegate reads have completed."""
        from llgm.core.errors import ProviderError

        raise ProviderError("Synthetic generation failure")

    async def scenario():
        """Inspect persisted roles after the runtime reports a failed generation."""
        async with Workspace.open(tmp_path) as workspace:
            models = Models()
            models.main = CallableModelClient(fail)
            result = await application(workspace, models=models).answer(
                "Remember my PostgreSQL project"
            )
            assert result.status == "failed" and not result.answer
            assert [
                (turn.role, turn.text) for turn in (await workspace.source(result.node_id)).turns
            ] == [("user", "Remember my PostgreSQL project")]
            next_result = await application(workspace).answer("Continue that project")
            assert next_result.node_id == result.node_id

    asyncio.run(scenario())


def test_shared_workspace_serializes_conversation_turn_pairs(tmp_path):
    """Separate application objects sharing storage do not interleave a chat's turn pairs."""

    async def scenario():
        """Dispatch simultaneous calls through the public conversation API."""
        async with Workspace.open(tmp_path) as workspace:
            left, right = application(workspace), application(workspace)
            results = await asyncio.gather(
                left.answer("Database schema"), right.answer("Database backups")
            )
            assert results[0].node_id == results[1].node_id
            assert [t.role for t in (await workspace.source(results[0].node_id)).turns] == [
                "user",
                "assistant",
            ] * 2

    asyncio.run(scenario())
