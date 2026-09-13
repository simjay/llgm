"""Durable live-index orchestration with native encoding replaced at its adapter boundary."""

from dataclasses import replace

import pytest

from llgm.core.errors import ConfigurationError
from llgm.retrieval.base import SearchHit, corpus_fingerprint
from llgm.retrieval.colbert import ColBERTConfig
from llgm.retrieval.live import WorkspaceIndexService
from llgm.retrieval.tokenizers import DiagnosticTokenizer


class SemanticFixture:
    """An explicitly synthetic semantic ranker for service protocol tests."""

    def __init__(self, passages, *, config):
        """Bind fixture passages while retaining the production adapter constructor shape."""
        self.passages = passages

    async def search(self, query, k):
        """Prefer later passages independently of lexical term matches."""
        return [
            SearchHit(passage, 1 / rank, rank)
            for rank, passage in enumerate(reversed(self.passages), 1)
        ][:k]

    def descriptor(self):
        """Use the official adapter's fingerprint spelling to exercise hybrid compatibility."""
        return {
            "backend": "fixture-semantic",
            "corpus_fingerprint": corpus_fingerprint(self.passages),
        }


@pytest.fixture
def service(tmp_path, monkeypatch):
    """Use real chunking, BM25, fusion and persistence without loading optional native libraries."""
    monkeypatch.setattr(
        "llgm.retrieval.tokenizers.ColBERTTokenizer", lambda path: DiagnosticTokenizer()
    )
    monkeypatch.setattr("llgm.retrieval.colbert_exact.ExactColBERTRetriever", SemanticFixture)
    config = ColBERTConfig(
        checkpoint_path=tmp_path / "checkpoint",
        checkpoint_sha256="a" * 64,
        repository_revision="b" * 40,
        index_root=tmp_path / "indexes",
        index_name="unused",
    )
    return WorkspaceIndexService(tmp_path / "live", config)


def records():
    """Return two independent topics with disjoint lexical terms."""
    return [
        {
            "node_id": name,
            "turns": [{"turn_id": "t", "role": "user", "text": text}],
            "metadata": {},
            "timestamp_ms": None,
        }
        for name, text in (("a", "Atlas backups"), ("b", "Seven day retention"))
    ]


def test_persisted_generation_reopens_with_matching_bm25_and_semantic_rankings(service):
    """Hybrid results fuse a lexical winner with a distinct semantic winner across restarts."""
    prepared = service.prepare(records())
    assert prepared["descriptor"]["backend"] == "H"
    assert prepared["descriptor"]["rank_constant"] == 60
    first = service.search(prepared["index_id"], "Atlas", 12)
    assert len(first["hits"]) == 2
    assert first["hits"][0]["passage"]["refs"][0]["node_id"] == "a"
    restarted = WorkspaceIndexService(service.root, service.config)
    assert restarted.prepare(records()) == prepared
    assert restarted.search(prepared["index_id"], "Atlas", 12) == first
    changed = records()
    changed[0]["turns"].append({"turn_id": "t2", "role": "user", "text": "Updated backup policy"})
    assert service.prepare(changed)["index_id"] != prepared["index_id"]


def test_corrupt_passages_and_changed_pins_cannot_reopen(service):
    """Saved text corruption and a different encoder identity fail before search."""
    prepared = service.prepare(records())
    other = WorkspaceIndexService(
        service.root, replace(service.config, repository_revision="c" * 40)
    )
    with pytest.raises(ConfigurationError, match="identity"):
        other.search(prepared["index_id"], "Atlas", 12)
    path = service.root / prepared["index_id"] / "workspace.json"
    path.write_text(path.read_text().replace("Atlas backups", "Invented text"))
    with pytest.raises(ConfigurationError, match="checksum"):
        service.search(prepared["index_id"], "Atlas", 12)


@pytest.mark.parametrize("index_id", ["../escape", "/etc/passwd", "A" * 64, "short"])
def test_index_paths_are_content_id_scoped(service, index_id):
    """Authenticated clients cannot select paths outside the live index namespace."""
    with pytest.raises(ConfigurationError):
        service.search(index_id, "Atlas", 12)
