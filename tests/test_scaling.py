"""Local scaling-probe accounting and index-reuse contracts without latency thresholds."""

import asyncio
import json

import pytest

from llgm.core.errors import ConfigurationError
from llgm.evaluation.scaling import run_scaling_probe


def test_probe_records_cold_warm_append_and_restart_work(tmp_path):
    """Observed index work follows changed records and unchanged restarted rankings agree."""
    output = tmp_path / "probe"
    result = asyncio.run(run_scaling_probe(output, sizes=[3, 9], source_chars=128, repeats=2))
    assert result["model_calls"] == 0
    assert result["status"] == "completed"
    assert result == json.loads((output / "summary.json").read_text())
    for measurement in result["measurements"]:
        size = measurement["source_count"]
        phases = measurement["phases"]
        assert phases["cold"]["index"]["source_records_indexed"] == size
        assert phases["cold"]["preparation_blob_gets"] == size
        assert phases["one_update"]["index"]["source_records_indexed"] == 1
        assert phases["one_update"]["preparation_blob_gets"] == 1
        for name in ("warm", "restart"):
            assert phases[name]["index"]["source_records_indexed"] == 0
            assert phases[name]["preparation_blob_gets"] == 0
        assert measurement["restart_ranking_unchanged"]
        assert measurement["storage_bytes"] > 0
        for phase in phases.values():
            assert len(phase["query_seconds"]) == 2
            assert phase["search_blob_gets"] == 0
            assert phase["canonical_read_blob_gets"] == 1
            assert (
                phase["preparation_python_peak_bytes"]
                >= phase["preparation_python_current_bytes"]
                >= 0
            )
            assert phase["retrieval"]["implementation"] == "sqlite-fts5-incremental"


@pytest.mark.parametrize("sizes", [[], [0], [True], [1, 1]])
def test_probe_rejects_invalid_sizes_before_creating_storage(tmp_path, sizes):
    """Ambiguous or invalid measurements fail before ingesting any source bytes."""
    output = tmp_path / "probe"
    with pytest.raises(ConfigurationError):
        asyncio.run(run_scaling_probe(output, sizes=sizes))
    assert not output.exists()
