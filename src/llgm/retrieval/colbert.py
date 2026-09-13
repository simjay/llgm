"""Optional adapter for the official ColBERTv2/PLAID index and search path.

Importing this module does not import ColBERT, load a model, or download files.
Call :meth:`ColBERTRetriever.build` or :meth:`ColBERTRetriever.open` explicitly
with a complete local checkpoint and a pinned official repository revision.
The checkpoint pin is a directory-content digest computed by
:func:`checkpoint_sha256`, not the digest of the downloaded archive.

The implementation follows ``colbert/indexer.py``, ``colbert/searcher.py``,
and the tokenization code in https://github.com/stanford-futuredata/ColBERT.
It deliberately provides no lexical fallback when the optional stack fails.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from llgm.core.errors import CapabilityError, ConfigurationError
from llgm.retrieval._colbert import OFFICIAL_REPOSITORY, ranked_hits, snapshot_passages
from llgm.retrieval.base import SearchHit, SearchPassage, check_passages, corpus_fingerprint

_MANIFEST_NAME = "llgm-manifest.json"
# Upstream Run is a process-wide singleton. Serialize our uses of it, and do
# not allow concurrent search to mutate the upstream search configuration.
_OFFICIAL_LOCK = threading.RLock()


def checkpoint_sha256(path: str | Path) -> str:
    """Hash sorted relative names and file contents in a local checkpoint.

    The digest is SHA256 of a compact UTF-8 JSON array whose entries are
    ``[relative_posix_name, file_sha256]``, ordered by relative filename.
    This includes tokenizer/configuration files, not only model weights.
    """
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ConfigurationError(f"ColBERT checkpoint must be a local directory: {root}")
    entries: list[list[str]] = []
    for filename in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if filename.is_file():
            digest = hashlib.sha256()
            with filename.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            entries.append([filename.relative_to(root).as_posix(), digest.hexdigest()])
    if not entries:
        raise ConfigurationError(f"ColBERT checkpoint directory is empty: {root}")
    return hashlib.sha256(
        json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ColBERTConfig:
    """Explicit, reproducible local ColBERT configuration.

    ``repository_revision`` must be the full 40-character Git commit of an
    official checkout or a VCS-installed ``colbert-ai`` distribution. An
    ordinary unpinned PyPI install cannot establish that provenance.
    Limits include the model's special tokens and the ColBERT marker.
    """

    checkpoint_path: str | Path
    checkpoint_sha256: str
    repository_revision: str
    index_root: str | Path
    index_name: str
    doc_maxlen: int = 180
    query_maxlen: int = 32
    nbits: int = 2
    ncells: int = 2
    centroid_score_threshold: float = 0.45
    ndocs: int = 1024
    nranks: int = 1
    gpus: int = 0
    index_bsize: int = 64
    kmeans_niters: int = 4
    load_index_with_mmap: bool = False

    def __post_init__(self) -> None:
        """Validate local artifact pins, index naming, and encoder/search limits."""
        missing = [
            name
            for name in ("checkpoint_path", "checkpoint_sha256", "repository_revision")
            if getattr(self, name) in (None, "")
        ]
        if missing:
            raise ConfigurationError(
                "Missing ColBERT pins: "
                + ", ".join(missing)
                + ". Supply the local released checkpoint, its directory SHA256, and the full official Git revision"
            )
        if not isinstance(self.checkpoint_path, (str, Path)):
            raise ConfigurationError(
                "ColBERT checkpoint_path must identify a local checkpoint directory"
            )
        if not isinstance(self.checkpoint_sha256, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", self.checkpoint_sha256
        ):
            raise ConfigurationError(
                "ColBERT checkpoint_sha256 must be a 64-character SHA256 digest"
            )
        if not isinstance(self.repository_revision, str) or not re.fullmatch(
            r"[0-9a-fA-F]{40}", self.repository_revision
        ):
            raise ConfigurationError(
                "ColBERT repository_revision must be a full 40-character Git commit"
            )
        if (
            not self.index_name
            or self.index_name in {".", ".."}
            or any(char in self.index_name for char in ("/", "\\", "\x00"))
        ):
            raise ConfigurationError("ColBERT index_name must be a single nonempty directory name")
        for name in ("doc_maxlen", "query_maxlen"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 3 <= value <= 512:
                raise ConfigurationError(f"ColBERT {name} must be an integer between 3 and 512")
        for name in ("ncells", "ndocs", "nranks", "index_bsize", "kmeans_niters"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"ColBERT {name} must be a positive integer")
        if isinstance(self.gpus, bool) or not isinstance(self.gpus, int) or self.gpus < 0:
            raise ConfigurationError("ColBERT gpus must be a nonnegative integer")
        if self.nbits not in {1, 2, 4, 8}:
            raise ConfigurationError("ColBERT nbits must be one of 1, 2, 4, 8")
        if not math.isfinite(self.centroid_score_threshold):
            raise ConfigurationError("ColBERT centroid_score_threshold must be finite")
        if self.load_index_with_mmap and self.gpus:
            raise ConfigurationError("ColBERT memory-mapped indexes require CPU search (gpus=0)")

    @property
    def index_path(self) -> Path:
        """Resolve the configured index name under its local index root."""
        return Path(self.index_root).expanduser().resolve() / self.index_name


def _git(directory: Path, *args: str) -> str | None:
    """Run a bounded Git metadata query, returning None when it is unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _official_url(url: str) -> bool:
    """Recognize the supported URL forms of the official ColBERT repository."""
    normalized = url.removesuffix(".git").rstrip("/").lower()
    return normalized in {
        OFFICIAL_REPOSITORY.lower(),
        "git@github.com:stanford-futuredata/colbert",
        "ssh://git@github.com/stanford-futuredata/colbert",
    }


def _check_repository_revision(revision: str) -> dict[str, str]:
    """Check provenance before importing the heavyweight optional package."""
    try:
        spec = importlib.util.find_spec("colbert")
    except (ImportError, ValueError) as exc:
        raise CapabilityError("Cannot locate the optional official ColBERT package") from exc
    if spec is None or spec.origin is None:
        raise CapabilityError(
            "ColBERTv2/PLAID is unavailable. Install the official ColBERT repository "
            "at the pinned commit with its required torch/FAISS dependencies."
        )
    package_dir = Path(spec.origin).resolve().parent
    actual_revision = _git(package_dir, "rev-parse", "HEAD")
    remote_urls = _git(package_dir, "remote", "-v") or ""
    official_remote = any(
        len(parts := line.split()) >= 2 and _official_url(parts[1])
        for line in remote_urls.splitlines()
    )
    if actual_revision and official_remote:
        if actual_revision.lower() != revision.lower():
            raise ConfigurationError(
                f"Installed ColBERT revision {actual_revision} does not match pin {revision}"
            )
        if _git(package_dir, "status", "--porcelain", "--untracked-files=no") != "":
            raise ConfigurationError("The pinned ColBERT checkout has tracked modifications")
        return {
            "repository": OFFICIAL_REPOSITORY,
            "revision": actual_revision,
            "verification": "git",
        }
    # A VCS wheel records its exact source commit in PEP 610 metadata. Check
    # that this distribution owns the package actually selected by import.
    for distribution_name in ("colbert-ai", "colbert"):
        try:
            distribution = importlib.metadata.distribution(distribution_name)
            direct_url_text = distribution.read_text("direct_url.json")
            if not direct_url_text:
                continue
            direct_url = json.loads(direct_url_text)
            owned_path = Path(distribution.locate_file("colbert/__init__.py")).resolve()
            vcs = direct_url.get("vcs_info", {})
            if owned_path != Path(spec.origin).resolve() or not _official_url(
                direct_url.get("url", "")
            ):
                continue
            actual_revision = vcs.get("commit_id", "")
            if vcs.get("vcs") != "git" or actual_revision.lower() != revision.lower():
                raise ConfigurationError(
                    "Installed ColBERT VCS revision does not match the configured pin"
                )
            return {
                "repository": OFFICIAL_REPOSITORY,
                "revision": actual_revision,
                "verification": "pep610",
            }
        except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
            continue
    raise ConfigurationError(
        "Cannot verify ColBERT against the official repository and pinned commit. "
        "Use a clean official Git checkout or install from its explicit Git commit."
    )


@contextmanager
def _offline() -> Iterator[None]:
    """Temporarily force offline loading for this process and inherited workers."""
    # Offline flags do not replace verification of a complete local checkpoint.
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    before = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _checkpoint(config: ColBERTConfig) -> Path:
    """Verify the pinned local BERT checkpoint and required tokenizer/model files."""
    path = Path(config.checkpoint_path).expanduser().resolve()
    actual = checkpoint_sha256(path)
    if actual != config.checkpoint_sha256.lower():
        raise ConfigurationError(
            f"ColBERT checkpoint checksum mismatch: expected {config.checkpoint_sha256}, got {actual}"
        )
    try:
        model_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        tokenizer_config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            "The local ColBERT checkpoint needs valid config.json and tokenizer_config.json"
        ) from exc
    if (
        model_config.get("model_type") != "bert"
        or model_config.get("auto_map")
        or tokenizer_config.get("auto_map")
    ):
        raise ConfigurationError(
            "This ColBERTv2 adapter requires the released BERT checkpoint without remote custom code"
        )
    if not (path / "vocab.txt").is_file() and not (path / "tokenizer.json").is_file():
        raise ConfigurationError(
            "The ColBERT tokenizer vocabulary must exist in the local checkpoint directory"
        )
    if not any((path / name).is_file() for name in ("pytorch_model.bin", "model.safetensors")):
        raise ConfigurationError(
            "The local ColBERT checkpoint must contain pytorch_model.bin or model.safetensors"
        )
    maximum = model_config.get("max_position_embeddings", 512)
    if config.doc_maxlen > maximum or config.query_maxlen > maximum:
        raise ConfigurationError(
            "Configured ColBERT encoder limit exceeds the checkpoint position limit"
        )
    return path


def preflight_colbert(config: ColBERTConfig) -> list[str]:
    """Report missing local assets/provenance without loading encoder weights.

    An empty list establishes static readiness only. It cannot establish
    working CUDA, FAISS kernels, compiler toolchains, or sufficient memory.
    These are exercised by an explicitly requested build/open operation.
    """
    errors: list[str] = []
    for check in (
        lambda: _checkpoint(config),
        lambda: _check_repository_revision(config.repository_revision),
    ):
        try:
            check()
        except (ConfigurationError, OSError) as exc:
            errors.append(str(exc))
    for module in ("torch", "transformers", "faiss"):
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            errors.append(
                f"Missing ColBERT dependency {module!r}; install the dependencies "
                "specified by the pinned official ColBERT revision."
            )
    return errors


class ColBERTRetriever:
    """An explicitly built/opened official PLAID retriever over a canonical corpus.

    Returned passages retain the indexed text and references. Their metadata is
    copied so callers cannot change later hits or the recorded corpus identity.
    """

    def __init__(
        self,
        *,
        config: ColBERTConfig,
        passages: tuple[SearchPassage, ...],
        tokenizer: Any,
        searcher: Any,
        repository: dict[str, str],
        build_seconds: float | None = None,
    ) -> None:
        """Bind a verified passage corpus to an initialized official searcher."""
        self.config = config
        self._passages = snapshot_passages(passages)
        self._by_pid = dict(enumerate(self._passages))
        self._fingerprint = corpus_fingerprint(self._passages)
        self.tokenizer = tokenizer
        self._searcher = searcher
        self._repository = repository
        self._build_seconds = build_seconds
        self.events: list[dict[str, Any]] = []

    @property
    def passages(self) -> tuple[SearchPassage, ...]:
        """Return canonical passages without exposing the index's owned metadata."""
        return deepcopy(self._passages)

    @classmethod
    def build(
        cls, passages: Iterable[SearchPassage], *, config: ColBERTConfig, tokenizer: Any = None
    ) -> ColBERTRetriever:
        """Build a new PLAID index. Never overwrite or implicitly reuse one.

        Indexing can require CUDA and compilation tools depending on the
        pinned official revision. Missing capabilities raise an explicit
        error. A failed build can leave an incomplete index directory. It
        has no completed LLGM manifest and cannot be opened by this adapter.
        """
        return cls._load(tuple(passages), config=config, tokenizer=tokenizer, build=True)

    @classmethod
    def open(
        cls, passages: Iterable[SearchPassage], *, config: ColBERTConfig, tokenizer: Any = None
    ) -> ColBERTRetriever:
        """Open an existing index only after checking corpus and encoder pins."""
        return cls._load(tuple(passages), config=config, tokenizer=tokenizer, build=False)

    @classmethod
    def _load(
        cls,
        passages: tuple[SearchPassage, ...],
        *,
        config: ColBERTConfig,
        tokenizer: Any,
        build: bool,
    ) -> ColBERTRetriever:
        """Validate provenance and corpus identity before building or opening PLAID."""
        if not passages:
            raise ConfigurationError("ColBERT cannot index an empty collection")
        check_passages(passages)
        index_path = config.index_path
        manifest_path = index_path / _MANIFEST_NAME
        identity = {
            "schema_version": 1,
            "corpus_fingerprint": corpus_fingerprint(passages),
            "passage_ids_in_pid_order": [passage.passage_id for passage in passages],
            "checkpoint_sha256": config.checkpoint_sha256.lower(),
            "repository_revision": config.repository_revision.lower(),
            "doc_maxlen": config.doc_maxlen,
            "query_maxlen": config.query_maxlen,
            "nbits": config.nbits,
        }
        if build:
            if index_path.exists():
                raise ConfigurationError(
                    f"Refusing to overwrite existing ColBERT index: {index_path}"
                )
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigurationError(
                    f"No completed LLGM ColBERT manifest at {manifest_path}"
                ) from exc
            if manifest.get("identity") != identity:
                raise ConfigurationError(
                    "ColBERT index corpus/order or checkpoint/revision/encoder pins do not match"
                )
        checkpoint = _checkpoint(config)
        with _OFFICIAL_LOCK, _offline():
            repository = _check_repository_revision(config.repository_revision)
            try:
                official = importlib.import_module("colbert")
                infra = importlib.import_module("colbert.infra")
                if tokenizer is None:
                    from llgm.retrieval.tokenizers import ColBERTTokenizer

                    tokenizer = ColBERTTokenizer(str(checkpoint))
                for passage in passages:
                    count = tokenizer.count(passage.text, kind="document")
                    if count > config.doc_maxlen:
                        raise ConfigurationError(
                            f"ColBERT passage {passage.passage_id!r} has {count} tokens "
                            f"including special/marker tokens; limit is {config.doc_maxlen}. "
                            "Rechunk the shared corpus; silent truncation is forbidden."
                        )
                official_config = infra.ColBERTConfig(
                    checkpoint=str(checkpoint),
                    index_root=str(index_path.parent),
                    index_path=str(index_path),
                    doc_maxlen=config.doc_maxlen,
                    query_maxlen=config.query_maxlen,
                    nbits=config.nbits,
                    ncells=config.ncells,
                    centroid_score_threshold=config.centroid_score_threshold,
                    ndocs=config.ndocs,
                    index_bsize=config.index_bsize,
                    kmeans_niters=config.kmeans_niters,
                    load_index_with_mmap=config.load_index_with_mmap,
                )
                run_config = infra.RunConfig(
                    nranks=config.nranks,
                    gpus=config.gpus,
                    root=str(index_path.parent),
                    index_root=str(index_path.parent),
                    experiment="llgm",
                    avoid_fork_if_possible=config.nranks == 1,
                )
                # Searcher chooses CUDA from total_visible_gpus rather than
                # the RunConfig.gpus selection. Refuse a misleading CPU label.
                if config.gpus == 0 and run_config.total_visible_gpus > 0:
                    raise CapabilityError(
                        "CPU ColBERT execution was requested but CUDA devices are visible. "
                        "Launch with CUDA_VISIBLE_DEVICES='' before importing torch, or "
                        "explicitly configure gpus for this experiment."
                    )
                collection = [passage.text for passage in passages]
                elapsed = None
                with infra.Run().context(run_config):
                    if build:
                        # Official tokenization is checked directly before Indexer
                        # gets any chance to truncate the document collection.
                        doc_module = importlib.import_module(
                            "colbert.modeling.tokenization.doc_tokenization"
                        )
                        doc_tokenizer = doc_module.DocTokenizer(official_config)
                        for passage in passages:
                            cls._check_tokens(
                                tokenizer,
                                doc_tokenizer.tok,
                                passage.text,
                                "document",
                                config.doc_maxlen,
                            )
                        started = time.perf_counter()
                        indexer = official.Indexer(
                            checkpoint=str(checkpoint), config=official_config, verbose=0
                        )
                        indexer.index(
                            name=config.index_name, collection=collection, overwrite=False
                        )
                        elapsed = time.perf_counter() - started
                    searcher = official.Searcher(
                        index=config.index_name,
                        checkpoint=str(checkpoint),
                        collection=collection,
                        config=official_config,
                        index_root=str(index_path.parent),
                        verbose=0,
                    )
                    for name in ("doc_maxlen", "query_maxlen", "nbits"):
                        if getattr(searcher.index_config, name) != getattr(config, name):
                            raise ConfigurationError(
                                f"Official ColBERT index {name} does not match the configured pin"
                            )
                    for passage in passages:
                        cls._check_tokens(
                            tokenizer,
                            searcher.checkpoint.doc_tokenizer.tok,
                            passage.text,
                            "document",
                            config.doc_maxlen,
                        )
                result = cls(
                    config=config,
                    passages=passages,
                    tokenizer=tokenizer,
                    searcher=searcher,
                    repository=repository,
                    build_seconds=elapsed,
                )
                if build:
                    pending_manifest = manifest_path.with_suffix(".json.tmp")
                    pending_manifest.write_text(
                        json.dumps(
                            {"identity": identity, "build": result.descriptor()},
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    pending_manifest.replace(manifest_path)
                return result
            except (CapabilityError, ConfigurationError):
                raise
            except Exception as exc:
                operation = "build/open" if build else "open"
                raise CapabilityError(
                    f"Official ColBERTv2/PLAID {operation} failed ({type(exc).__name__}: {exc}). "
                    "Check the pinned implementation's torch, FAISS, CUDA/compiler requirements "
                    "and complete local checkpoint. No fallback was used."
                ) from exc

    @staticmethod
    def _check_tokens(
        tokenizer: Any, official_tokenizer: Any, text: str, kind: str, limit: int
    ) -> int:
        """Require shared and official token counts to agree and fit the encoder."""
        # Official tensorize reserves one slot for an inserted [D]/[Q].
        # HF adds [CLS]/[SEP]; query mask padding does not consume text.
        encoded = official_tokenizer(text, add_special_tokens=True, truncation=False)
        actual = len(encoded["input_ids"]) + 1
        declared = tokenizer.count(text, kind=kind)
        if declared != actual:
            raise ConfigurationError(
                f"Shared and official ColBERT {kind} token counts disagree: {declared} versus {actual}"
            )
        if actual > limit:
            raise ConfigurationError(
                f"ColBERT {kind} contains {actual} tokens including special/marker tokens; "
                f"limit is {limit}. Silent truncation is forbidden."
            )
        return actual

    async def search(self, query: str, k: int = 10) -> list[SearchHit]:
        """Search locally, rejecting overflow before the official encoder runs."""
        if not isinstance(query, str) or not query.strip():
            raise ConfigurationError("ColBERT query must be nonempty text")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ConfigurationError("ColBERT k must be a positive integer")
        return await asyncio.to_thread(self._search, query, k)

    def _search(self, query: str, k: int) -> list[SearchHit]:
        """Serialize official search, validate returned hits, and retain timing on failure."""
        with _OFFICIAL_LOCK, _offline():
            query_tokens = self._check_tokens(
                self.tokenizer,
                self._searcher.checkpoint.query_tokenizer.tok,
                query,
                "query",
                self.config.query_maxlen,
            )
            started = time.perf_counter()
            event: dict[str, Any] = {
                "operation": "colbert_plaid_search",
                "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "query_tokens": query_tokens,
                "requested_k": k,
                "returned": 0,
                "status": "failed",
            }
            try:
                pids, ranks, scores = self._searcher.search(query, k=k, full_length_search=False)
                if not len(pids) == len(ranks) == len(scores):
                    raise CapabilityError(
                        "Official ColBERT returned mismatched passage IDs, ranks and scores"
                    )
                hits = ranked_hits(
                    [
                        {"passage_id": pid, "rank": rank, "score": score}
                        for pid, rank, score in zip(pids, ranks, scores)
                    ],
                    self._by_pid,
                    k,
                )
                event.update(returned=len(hits), status="succeeded")
                return hits
            except CapabilityError:
                raise
            except Exception as exc:
                raise CapabilityError(f"Official ColBERTv2/PLAID search failed: {exc}") from exc
            finally:
                event["elapsed_seconds"] = time.perf_counter() - started
                self.events.append(event)

    def descriptor(self) -> dict[str, Any]:
        """Serializable configuration/provenance and measured local work."""
        configuration = asdict(self.config)
        configuration["checkpoint_path"] = str(
            Path(self.config.checkpoint_path).expanduser().resolve()
        )
        configuration["index_root"] = str(self.config.index_path.parent)
        return {
            "backend": "colbertv2_plaid",
            "deployment": "local",
            "implementation": dict(self._repository),
            "configuration": configuration,
            "checkpoint_digest_kind": "sha256-sorted-relative-path-and-file-sha256-json-v1",
            "corpus_fingerprint": self._fingerprint,
            "passage_count": len(self._passages),
            "index_path": str(self.config.index_path),
            "token_limits_include_special_and_marker_tokens": True,
            "silent_truncation": False,
            "build_seconds": self._build_seconds,
            "search_calls": len(self.events),
            "search_seconds": sum(event["elapsed_seconds"] for event in self.events),
            "tokenizer": self.tokenizer.descriptor()
            if hasattr(self.tokenizer, "descriptor")
            else None,
        }
