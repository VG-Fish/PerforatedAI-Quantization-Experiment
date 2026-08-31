"""Index the generated evidence trees before anything is archived or deleted.

The working directory holds tens of gigabytes of generated runs from several
experiment generations.  Deleting any of it while it is the only record of a
finding would destroy evidence; keeping all of it unlabelled makes it
impossible to tell which tree a claim came from.  This module builds the
in-between artifact: a machine-readable index of every training record that
exists on disk, what run namespace it belongs to, whether its artifact manifest
still validates, and how much space each tree occupies.

Hashing is opt-in.  A full re-hash of every ``model.pt`` costs minutes and is
only needed when a tree is about to be archived or deleted; the default pass
reads the small provenance files and stats the rest.
"""

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import ARTIFACT_MANIFEST_NAME, validate_artifact_manifest

#: Generated roots that hold evidence. ``data/`` is a re-downloadable cache and
#: is deliberately not indexed as evidence.
DEFAULT_EVIDENCE_ROOTS: tuple[str, ...] = (
    "results",
    "experiments",
    "comparison",
    "logs",
    "logs_top10",
    "logs_dynamic5",
    "archive",
)

EVIDENCE_INDEX_JSON = Path("information") / "evidence_index.json"
EVIDENCE_INDEX_MARKDOWN = Path("information") / "EVIDENCE_INDEX.md"

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ArtifactEntry:
    """One training record found on disk, with its provenance and verdict."""

    path: str
    run_namespace: str
    model_key: str
    condition_key: str
    artifact_id: str
    seed: int | None
    metric_name: str
    metric_value: float | None
    dendrite_audit_status: str
    manifest_state: str
    manifest_status: str
    manifest_reason: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "run_namespace": self.run_namespace,
            "model_key": self.model_key,
            "condition_key": self.condition_key,
            "artifact_id": self.artifact_id,
            "seed": self.seed,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "dendrite_audit_status": self.dendrite_audit_status,
            "manifest_state": self.manifest_state,
            "manifest_status": self.manifest_status,
            "manifest_reason": self.manifest_reason,
            "bytes": self.bytes,
        }


@dataclass(frozen=True)
class RootSummary:
    """Space and provenance totals for one generated root."""

    path: str
    exists: bool
    file_count: int
    bytes: int
    newest_modified: str | None
    artifact_count: int
    manifest_statuses: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "file_count": self.file_count,
            "bytes": self.bytes,
            "newest_modified": self.newest_modified,
            "artifact_count": self.artifact_count,
            "manifest_statuses": dict(sorted(self.manifest_statuses.items())),
        }


@dataclass(frozen=True)
class EvidenceIndex:
    generated_at: str
    verified: bool
    roots: list[RootSummary]
    artifacts: list[ArtifactEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "generated_at": self.generated_at,
            "manifest_hashes_verified": self.verified,
            "roots": [summary.to_dict() for summary in self.roots],
            "artifacts": [entry.to_dict() for entry in self.artifacts],
        }


def _directory_stats(root: Path) -> tuple[int, int, float | None]:
    file_count = 0
    total_bytes = 0
    newest: float | None = None
    for directory, _, filenames in os.walk(root, onerror=lambda _error: None):
        for name in filenames:
            try:
                stat = os.stat(os.path.join(directory, name))
            except OSError:
                continue
            file_count += 1
            total_bytes += stat.st_size
            if newest is None or stat.st_mtime > newest:
                newest = stat.st_mtime
    return file_count, total_bytes, newest


def _condition_bytes(condition_dir: Path) -> int:
    total = 0
    for entry in condition_dir.iterdir():
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _artifact_entry(record_path: Path, *, verify: bool) -> ArtifactEntry | None:
    record = _read_json(record_path)
    if record is None:
        return None
    condition_dir = record_path.parent
    manifest = _read_json(condition_dir / ARTIFACT_MANIFEST_NAME)
    identity = manifest.get("identity") if isinstance(manifest, dict) else None
    identity = identity if isinstance(identity, dict) else {}
    seed = identity.get("seed")
    if verify:
        verdict = validate_artifact_manifest(condition_dir)
        status, reason = verdict.status, verdict.reason
    elif manifest is None:
        status, reason = "unknown", "artifact manifest is missing"
    else:
        status, reason = "unverified", "file hashes were not checked"
    return ArtifactEntry(
        path=str(condition_dir),
        run_namespace=str(condition_dir.parent.parent),
        model_key=str(record.get("model_key", "")),
        condition_key=str(record.get("condition_key", "")),
        artifact_id=str(record.get("artifact_id", "")),
        seed=seed if isinstance(seed, int) and not isinstance(seed, bool) else None,
        metric_name=str(record.get("metric_name", "")),
        metric_value=_coerce_float(record.get("metric_value")),
        dendrite_audit_status=str(record.get("dendrite_audit_status", "unknown")),
        manifest_state=str((manifest or {}).get("state", "missing")),
        manifest_status=status,
        manifest_reason=reason,
        bytes=_condition_bytes(condition_dir),
    )


def _iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat(timespec="seconds")


def build_evidence_index(
    roots: Iterable[Path | str] = DEFAULT_EVIDENCE_ROOTS,
    *,
    verify: bool = False,
) -> EvidenceIndex:
    """Index every training record under ``roots``.

    ``verify`` re-hashes each artifact's manifest-owned files.  That is the
    check to run before archiving or deleting a tree; leave it off for a quick
    inventory.
    """
    summaries: list[RootSummary] = []
    entries: list[ArtifactEntry] = []
    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists():
            summaries.append(RootSummary(str(root), False, 0, 0, None, 0))
            continue
        file_count, total_bytes, newest = _directory_stats(root)
        root_entries = [
            entry
            for entry in (
                _artifact_entry(record_path, verify=verify)
                for record_path in sorted(root.rglob("record.json"))
            )
            if entry is not None
        ]
        statuses: dict[str, int] = {}
        for entry in root_entries:
            statuses[entry.manifest_status] = statuses.get(entry.manifest_status, 0) + 1
        summaries.append(
            RootSummary(
                path=str(root),
                exists=True,
                file_count=file_count,
                bytes=total_bytes,
                newest_modified=_iso(newest),
                artifact_count=len(root_entries),
                manifest_statuses=statuses,
            )
        )
        entries.extend(root_entries)
    return EvidenceIndex(
        generated_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        verified=verify,
        roots=summaries,
        artifacts=entries,
    )


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def _namespace_rows(index: EvidenceIndex) -> list[tuple[str, int, int, str, str]]:
    grouped: dict[str, list[ArtifactEntry]] = {}
    for entry in index.artifacts:
        grouped.setdefault(entry.run_namespace, []).append(entry)
    rows: list[tuple[str, int, int, str, str]] = []
    for namespace, group in sorted(grouped.items()):
        models = sorted({entry.model_key for entry in group if entry.model_key})
        seeds = sorted({entry.seed for entry in group if entry.seed is not None})
        rows.append(
            (
                namespace,
                len(group),
                sum(entry.bytes for entry in group),
                ", ".join(models) if models else "—",
                ", ".join(str(seed) for seed in seeds) if seeds else "unrecorded",
            )
        )
    return rows


def render_evidence_markdown(index: EvidenceIndex) -> str:
    """Render the human-readable half of the evidence index."""
    lines = [
        "<!-- generated by `uv run dqb evidence_index`; do not edit by hand -->",
        "",
        "# Evidence index",
        "",
        f"Generated {index.generated_at} from the working tree. "
        + (
            "Manifest file hashes were re-verified."
            if index.verified
            else "Manifest file hashes were **not** re-verified — rerun with "
            "`--verify` before archiving or deleting a tree."
        ),
        "",
        "This index exists so that generated runs can be archived or deleted "
        "without losing the record of what they contained. See "
        "`information/RETENTION_POLICY.md` for what may be removed and when.",
        "",
        "## Generated roots",
        "",
        "| root | files | size | newest file | training records |",
        "|---|---|---|---|---|",
    ]
    for summary in index.roots:
        if not summary.exists:
            lines.append(f"| `{summary.path}` | — | — | absent | 0 |")
            continue
        lines.append(
            f"| `{summary.path}` | {summary.file_count} | "
            f"{_human_bytes(summary.bytes)} | {summary.newest_modified or '—'} | "
            f"{summary.artifact_count} |"
        )
    total_bytes = sum(summary.bytes for summary in index.roots)
    lines.extend(
        [
            "",
            f"Total indexed: **{_human_bytes(total_bytes)}** across "
            f"{len(index.artifacts)} training records.",
            "",
            "## Run namespaces",
            "",
            "| namespace | records | size | models | seeds |",
            "|---|---|---|---|---|",
        ]
    )
    for namespace, count, size, models, seeds in _namespace_rows(index):
        lines.append(
            f"| `{namespace}` | {count} | {_human_bytes(size)} | {models} | {seeds} |"
        )
    statuses: dict[str, int] = {}
    for entry in index.artifacts:
        statuses[entry.manifest_status] = statuses.get(entry.manifest_status, 0) + 1
    lines.extend(["", "## Manifest verdicts", "", "| status | records |", "|---|---|"])
    for status, count in sorted(statuses.items()):
        lines.append(f"| `{status}` | {count} |")
    lines.extend(
        [
            "",
            "`unverified` means the record was found but its file hashes were not "
            "checked in this pass. `unknown` means the run predates the artifact "
            "manifest and cannot be made reportable without being re-run.",
            "",
            "The full machine-readable index, including every record's path, "
            "artifact id, seed, and metric, is in "
            f"`{EVIDENCE_INDEX_JSON.as_posix()}`.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_evidence_index(
    index: EvidenceIndex,
    *,
    json_path: Path = EVIDENCE_INDEX_JSON,
    markdown_path: Path = EVIDENCE_INDEX_MARKDOWN,
) -> tuple[Path, Path]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(index.to_dict(), indent=2, sort_keys=True) + "\n")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_evidence_markdown(index))
    return json_path, markdown_path


def index_roots(roots: Sequence[str] | None) -> list[str]:
    return list(roots) if roots else list(DEFAULT_EVIDENCE_ROOTS)
