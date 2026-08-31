import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"
ARTIFACT_MANIFEST_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactVerdict:
    status: str
    reason: str
    manifest: dict[str, Any] | None = None

    @property
    def valid(self) -> bool:
        return self.status == "verified"


def write_artifact_manifest(
    output_dir: Path,
    *,
    artifact_id: str,
    identity: dict[str, Any],
    pai_save_name: str | None,
    validity: dict[str, Any],
    telemetry: dict[str, Any] | None = None,
    additional_files: tuple[str, ...] = (),
) -> Path:
    """Atomically bind one immutable artifact identity to its owned files."""
    required_files = ("model.pt", "metrics.json", "history.csv", *additional_files)
    files = {name: {"sha256": _sha256(output_dir / name)} for name in required_files}
    payload = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "state": "core_written",
        "artifact_id": artifact_id,
        "identity": identity,
        "pai_namespace": pai_save_name,
        "files": files,
        "validity": validity,
        "telemetry": telemetry or {},
    }
    path = output_dir / ARTIFACT_MANIFEST_NAME
    temporary = output_dir / f".{ARTIFACT_MANIFEST_NAME}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)
    return path


def finalize_artifact_manifest(output_dir: Path, *, artifact_id: str) -> Path:
    """Seal record files into the manifest after the pipeline writes them."""
    path = output_dir / ARTIFACT_MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot finalize missing or invalid artifact manifest: {path}"
        ) from exc
    if manifest.get("artifact_id") != artifact_id:
        raise RuntimeError(
            "record artifact_id does not match the core artifact manifest"
        )
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("core artifact manifest has no file inventory")
    for name in ("record.json", "record.csv", "best_model_stats.csv"):
        target = output_dir / name
        if not target.is_file():
            raise RuntimeError(f"cannot finalize artifact without {target}")
        files[name] = {"sha256": _sha256(target)}
    manifest["state"] = "complete"
    temporary = output_dir / f".{ARTIFACT_MANIFEST_NAME}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(path)
    return path


def validate_artifact_manifest(
    condition_dir: Path,
    *,
    expected_artifact_id: str | None = None,
    expected_model_key: str | None = None,
    expected_condition_key: str | None = None,
) -> ArtifactVerdict:
    path = condition_dir / ARTIFACT_MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError:
        return ArtifactVerdict("unknown", "artifact manifest is missing")
    except (OSError, json.JSONDecodeError) as exc:
        return ArtifactVerdict("invalid", f"artifact manifest is unreadable: {exc}")
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        return ArtifactVerdict(
            "invalid", "artifact manifest schema is unsupported", manifest
        )
    if manifest.get("state") != "complete":
        return ArtifactVerdict(
            "unknown", "artifact manifest is not finalized", manifest
        )
    artifact_id = manifest.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        return ArtifactVerdict(
            "invalid", "artifact manifest has no artifact_id", manifest
        )
    if expected_artifact_id is not None and artifact_id != expected_artifact_id:
        return ArtifactVerdict(
            "invalid", "record and artifact manifest IDs disagree", manifest
        )
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        return ArtifactVerdict("invalid", "artifact identity is missing", manifest)
    if (
        expected_model_key is not None
        and identity.get("model_key") != expected_model_key
    ):
        return ArtifactVerdict(
            "invalid", "artifact model identity disagrees with record", manifest
        )
    if (
        expected_condition_key is not None
        and identity.get("condition_key") != expected_condition_key
    ):
        return ArtifactVerdict(
            "invalid", "artifact condition identity disagrees with record", manifest
        )
    files = manifest.get("files")
    if not isinstance(files, dict):
        return ArtifactVerdict(
            "invalid", "artifact file inventory is missing", manifest
        )
    required_files = {
        "model.pt",
        "metrics.json",
        "history.csv",
        "record.json",
        "record.csv",
        "best_model_stats.csv",
    }
    if not required_files.issubset(files):
        return ArtifactVerdict(
            "invalid", "artifact file inventory is incomplete", manifest
        )
    for name in files:
        if not isinstance(name, str) or Path(name).name != name:
            return ArtifactVerdict(
                "invalid", "artifact file inventory contains an unsafe path", manifest
            )
        entry = files.get(name)
        expected_hash = entry.get("sha256") if isinstance(entry, dict) else None
        target = condition_dir / name
        if not isinstance(expected_hash, str) or not target.is_file():
            return ArtifactVerdict(
                "invalid", f"manifest-owned file is missing: {name}", manifest
            )
        try:
            actual_hash = _sha256(target)
        except OSError as exc:
            return ArtifactVerdict(
                "invalid", f"manifest-owned file is unreadable: {name}: {exc}", manifest
            )
        if actual_hash != expected_hash:
            return ArtifactVerdict(
                "invalid", f"manifest-owned file changed: {name}", manifest
            )
    return ArtifactVerdict(
        "verified", "manifest identity and file hashes verified", manifest
    )
