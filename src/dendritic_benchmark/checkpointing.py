from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


IgnoreKey = Callable[[str], bool]


def is_ignorable_state_key(key: str) -> bool:
    """Return whether ``key`` is runtime bookkeeping rather than model state."""
    return key.endswith("tracker_string")


def _tensor_shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(shape)
    except TypeError:
        return None


@dataclass(frozen=True)
class ShapeMismatch:
    key: str
    source_shape: tuple[int, ...] | None
    target_shape: tuple[int, ...] | None


@dataclass(frozen=True)
class CheckpointLoadReport:
    loaded: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatches: tuple[ShapeMismatch, ...]
    ignored: tuple[str, ...]
    source_revision: str | None = None

    @property
    def complete(self) -> bool:
        return not (self.missing or self.unexpected or self.shape_mismatches)

    def summary(self, *, limit: int = 3) -> str:
        details: list[str] = []
        if self.missing:
            details.append(
                f"{len(self.missing)} missing target tensor(s): "
                + ", ".join(self.missing[:limit])
            )
        if self.unexpected:
            details.append(
                f"{len(self.unexpected)} unexpected source tensor(s): "
                + ", ".join(self.unexpected[:limit])
            )
        if self.shape_mismatches:
            rendered = [
                f"{item.key} {item.source_shape}->{item.target_shape}"
                for item in self.shape_mismatches[:limit]
            ]
            details.append(
                f"{len(self.shape_mismatches)} shape mismatch(es): "
                + ", ".join(rendered)
            )
        return "; ".join(details) if details else f"{len(self.loaded)} tensor(s) loaded"


class CheckpointMismatchError(RuntimeError):
    def __init__(
        self, report: CheckpointLoadReport, *, context: str = "checkpoint"
    ) -> None:
        self.report = report
        super().__init__(f"{context} is structurally incompatible: {report.summary()}")


def inspect_state_dict(
    target_state: Mapping[str, Any],
    source_state: Mapping[str, Any],
    *,
    ignore_key: IgnoreKey = is_ignorable_state_key,
    allowed_unexpected: Callable[[str], bool] | None = None,
    source_revision: str | None = None,
) -> CheckpointLoadReport:
    """Compare state dictionaries in both directions without mutating a model."""
    loaded: list[str] = []
    unexpected: list[str] = []
    mismatches: list[ShapeMismatch] = []
    ignored: list[str] = []

    for key, source_value in source_state.items():
        if ignore_key(key):
            ignored.append(key)
            continue
        if key not in target_state:
            if allowed_unexpected is not None and allowed_unexpected(key):
                ignored.append(key)
            else:
                unexpected.append(key)
            continue
        target_value = target_state[key]
        source_shape = _tensor_shape(source_value)
        target_shape = _tensor_shape(target_value)
        if source_shape is None or target_shape is None or source_shape != target_shape:
            mismatches.append(ShapeMismatch(key, source_shape, target_shape))
            continue
        loaded.append(key)

    missing = [
        key for key in target_state if not ignore_key(key) and key not in source_state
    ]
    return CheckpointLoadReport(
        loaded=tuple(sorted(loaded)),
        missing=tuple(sorted(missing)),
        unexpected=tuple(sorted(unexpected)),
        shape_mismatches=tuple(sorted(mismatches, key=lambda item: item.key)),
        ignored=tuple(sorted(ignored)),
        source_revision=source_revision,
    )


def load_state_dict_checked(
    model: Any,
    source_state: Mapping[str, Any],
    *,
    context: str = "checkpoint",
    source_revision: str | None = None,
) -> CheckpointLoadReport:
    """Load all scientific model state or raise before changing any tensor."""
    report = inspect_state_dict(
        model.state_dict(), source_state, source_revision=source_revision
    )
    if not report.complete:
        raise CheckpointMismatchError(report, context=context)
    # ``tracker_string`` is optional PerforatedAI runtime bookkeeping. The
    # complete bidirectional comparison above has already established that all
    # scientific tensors and buffers match.
    model.load_state_dict(dict(source_state), strict=False)
    return report
