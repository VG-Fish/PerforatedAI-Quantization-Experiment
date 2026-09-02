"""Declarative model capabilities shared by planning and training.

Architectures and dataset builders remain in their domain modules, but policy
that used to be repeated as model-key conditionals lives here.  Exploratory
models stay registered and can be selected explicitly; only the evidence-backed
roster is selected by a bare ``dqb run``.
"""

from dataclasses import dataclass
from typing import Literal

from .specs import MODEL_SPECS, ModelSpec

TaskKind = Literal[
    "classification",
    "regression",
    "anomaly",
    "on_policy",
    "segmentation",
    "vae",
]


@dataclass(frozen=True)
class ModelAdapter:
    spec: ModelSpec
    task_kind: TaskKind
    primary_metric_key: str
    num_classes: int | None = None
    categorical_input: bool = False
    default_enabled: bool = False


_SPECS = {spec.key: spec for spec in MODEL_SPECS}


def _adapter(
    key: str,
    task_kind: TaskKind,
    primary_metric_key: str,
    *,
    num_classes: int | None = None,
    categorical_input: bool = False,
    default_enabled: bool = False,
) -> ModelAdapter:
    return ModelAdapter(
        spec=_SPECS[key],
        task_kind=task_kind,
        primary_metric_key=primary_metric_key,
        num_classes=num_classes,
        categorical_input=categorical_input,
        default_enabled=default_enabled,
    )


MODEL_ADAPTERS: dict[str, ModelAdapter] = {
    adapter.spec.key: adapter
    for adapter in (
        _adapter("lenet5", "classification", "accuracy", num_classes=10, default_enabled=True),
        _adapter("m5", "classification", "accuracy", num_classes=12),
        _adapter("lstm_forecaster", "regression", "mae"),
        _adapter("textcnn", "classification", "accuracy", num_classes=4),
        _adapter("gcn", "classification", "accuracy", num_classes=7),
        _adapter("tabnet", "classification", "accuracy", num_classes=2, categorical_input=True),
        _adapter("mpnn", "regression", "rmse"),
        _adapter("actor_critic", "classification", "reward_proxy"),
        _adapter("lstm_autoencoder", "anomaly", "auc"),
        _adapter("distilbert", "classification", "accuracy", num_classes=2),
        _adapter("dqn_lunarlander", "classification", "reward_proxy"),
        _adapter("ppo_bipedalwalker", "on_policy", "episodic_return"),
        _adapter("attentivefp_freesolv", "regression", "rmse"),
        _adapter("gin_imdbb", "classification", "accuracy", num_classes=2),
        _adapter("tcn_forecaster", "regression", "mae", default_enabled=True),
        _adapter("gru_forecaster", "regression", "mae"),
        _adapter("pointnet_modelnet40", "classification", "accuracy", num_classes=40, default_enabled=True),
        _adapter("vae_mnist", "vae", "elbo"),
        _adapter("snn_nmnist", "classification", "accuracy", num_classes=10),
        _adapter("resnet18_cifar10", "classification", "accuracy", num_classes=10, default_enabled=True),
        _adapter("resnet18_hf_perforated_cifar10", "classification", "accuracy", num_classes=10),
        _adapter("mobilenetv2_cifar10", "classification", "accuracy", num_classes=10),
        _adapter("saint_adult", "classification", "accuracy", num_classes=2, categorical_input=True, default_enabled=True),
        _adapter("capsnet_mnist", "classification", "accuracy", num_classes=10),
        # --- PerforatedAI upstream base examples ---------------------------
        # default_enabled for all five: they are the roster this experiment
        # was re-scoped onto, so a bare `dqb run` selects exactly them plus
        # the previously evidence-backed models.
        _adapter(
            "mnist_pai", "classification", "accuracy",
            num_classes=10, default_enabled=True,
        ),
        _adapter(
            "resnet18_hf_perforated_cifar100", "classification", "accuracy",
            num_classes=100, default_enabled=True,
        ),
        _adapter(
            "resnet18_kd_cifar100", "classification", "accuracy",
            num_classes=100, default_enabled=True,
        ),
        _adapter("unet_carvana", "segmentation", "dice", default_enabled=True),
        _adapter("unet_supervisely", "segmentation", "miou", default_enabled=True),
    )
}

DEFAULT_MODEL_KEYS: tuple[str, ...] = tuple(
    spec.key for spec in MODEL_SPECS if MODEL_ADAPTERS[spec.key].default_enabled
)
ALL_MODEL_KEYS: tuple[str, ...] = tuple(spec.key for spec in MODEL_SPECS)


def model_adapter(key: str) -> ModelAdapter:
    try:
        return MODEL_ADAPTERS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown model key: {key}") from exc


def selected_model_keys(requested: list[str] | None) -> list[str]:
    """Resolve CLI selection while making exploratory breadth explicit."""
    if not requested:
        return list(DEFAULT_MODEL_KEYS)
    if "all" in requested:
        if requested != ["all"]:
            raise ValueError("'all' cannot be combined with individual model keys")
        return list(ALL_MODEL_KEYS)
    unknown = sorted(set(requested) - set(MODEL_ADAPTERS))
    if unknown:
        raise KeyError(f"Unknown model key(s): {', '.join(unknown)}")
    return list(dict.fromkeys(requested))
