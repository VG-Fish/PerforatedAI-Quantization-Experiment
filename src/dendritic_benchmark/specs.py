from dataclasses import dataclass
from typing import Literal

MetricDirection = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    dataset: str
    metric_name: str
    metric_direction: MetricDirection


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    display_name: str
    source_key: str
    bit_width: int | None
    quantization_mode: str | None
    use_dendrites: bool
    use_pruning: bool
    use_qat: bool
    prune_amount: float = 0.4
    fine_tune_epochs: int = 0
    # ``None`` is an ordinary base/dendritic condition.  The two controls use
    # the retained dendritic artifact's saved fork rather than its final weights.
    control_kind: Literal["base_more_training", "capacity_dense"] | None = None

    @property
    def quantized(self) -> bool:
        return self.bit_width is not None and self.bit_width < 32


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec("lenet5", "LeNet-5", "MNIST", "Accuracy", "maximize"),
    ModelSpec("m5", "M5 (1D-CNN)", "SpeechCommands", "Accuracy", "maximize"),
    ModelSpec("lstm_forecaster", "LSTM Univariate", "ETTh1", "MAE", "minimize"),
    ModelSpec("textcnn", "TextCNN", "AG News", "Accuracy", "maximize"),
    ModelSpec("gcn", "GCN", "Cora", "Accuracy", "maximize"),
    ModelSpec("tabnet", "TabNet", "Adult Income", "Accuracy", "maximize"),
    ModelSpec("mpnn", "MPNN", "ESOL", "RMSE", "minimize"),
    ModelSpec("actor_critic", "Actor-Critic", "CartPole-v1", "Action Accuracy", "maximize"),
    ModelSpec("lstm_autoencoder", "LSTM Autoencoder", "MIT-BIH", "AUC", "maximize"),
    ModelSpec("distilbert", "DistilBERT", "SST-2", "Accuracy", "maximize"),
    ModelSpec("dqn_lunarlander", "DQN (LunarLander)", "LunarLander-v2", "Action Accuracy", "maximize"),
    ModelSpec("ppo_bipedalwalker", "PPO Policy Network", "BipedalWalker-v3", "Episodic Return", "maximize"),
    ModelSpec("attentivefp_freesolv", "AttentiveFP", "FreeSolv", "RMSE", "minimize"),
    ModelSpec("gin_imdbb", "GIN", "IMDB-Binary", "Accuracy", "maximize"),
    ModelSpec("tcn_forecaster", "TCN Forecaster", "ETTm1", "MAE", "minimize"),
    ModelSpec("gru_forecaster", "GRU Forecaster", "Weather", "MAE", "minimize"),
    ModelSpec("pointnet_modelnet40", "PointNet", "ModelNet40", "Accuracy", "maximize"),
    ModelSpec("vae_mnist", "VAE", "MNIST", "ELBO", "maximize"),
    ModelSpec("snn_nmnist", "Spiking Neural Network", "N-MNIST", "Accuracy", "maximize"),
    ModelSpec("resnet18_cifar10", "ResNet-18", "CIFAR-10", "Accuracy", "maximize"),
    ModelSpec(
        "resnet18_hf_perforated_cifar10",
        "HF Perforated ResNet-18",
        "CIFAR-10",
        "Accuracy",
        "maximize",
    ),
    ModelSpec("mobilenetv2_cifar10", "MobileNetV2", "CIFAR-10", "Accuracy", "maximize"),
    ModelSpec("saint_adult", "SAINT", "Adult Income", "Accuracy", "maximize"),
    ModelSpec("capsnet_mnist", "CapsNet", "MNIST", "Accuracy", "maximize"),
]


CONDITION_SPECS: list[ConditionSpec] = [
    ConditionSpec("base_fp32", "Base FP32", "base_fp32", 32, None, False, False, False),
    ConditionSpec("base_q8", "Base + Q8", "base_fp32", 8, "int", False, False, False),
    ConditionSpec("base_q4", "Base + Q4", "base_fp32", 4, "int", False, False, False),
    ConditionSpec("base_q2", "Base + Q2", "base_fp32", 2, "int", False, False, False),
    ConditionSpec("base_q1_58", "Base + Q1.58", "base_fp32", 2, "ternary", False, False, False),
    ConditionSpec("base_q1", "Base + Q1", "base_fp32", 1, "binary", False, False, False),
    ConditionSpec("dendrites_fp32", "+Dendrites", "base_fp32", 32, None, True, False, False),
    ConditionSpec("dendrites_q8", "+Dendrites + Q8", "dendrites_fp32", 8, "int", True, False, False),
    ConditionSpec("dendrites_q4", "+Dendrites + Q4", "dendrites_fp32", 4, "int", True, False, False),
    ConditionSpec("dendrites_q2", "+Dendrites + Q2", "dendrites_fp32", 2, "int", True, False, False),
    ConditionSpec("dendrites_q1_58", "+Dendrites + Q1.58", "dendrites_fp32", 2, "ternary", True, False, False),
    ConditionSpec("dendrites_q1", "+Dendrites + Q1", "dendrites_fp32", 1, "binary", True, False, False),
    ConditionSpec("base_more_training_fp32", "Base + Matched Training", "dendrites_fp32", 32, None, False, False, False, control_kind="base_more_training"),
    ConditionSpec("base_more_training_q8", "Base + Matched Training + Q8", "base_more_training_fp32", 8, "int", False, False, False, control_kind="base_more_training"),
    ConditionSpec("base_more_training_q4", "Base + Matched Training + Q4", "base_more_training_fp32", 4, "int", False, False, False, control_kind="base_more_training"),
    ConditionSpec("base_more_training_q2", "Base + Matched Training + Q2", "base_more_training_fp32", 2, "int", False, False, False, control_kind="base_more_training"),
    ConditionSpec("base_more_training_q1_58", "Base + Matched Training + Q1.58", "base_more_training_fp32", 2, "ternary", False, False, False, control_kind="base_more_training"),
    ConditionSpec("base_more_training_q1", "Base + Matched Training + Q1", "base_more_training_fp32", 1, "binary", False, False, False, control_kind="base_more_training"),
    ConditionSpec("capacity_dense_fp32", "Topology-Matched Dense", "dendrites_fp32", 32, None, False, False, False, control_kind="capacity_dense"),
    ConditionSpec("capacity_dense_q8", "Topology-Matched Dense + Q8", "capacity_dense_fp32", 8, "int", False, False, False, control_kind="capacity_dense"),
    ConditionSpec("capacity_dense_q4", "Topology-Matched Dense + Q4", "capacity_dense_fp32", 4, "int", False, False, False, control_kind="capacity_dense"),
    ConditionSpec("capacity_dense_q2", "Topology-Matched Dense + Q2", "capacity_dense_fp32", 2, "int", False, False, False, control_kind="capacity_dense"),
    ConditionSpec("capacity_dense_q1_58", "Topology-Matched Dense + Q1.58", "capacity_dense_fp32", 2, "ternary", False, False, False, control_kind="capacity_dense"),
    ConditionSpec("capacity_dense_q1", "Topology-Matched Dense + Q1", "capacity_dense_fp32", 1, "binary", False, False, False, control_kind="capacity_dense"),
]


HF_PERFORATED_RESNET18_KEY = "resnet18_hf_perforated_cifar10"


def condition_supported_by_model(model_key: str, condition_key: str) -> bool:
    """Return whether a condition represents a distinct model comparison.

    The Hugging Face ResNet checkpoint already contains its trained dendritic
    graph.  Its ``base_*`` conditions are therefore the perforated model;
    another ``dendrites_*`` conversion would stack a second search graph on
    top and would not be a meaningful control.
    """
    return not (
        model_key == HF_PERFORATED_RESNET18_KEY
        and condition_key.startswith("dendrites_")
    )


def model_by_key(key: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown model key: {key}")


def condition_by_key(key: str) -> ConditionSpec:
    for spec in CONDITION_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown condition key: {key}")
