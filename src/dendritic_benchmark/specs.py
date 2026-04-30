from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MetricDirection = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    domain: str
    dataset: str
    metric_name: str
    metric_direction: MetricDirection
    factory: str


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

    @property
    def quantized(self) -> bool:
        return self.bit_width is not None and self.bit_width < 32


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec("lenet5", "LeNet-5", "Image Classification", "MNIST", "Accuracy", "maximize", "lenet5"),
    ModelSpec("m5", "M5 (1D-CNN)", "Audio Classification", "SpeechCommands", "Accuracy", "maximize", "m5"),
    ModelSpec("lstm_forecaster", "LSTM Univariate", "Time-Series Forecasting", "ETTh1", "MAE", "minimize", "lstm_forecaster"),
    ModelSpec("textcnn", "TextCNN", "NLP / Text Classification", "AG News", "Accuracy", "maximize", "textcnn"),
    ModelSpec("gcn", "GCN", "Graph / Node Classification", "Cora", "Accuracy", "maximize", "gcn"),
    ModelSpec("tabnet", "TabNet", "Tabular Classification", "Adult Income", "Accuracy", "maximize", "tabnet"),
    ModelSpec("mpnn", "MPNN", "Drug Discovery / Molecular", "ESOL", "RMSE", "minimize", "mpnn"),
    ModelSpec("actor_critic", "Actor-Critic", "Reinforcement Learning", "CartPole-v1", "Reward", "maximize", "actor_critic"),
    ModelSpec("lstm_autoencoder", "LSTM Autoencoder", "Anomaly Detection", "MIT-BIH", "AUC", "maximize", "lstm_autoencoder"),
    ModelSpec("distilbert", "DistilBERT", "NLP / Seq Classification", "SST-2", "Accuracy", "maximize", "distilbert"),
]


CONDITION_SPECS: list[ConditionSpec] = [
    ConditionSpec("base_fp32", "Base FP32", "base_fp32", 32, None, False, False, False),
    ConditionSpec("base_q8", "Base + Q8", "base_fp32", 8, "int", False, False, True),
    ConditionSpec("base_q4", "Base + Q4", "base_fp32", 4, "int", False, False, True),
    ConditionSpec("base_q2", "Base + Q2", "base_fp32", 2, "int", False, False, True),
    ConditionSpec("base_q1_58", "Base + Q1.58", "base_fp32", 2, "ternary", False, False, True),
    ConditionSpec("base_q1", "Base + Q1", "base_fp32", 1, "binary", False, False, True),
    ConditionSpec("dendrites_fp32", "+Dendrites", "base_fp32", 32, None, True, False, False),
    ConditionSpec("dendrites_pruned", "+Dend + Prune", "dendrites_fp32", 32, None, True, True, False, fine_tune_epochs=5),
    ConditionSpec("dendrites_pruned_q8", "+Dend + Prune + Q8", "dendrites_pruned", 8, "int", True, False, True),
    ConditionSpec("dendrites_pruned_q4", "+Dend + Prune + Q4", "dendrites_pruned", 4, "int", True, False, True),
    ConditionSpec("dendrites_pruned_q2", "+Dend + Prune + Q2", "dendrites_pruned", 2, "int", True, False, True),
    ConditionSpec("dendrites_pruned_q1_58", "+Dend + Prune + Q1.58", "dendrites_pruned", 2, "ternary", True, False, True),
    ConditionSpec("dendrites_pruned_q1", "+Dend + Prune + Q1", "dendrites_pruned", 1, "binary", True, False, True),
]


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
