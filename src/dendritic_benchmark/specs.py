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


_IMAGE_CLASSIFICATION = "Image Classification"
_TIME_SERIES_FORECASTING = "Time-Series Forecasting"
_REINFORCEMENT_LEARNING = "Reinforcement Learning"

MODEL_SPECS: list[ModelSpec] = [
    ModelSpec("lenet5", "LeNet-5", _IMAGE_CLASSIFICATION, "MNIST", "Accuracy", "maximize", "lenet5"),
    ModelSpec("m5", "M5 (1D-CNN)", "Audio Classification", "SpeechCommands", "Accuracy", "maximize", "m5"),
    ModelSpec("lstm_forecaster", "LSTM Univariate", _TIME_SERIES_FORECASTING, "ETTh1", "MAE", "minimize", "lstm_forecaster"),
    ModelSpec("textcnn", "TextCNN", "NLP / Text Classification", "AG News", "Accuracy", "maximize", "textcnn"),
    ModelSpec("gcn", "GCN", "Graph / Node Classification", "Cora", "Accuracy", "maximize", "gcn"),
    ModelSpec("tabnet", "TabNet", "Tabular Classification", "Adult Income", "Accuracy", "maximize", "tabnet"),
    ModelSpec("mpnn", "MPNN", "Drug Discovery / Molecular", "ESOL", "RMSE", "minimize", "mpnn"),
    ModelSpec("actor_critic", "Actor-Critic", _REINFORCEMENT_LEARNING, "CartPole-v1", "Reward", "maximize", "actor_critic"),
    ModelSpec("lstm_autoencoder", "LSTM Autoencoder", "Anomaly Detection", "MIT-BIH", "AUC", "maximize", "lstm_autoencoder"),
    ModelSpec("distilbert", "DistilBERT", "NLP / Seq Classification", "SST-2", "Accuracy", "maximize", "distilbert"),
    ModelSpec("dqn_lunarlander", "DQN (LunarLander)", _REINFORCEMENT_LEARNING, "LunarLander-v2", "Reward", "maximize", "dqn_lunarlander"),
    ModelSpec("ppo_bipedalwalker", "PPO Policy Network", _REINFORCEMENT_LEARNING, "BipedalWalker-v3", "Reward", "maximize", "ppo_bipedalwalker"),
    ModelSpec("attentivefp_freesolv", "AttentiveFP", "Drug Discovery / Molecular", "FreeSolv", "RMSE", "minimize", "attentivefp_freesolv"),
    ModelSpec("gin_imdbb", "GIN", "Graph Classification", "IMDB-Binary", "Accuracy", "maximize", "gin_imdbb"),
    ModelSpec("tcn_forecaster", "TCN Forecaster", _TIME_SERIES_FORECASTING, "ETTm1", "MAE", "minimize", "tcn_forecaster"),
    ModelSpec("gru_forecaster", "GRU Forecaster", _TIME_SERIES_FORECASTING, "Weather", "MAE", "minimize", "gru_forecaster"),
    ModelSpec("pointnet_modelnet40", "PointNet", "3D Point Cloud Classification", "ModelNet40", "Accuracy", "maximize", "pointnet_modelnet40"),
    ModelSpec("vae_mnist", "VAE", "Generative Modeling", "MNIST", "ELBO", "maximize", "vae_mnist"),
    ModelSpec("snn_nmnist", "Spiking Neural Network", "Neuromorphic Computing", "N-MNIST", "Accuracy", "maximize", "snn_nmnist"),
    ModelSpec("unet_isic", "Tiny U-Net", "Medical Image Segmentation", "ISIC 2018 Task 1", "Dice", "maximize", "unet_isic"),
    ModelSpec("resnet18_cifar10", "ResNet-18", _IMAGE_CLASSIFICATION, "CIFAR-10", "Accuracy", "maximize", "resnet18_cifar10"),
    ModelSpec("mobilenetv2_cifar10", "MobileNetV2", _IMAGE_CLASSIFICATION, "CIFAR-10", "Accuracy", "maximize", "mobilenetv2_cifar10"),
    ModelSpec("saint_adult", "SAINT", "Tabular Classification", "Adult Income", "Accuracy", "maximize", "saint_adult"),
    ModelSpec("capsnet_mnist", "CapsNet", _IMAGE_CLASSIFICATION, "MNIST", "Accuracy", "maximize", "capsnet_mnist"),
    ModelSpec("convlstm_movingmnist", "ConvLSTM", "Spatiotemporal Prediction", "Moving MNIST", "SSIM", "maximize", "convlstm_movingmnist"),
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
