from __future__ import annotations

from typing import Any, Callable

from .compat import F, nn, require_torch


def _ensure_torch() -> Any:
    return require_torch()


if nn is not None:  # pragma: no branch - optional dependency gating
    import torch

    class LeNet5(nn.Module):
        def __init__(self, num_classes: int = 10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.AvgPool2d(2),
                nn.Conv2d(6, 16, kernel_size=5),
                nn.ReLU(),
                nn.AvgPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))


    class M5(nn.Module):
        def __init__(self, num_classes: int = 12):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=80, stride=4, padding=38),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, num_classes))

        def forward(self, x):
            return self.classifier(self.features(x))


    class LSTMForecaster(nn.Module):
        def __init__(self, input_size: int = 1, hidden_size: int = 64):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (hidden, _) = self.lstm(x)
            return self.head(hidden[-1]).squeeze(-1)


    class TextCNN(nn.Module):
        def __init__(self, vocab_size: int = 5000, embed_dim: int = 128, num_classes: int = 4):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.convs = nn.ModuleList(
                [nn.Conv1d(embed_dim, 64, kernel_size=k) for k in (3, 4, 5)]
            )
            self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(64 * 3, num_classes))

        def forward(self, x):
            x = self.embedding(x.long()).transpose(1, 2)
            pooled = [F.relu(conv(x)).max(dim=-1).values for conv in self.convs]
            return self.classifier(torch.cat(pooled, dim=1))


    class GraphConv(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, x, adjacency):
            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            norm_adj = adjacency / degree
            return self.linear(norm_adj @ x)


    class GCN(nn.Module):
        def __init__(self, in_features: int = 1433, hidden: int = 64, num_classes: int = 7):
            super().__init__()
            self.conv1 = GraphConv(in_features, hidden)
            self.conv2 = GraphConv(hidden, num_classes)

        def forward(self, x, adjacency):
            x = F.relu(self.conv1(x, adjacency))
            x = self.conv2(x, adjacency)
            return x.mean(dim=1)


    class TabNetLite(nn.Module):
        def __init__(self, in_features: int = 14, hidden: int = 64, num_classes: int = 2):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.head = nn.Linear(hidden, num_classes)

        def forward(self, x):
            return self.head(self.backbone(x))


    class MPNN(nn.Module):
        def __init__(self, node_features: int = 9, hidden: int = 64):
            super().__init__()
            self.message = nn.Sequential(nn.Linear(node_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.update = nn.GRUCell(hidden, hidden)
            self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, node_features, adjacency):
            batch_size, num_nodes, _ = node_features.shape
            h = self.message(node_features)
            agg = torch.bmm(adjacency, h)
            state = torch.zeros(batch_size * num_nodes, h.shape[-1], device=node_features.device, dtype=node_features.dtype)
            updated = self.update(agg.reshape(batch_size * num_nodes, -1), state)
            graph_repr = updated.view(batch_size, num_nodes, -1).mean(dim=1)
            return self.readout(graph_repr).squeeze(-1)


    class ActorCritic(nn.Module):
        def __init__(self, obs_dim: int = 4, hidden: int = 64, action_dim: int = 2):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            self.policy = nn.Linear(hidden, action_dim)
            self.value = nn.Linear(hidden, 1)

        def forward(self, x):
            hidden = self.backbone(x)
            return self.policy(hidden), self.value(hidden).squeeze(-1)


    class LSTMAutoencoder(nn.Module):
        def __init__(self, input_size: int = 1, hidden: int = 64):
            super().__init__()
            self.encoder = nn.LSTM(input_size, hidden, batch_first=True)
            self.decoder = nn.LSTM(input_size, hidden, batch_first=True)
            self.output = nn.Linear(hidden, input_size)

        def forward(self, x):
            batch, seq_len, feat = x.shape
            _, (hidden, _) = self.encoder(x)
            decoder_input = x.new_zeros((batch, seq_len, feat))
            decoded, _ = self.decoder(decoder_input, (hidden, torch.zeros_like(hidden)))
            return self.output(decoded)


    class DistilBertFallback(nn.Module):
        def __init__(self, vocab_size: int = 5000, embed_dim: int = 128, num_classes: int = 2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.encoder = nn.GRU(embed_dim, 128, batch_first=True, bidirectional=True)
            self.head = nn.Linear(256, num_classes)

        def forward(self, x):
            embedded = self.embedding(x.long())
            encoded, _ = self.encoder(embedded)
            pooled = encoded.mean(dim=1)
            return self.head(pooled)

else:  # pragma: no cover - import-time fallback

    LeNet5 = M5 = LSTMForecaster = TextCNN = GCN = TabNetLite = MPNN = ActorCritic = LSTMAutoencoder = DistilBertFallback = object


MODEL_FACTORIES: dict[str, Callable[..., Any]] = {
    "lenet5": lambda num_classes=10, **_: LeNet5(num_classes=num_classes),
    "m5": lambda num_classes=12, **_: M5(num_classes=num_classes),
    "lstm_forecaster": lambda **_: LSTMForecaster(),
    "textcnn": lambda num_classes=4, **_: TextCNN(num_classes=num_classes),
    "gcn": lambda num_classes=7, **_: GCN(num_classes=num_classes),
    "tabnet": lambda num_classes=2, **_: TabNetLite(num_classes=num_classes),
    "mpnn": lambda **_: MPNN(),
    "actor_critic": lambda **_: ActorCritic(),
    "lstm_autoencoder": lambda **_: LSTMAutoencoder(),
    "distilbert": lambda num_classes=2, **_: DistilBertFallback(num_classes=num_classes),
}


def build_model(model_key: str, **kwargs: Any) -> Any:
    if model_key not in MODEL_FACTORIES:
        raise KeyError(f"Unknown model key: {model_key}")
    _ensure_torch()
    return MODEL_FACTORIES[model_key](**kwargs)
