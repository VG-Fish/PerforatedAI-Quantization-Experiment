from __future__ import annotations

from typing import Any, Callable, cast

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

        def forward(self, x: Any) -> Any:
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

        def forward(self, x: Any) -> Any:
            return self.classifier(self.features(x))


    class LSTMForecaster(nn.Module):
        def __init__(self, input_size: int = 1, hidden_size: int = 64):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: Any) -> Any:
            output, _ = self.lstm(x)
            return self.head(output[:, -1]).squeeze(-1)


    class TextCNN(nn.Module):
        def __init__(self, vocab_size: int = 5000, embed_dim: int = 128, num_classes: int = 4):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.convs = nn.ModuleList(
                [nn.Conv1d(embed_dim, 64, kernel_size=k) for k in (3, 4, 5)]
            )
            self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(64 * 3, num_classes))

        def forward(self, x: Any) -> Any:
            x = self.embedding(x.long()).transpose(1, 2)
            pooled = [F.relu(conv(x)).max(dim=-1).values for conv in self.convs]
            return self.classifier(torch.cat(pooled, dim=1))


    class GraphConv(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, x: Any, adjacency: Any) -> Any:
            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            norm_adj = adjacency / degree
            return self.linear(norm_adj @ x)


    class GCN(nn.Module):
        def __init__(self, in_features: int = 1433, hidden: int = 64, num_classes: int = 7):
            super().__init__()
            self.conv1 = GraphConv(in_features, hidden)
            self.conv2 = GraphConv(hidden, num_classes)

        def forward(self, x: Any, adjacency: Any) -> Any:
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

        def forward(self, x: Any) -> Any:
            return self.head(self.backbone(x))


    class MPNN(nn.Module):
        def __init__(self, node_features: int = 9, hidden: int = 64):
            super().__init__()
            self.message = nn.Sequential(nn.Linear(node_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.update = nn.GRUCell(hidden, hidden)
            self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, node_features: Any, adjacency: Any) -> Any:
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

        def forward(self, x: Any) -> tuple[Any, Any]:
            hidden = self.backbone(x)
            return self.policy(hidden), self.value(hidden).squeeze(-1)


    class LSTMAutoencoder(nn.Module):
        def __init__(self, input_size: int = 1, hidden: int = 64):
            super().__init__()
            self.encoder = nn.LSTM(input_size, hidden, batch_first=True)
            self.decoder = nn.LSTM(input_size, hidden, batch_first=True)
            self.output = nn.Linear(hidden, input_size)

        def forward(self, x: Any) -> Any:
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

        def forward(self, x: Any) -> Any:
            embedded = self.embedding(x.long())
            encoded, _ = self.encoder(embedded)
            pooled = encoded.mean(dim=1)
            return self.head(pooled)


    class DQN(nn.Module):
        def __init__(self, obs_dim: int = 8, hidden: int = 128, action_dim: int = 4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_dim),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)


    class PPOPolicy(nn.Module):
        def __init__(self, obs_dim: int = 24, hidden: int = 128, action_dim: int = 4):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )
            self.actor = nn.Linear(hidden, action_dim)
            self.critic = nn.Linear(hidden, 1)

        def forward(self, x: Any) -> Any:
            hidden = self.backbone(x)
            return torch.tanh(self.actor(hidden))


    class AttentiveFPLite(nn.Module):
        def __init__(self, node_features: int = 9, hidden: int = 96):
            super().__init__()
            self.node_proj = nn.Linear(node_features, hidden)
            self.edge_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
            self.update = nn.GRUCell(hidden, hidden)
            self.readout_gru = nn.GRUCell(hidden, hidden)
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, node_features: Any, adjacency: Any) -> Any:
            h = F.relu(self.node_proj(node_features))
            for _ in range(3):
                agg = torch.bmm(adjacency, h)
                gate = self.edge_gate(torch.cat([h, agg], dim=-1))
                h = self.update((gate * agg).reshape(-1, h.shape[-1]), h.reshape(-1, h.shape[-1])).view_as(h)
            graph = h.mean(dim=1)
            context = torch.zeros_like(graph)
            for _ in range(2):
                context = self.readout_gru(graph, context)
            return self.head(context).squeeze(-1)


    class GINLayer(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.eps = nn.Parameter(torch.zeros(1))
            self.mlp = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Linear(out_features, out_features),
            )

        def forward(self, x: Any, adjacency: Any) -> Any:
            return self.mlp((1.0 + self.eps) * x + torch.bmm(adjacency, x))


    class GIN(nn.Module):
        def __init__(self, in_features: int = 8, hidden: int = 64, num_classes: int = 2):
            super().__init__()
            self.layers = nn.ModuleList(
                [GINLayer(in_features, hidden), GINLayer(hidden, hidden), GINLayer(hidden, hidden)]
            )
            self.head = nn.Linear(hidden, num_classes)

        def forward(self, x: Any, adjacency: Any) -> Any:
            for layer in self.layers:
                x = F.relu(layer(x, adjacency))
            return self.head(x.mean(dim=1))


    class TemporalBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int):
            super().__init__()
            padding = dilation
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 3, padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 3, padding=padding, dilation=dilation),
                nn.ReLU(),
            )
            self.downsample = (
                nn.Conv1d(in_channels, out_channels, 1)
                if in_channels != out_channels
                else nn.Identity()
            )

        def forward(self, x: Any) -> Any:
            out = self.net(x)
            out = out[..., : x.shape[-1]]
            return F.relu(out + self.downsample(x))


    class TCNForecaster(nn.Module):
        def __init__(self, input_size: int = 7, horizon: int = 24, hidden: int = 64):
            super().__init__()
            self.horizon = horizon
            self.input_size = input_size
            self.net = nn.Sequential(
                TemporalBlock(input_size, hidden, 1),
                TemporalBlock(hidden, hidden, 2),
                TemporalBlock(hidden, hidden, 4),
                TemporalBlock(hidden, hidden, 8),
            )
            self.head = nn.Linear(hidden, horizon * input_size)

        def forward(self, x: Any) -> Any:
            x = x.transpose(1, 2)
            h = self.net(x)[..., -1]
            return self.head(h).view(-1, self.horizon, self.input_size)


    class GRUForecaster(nn.Module):
        def __init__(self, input_size: int = 21, horizon: int = 24, hidden: int = 64):
            super().__init__()
            self.horizon = horizon
            self.input_size = input_size
            self.gru = nn.GRU(input_size, hidden, num_layers=2, batch_first=True, bidirectional=True)
            self.head = nn.Linear(hidden * 2, horizon * input_size)

        def forward(self, x: Any) -> Any:
            encoded, _ = self.gru(x)
            return self.head(encoded[:, -1]).view(-1, self.horizon, self.input_size)


    class PointNet(nn.Module):
        def __init__(self, num_classes: int = 40):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 1024),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes),
            )

        def forward(self, points: Any) -> Any:
            features = self.mlp(points)
            return self.head(features.max(dim=1).values)


    class VAE(nn.Module):
        def __init__(self, latent_dim: int = 32):
            super().__init__()
            self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 400), nn.ReLU())
            self.mu = nn.Linear(400, latent_dim)
            self.logvar = nn.Linear(400, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 400),
                nn.ReLU(),
                nn.Linear(400, 784),
                nn.Sigmoid(),
            )

        def forward(self, x: Any) -> tuple[Any, Any, Any]:
            hidden = self.encoder(x)
            mu = self.mu(hidden)
            logvar = self.logvar(hidden)
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std if self.training else mu
            recon = self.decoder(z).view(-1, 1, 28, 28)
            return recon, mu, logvar


    class SNNLite(nn.Module):
        def __init__(self, num_classes: int = 10):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(2, 24, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(48 * 4 * 4, num_classes),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)


    class DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)


    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = DoubleConv(3, 16)
            self.enc2 = DoubleConv(16, 32)
            self.enc3 = DoubleConv(32, 64)
            self.pool = nn.MaxPool2d(2)
            self.mid = DoubleConv(64, 128)
            self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec3 = DoubleConv(128, 64)
            self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec2 = DoubleConv(64, 32)
            self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.dec1 = DoubleConv(32, 16)
            self.out = nn.Conv2d(16, 1, 1)

        def forward(self, x: Any) -> Any:
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            m = self.mid(self.pool(e3))
            d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out(d1)


    class SAINTLite(nn.Module):
        def __init__(self, in_features: int = 14, hidden: int = 64, num_classes: int = 2):
            super().__init__()
            self.feature_embed = nn.Linear(1, hidden)
            self.column_attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
            self.row_attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
            self.ffn = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.ReLU())
            self.head = nn.Linear(hidden, num_classes)

        def forward(self, x: Any) -> Any:
            tokens = self.feature_embed(x.unsqueeze(-1))
            tokens, _ = self.column_attn(tokens, tokens, tokens, need_weights=False)
            row_tokens = tokens.mean(dim=1, keepdim=True)
            row_tokens, _ = self.row_attn(row_tokens, row_tokens, row_tokens, need_weights=False)
            return self.head(self.ffn(tokens.mean(dim=1) + row_tokens.squeeze(1)))


    class CapsNetLite(nn.Module):
        def __init__(self, num_classes: int = 10):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 128, 5, padding=2),
                nn.ReLU(),
                nn.Conv2d(128, 256, 5, stride=2, padding=2),
                nn.ReLU(),
            )
            self.primary = nn.Linear(256 * 14 * 14, 8 * 32)
            self.digit_caps = nn.Parameter(torch.randn(1, 32, num_classes, 8, 16) * 0.02)
            self.decoder = nn.Sequential(nn.Linear(num_classes * 16, 128), nn.ReLU(), nn.Linear(128, num_classes))

        @staticmethod
        def squash(x: Any) -> Any:
            norm_sq = (x * x).sum(dim=-1, keepdim=True)
            return norm_sq / (1.0 + norm_sq) * x / torch.sqrt(norm_sq + 1e-8)

        def forward(self, x: Any) -> Any:
            batch = x.shape[0]
            caps = self.primary(self.conv(x).flatten(1)).view(batch, 32, 8)
            votes = torch.einsum("bni,bncij->bncj", caps, self.digit_caps.expand(batch, -1, -1, -1, -1))
            logits = torch.zeros(batch, 32, votes.shape[2], device=x.device, dtype=x.dtype)
            for _ in range(3):
                coeffs = torch.softmax(logits, dim=-1)
                outputs = self.squash((coeffs.unsqueeze(-1) * votes).sum(dim=1))
                logits = logits + (votes * outputs.unsqueeze(1)).sum(dim=-1)
            return outputs.norm(dim=-1)


    class ConvLSTMCell(nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int):
            super().__init__()
            self.hidden_channels = hidden_channels
            self.gates = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, 3, padding=1)

        def forward(self, x: Any, state: tuple[Any, Any]) -> tuple[Any, Any]:
            h, c = state
            i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            c = f * c + i * g
            h = o * torch.tanh(c)
            return h, c


    class ConvLSTM(nn.Module):
        def __init__(self, hidden_channels: int = 32, horizon: int = 10):
            super().__init__()
            self.horizon = horizon
            self.cell1 = ConvLSTMCell(1, hidden_channels)
            self.cell2 = ConvLSTMCell(hidden_channels, hidden_channels)
            self.decoder = nn.Conv2d(hidden_channels, 1, 3, padding=1)

        def forward(self, x: Any) -> Any:
            batch, seq, _, height, width = x.shape
            h1 = x.new_zeros(batch, self.cell1.hidden_channels, height, width)
            c1 = torch.zeros_like(h1)
            h2 = x.new_zeros(batch, self.cell2.hidden_channels, height, width)
            c2 = torch.zeros_like(h2)
            frame = x[:, 0]
            outputs = []
            for step in range(seq + self.horizon):
                if step < seq:
                    frame = x[:, step]
                h1, c1 = self.cell1(frame, (h1, c1))
                h2, c2 = self.cell2(h1, (h2, c2))
                frame = torch.sigmoid(self.decoder(h2))
                if step >= seq:
                    outputs.append(frame)
            return torch.stack(outputs, dim=1)

else:  # pragma: no cover - import-time fallback

    LeNet5 = M5 = LSTMForecaster = TextCNN = GCN = TabNetLite = MPNN = ActorCritic = LSTMAutoencoder = DistilBertFallback = object
    DQN = PPOPolicy = AttentiveFPLite = GIN = TCNForecaster = GRUForecaster = PointNet = VAE = SNNLite = TinyUNet = SAINTLite = CapsNetLite = ConvLSTM = object


LeNet5 = cast(Any, LeNet5)
M5 = cast(Any, M5)
TextCNN = cast(Any, TextCNN)
GCN = cast(Any, GCN)
TabNetLite = cast(Any, TabNetLite)
DistilBertFallback = cast(Any, DistilBertFallback)
GIN = cast(Any, GIN)
TCNForecaster = cast(Any, TCNForecaster)
GRUForecaster = cast(Any, GRUForecaster)
PointNet = cast(Any, PointNet)
SNNLite = cast(Any, SNNLite)
SAINTLite = cast(Any, SAINTLite)
CapsNetLite = cast(Any, CapsNetLite)


def _build_resnet18_cifar10(**_: Any) -> Any:
    torchvision_models = __import__("torchvision.models", fromlist=["models"])
    model = torchvision_models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def _build_mobilenetv2_cifar10(**_: Any) -> Any:
    torchvision_models = __import__("torchvision.models", fromlist=["models"])
    model = torchvision_models.mobilenet_v2(weights=None, num_classes=10)
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    return model


def _construct(model_class: Any, **kwargs: Any) -> Any:
    return model_class(**kwargs)


MODEL_FACTORIES: dict[str, Callable[..., Any]] = {
    "lenet5": lambda num_classes=10, **_: _construct(LeNet5, num_classes=num_classes),
    "m5": lambda num_classes=12, **_: _construct(M5, num_classes=num_classes),
    "lstm_forecaster": lambda **_: LSTMForecaster(),
    "textcnn": lambda num_classes=4, **_: _construct(TextCNN, num_classes=num_classes),
    "gcn": lambda num_classes=7, **_: _construct(GCN, num_classes=num_classes),
    "tabnet": lambda num_classes=2, **_: _construct(TabNetLite, num_classes=num_classes),
    "mpnn": lambda **_: MPNN(),
    "actor_critic": lambda **_: ActorCritic(),
    "lstm_autoencoder": lambda **_: LSTMAutoencoder(),
    "distilbert": lambda num_classes=2, **_: _construct(DistilBertFallback, num_classes=num_classes),
    "dqn_lunarlander": lambda **_: DQN(),
    "ppo_bipedalwalker": lambda **_: PPOPolicy(),
    "attentivefp_freesolv": lambda **_: AttentiveFPLite(),
    "gin_imdbb": lambda num_classes=2, **_: _construct(GIN, num_classes=num_classes),
    "tcn_forecaster": lambda **_: _construct(TCNForecaster, input_size=7),
    "gru_forecaster": lambda **_: _construct(GRUForecaster, input_size=21),
    "pointnet_modelnet40": lambda num_classes=40, **_: _construct(PointNet, num_classes=num_classes),
    "vae_mnist": lambda **_: VAE(),
    "snn_nmnist": lambda num_classes=10, **_: _construct(SNNLite, num_classes=num_classes),
    "unet_isic": lambda **_: TinyUNet(),
    "resnet18_cifar10": _build_resnet18_cifar10,
    "mobilenetv2_cifar10": _build_mobilenetv2_cifar10,
    "saint_adult": lambda num_classes=2, **_: _construct(SAINTLite, num_classes=num_classes),
    "capsnet_mnist": lambda num_classes=10, **_: _construct(CapsNetLite, num_classes=num_classes),
    "convlstm_movingmnist": lambda **_: ConvLSTM(),
}


def build_model(model_key: str, **kwargs: Any) -> Any:
    if model_key not in MODEL_FACTORIES:
        raise KeyError(f"Unknown model key: {model_key}")
    _ensure_torch()
    return MODEL_FACTORIES[model_key](**kwargs)
