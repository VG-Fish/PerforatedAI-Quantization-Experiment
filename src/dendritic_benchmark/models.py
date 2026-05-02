from __future__ import annotations

import math
from typing import Any, Callable, cast

from .compat import F, nn, require_torch


def _ensure_torch() -> Any:
    return require_torch()


if nn is not None:  # pragma: no branch - optional dependency gating
    import torch

    class LeNet5(nn.Module):
        """LeNet-5 style CNN for MNIST-sized grayscale images."""

        def __init__(self, num_classes: int = 10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
                nn.Tanh(),
                nn.AvgPool2d(2),
                nn.Conv2d(6, 16, kernel_size=5),
                nn.Tanh(),
                nn.AvgPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 5 * 5, 120),
                nn.Tanh(),
                nn.Linear(120, 84),
                nn.Tanh(),
                nn.Linear(84, num_classes),
            )

        def forward(self, x: Any) -> Any:
            return self.classifier(self.features(x))


    class M5(nn.Module):
        """M5 audio classifier from the PyTorch Speech Commands tutorial."""

        def __init__(self, num_classes: int = 12, n_channel: int = 32):
            super().__init__()
            self.conv1 = nn.Conv1d(1, n_channel, kernel_size=80, stride=16)
            self.bn1 = nn.BatchNorm1d(n_channel)
            self.pool1 = nn.MaxPool1d(4)
            self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
            self.bn2 = nn.BatchNorm1d(n_channel)
            self.pool2 = nn.MaxPool1d(4)
            self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
            self.bn3 = nn.BatchNorm1d(2 * n_channel)
            self.pool3 = nn.MaxPool1d(4)
            self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
            self.bn4 = nn.BatchNorm1d(2 * n_channel)
            self.fc1 = nn.Linear(2 * n_channel, num_classes)

        def forward(self, x: Any) -> Any:
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.avg_pool1d(x, x.shape[-1]).squeeze(-1)
            return self.fc1(x)


    class DendriticLSTMCell(nn.Module):
        """LSTM cell built from Linear modules so PAI can perforate the gates."""

        def __init__(self, input_size: int, hidden_size: int):
            super().__init__()
            self.hidden_size = hidden_size
            self.input_gates = nn.Linear(input_size, 4 * hidden_size)
            self.hidden_gates = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        def forward(self, x: Any, state: tuple[Any, Any]) -> tuple[Any, Any]:
            h, c = state
            i, f, g, o = (self.input_gates(x) + self.hidden_gates(h)).chunk(4, dim=-1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c = f * c + i * g
            h = o * torch.tanh(c)
            return h, c


    class DendriticGRUCell(nn.Module):
        """GRU cell built from Linear modules for dendritic gate perforation."""

        def __init__(self, input_size: int, hidden_size: int):
            super().__init__()
            self.hidden_size = hidden_size
            self.input_gates = nn.Linear(input_size, 3 * hidden_size)
            self.hidden_gates = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

        def forward(self, x: Any, h: Any) -> Any:
            x_z, x_r, x_n = self.input_gates(x).chunk(3, dim=-1)
            h_z, h_r, h_n = self.hidden_gates(h).chunk(3, dim=-1)
            z = torch.sigmoid(x_z + h_z)
            r = torch.sigmoid(x_r + h_r)
            n = torch.tanh(x_n + r * h_n)
            return (1.0 - z) * n + z * h


    class LSTMForecaster(nn.Module):
        def __init__(
            self,
            input_size: int = 1,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.cells = nn.ModuleList(
                DendriticLSTMCell(input_size if layer == 0 else hidden_size, hidden_size)
                for layer in range(num_layers)
            )
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
            )

        def forward(self, x: Any) -> Any:
            batch = x.shape[0]
            states = [
                (
                    x.new_zeros(batch, self.hidden_size),
                    x.new_zeros(batch, self.hidden_size),
                )
                for _ in self.cells
            ]
            output = x
            for timestep in range(x.shape[1]):
                step = output[:, timestep] if output.dim() == 3 else output
                next_states = []
                for index, cell in enumerate(self.cells):
                    h, c = cell(step, states[index])
                    next_states.append((h, c))
                    step = self.dropout(h) if index < len(self.cells) - 1 else h
                states = next_states
            return self.head(states[-1][0]).squeeze(-1)


    class TextCNN(nn.Module):
        def __init__(
            self,
            vocab_size: int = 5000,
            embed_dim: int = 128,
            num_classes: int = 4,
            channels: int = 128,
        ):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.embedding_dropout = nn.Dropout(0.2)
            self.convs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(embed_dim, channels, kernel_size=k),
                        nn.BatchNorm1d(channels),
                        nn.ReLU(),
                    )
                    for k in (2, 3, 4, 5)
                ]
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(channels * 4, num_classes),
            )

        def forward(self, x: Any) -> Any:
            x = self.embedding_dropout(self.embedding(x.long())).transpose(1, 2)
            pooled = [conv(x).amax(dim=-1) for conv in self.convs]
            return self.classifier(torch.cat(pooled, dim=1))


    class GraphConv(nn.Module):
        """Dense Kipf-Welling GCN convolution for fixed ego-graph tensors."""

        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=bias)

        def forward(self, x: Any, adjacency: Any) -> Any:
            degree = adjacency.sum(dim=-1).clamp_min(1.0)
            inv_sqrt = degree.rsqrt()
            norm_adj = adjacency * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)
            return self.linear(torch.bmm(norm_adj, x))


    class GCN(nn.Module):
        def __init__(
            self,
            in_features: int = 1433,
            hidden: int = 64,
            num_classes: int = 7,
            dropout: float = 0.5,
        ):
            super().__init__()
            self.conv1 = GraphConv(in_features, hidden)
            self.conv2 = GraphConv(hidden, num_classes)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Any, adjacency: Any) -> Any:
            x = self.dropout(F.relu(self.conv1(x, adjacency)))
            x = self.conv2(x, adjacency)
            # The Cora loader puts the target paper at node slot 0.
            return x[:, 0]


    def _sparsemax(logits: Any, dim: int = -1) -> Any:
        logits = logits - logits.max(dim=dim, keepdim=True).values
        zs = torch.sort(logits, descending=True, dim=dim).values
        range_shape = [1] * logits.dim()
        range_shape[dim] = logits.shape[dim]
        rhos = torch.arange(
            1, logits.shape[dim] + 1, device=logits.device, dtype=logits.dtype
        ).view(range_shape)
        cumsum = zs.cumsum(dim)
        support = 1 + rhos * zs > cumsum
        support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
        tau = (
            cumsum.gather(dim, support_size.long() - 1) - 1
        ) / support_size.to(logits.dtype)
        return torch.clamp(logits - tau, min=0.0)


    class GLUBlock(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.fc = nn.Linear(in_features, out_features * 2, bias=False)
            self.bn = nn.BatchNorm1d(out_features * 2)

        def forward(self, x: Any) -> Any:
            value, gate = self.bn(self.fc(x)).chunk(2, dim=-1)
            return value * torch.sigmoid(gate)


    class FeatureTransformer(nn.Module):
        def __init__(self, in_features: int, out_features: int, blocks: int = 4):
            super().__init__()
            layers = []
            current = in_features
            for _ in range(blocks):
                layers.append(GLUBlock(current, out_features))
                current = out_features
            self.layers = nn.ModuleList(layers)

        def forward(self, x: Any) -> Any:
            residual = None
            for layer in self.layers:
                out = layer(x)
                if residual is not None and residual.shape == out.shape:
                    out = (out + residual) * math.sqrt(0.5)
                residual = out
                x = out
            return x


    class AttentiveTransformer(nn.Module):
        def __init__(self, in_features: int, feature_count: int):
            super().__init__()
            self.fc = nn.Linear(in_features, feature_count, bias=False)
            self.bn = nn.BatchNorm1d(feature_count)

        def forward(self, x: Any, prior: Any) -> Any:
            return _sparsemax(self.bn(self.fc(x)) * prior, dim=-1)


    class TabNet(nn.Module):
        """TabNet-style sequential attentive tabular classifier."""

        def __init__(
            self,
            in_features: int = 14,
            n_d: int = 16,
            n_a: int = 16,
            n_steps: int = 4,
            gamma: float = 1.5,
            num_classes: int = 2,
        ):
            super().__init__()
            self.n_d = n_d
            self.n_a = n_a
            self.n_steps = n_steps
            self.gamma = gamma
            self.initial_bn = nn.BatchNorm1d(in_features)
            self.shared = FeatureTransformer(in_features, n_d + n_a)
            self.step_transformers = nn.ModuleList(
                FeatureTransformer(in_features, n_d + n_a) for _ in range(n_steps)
            )
            self.attentive = nn.ModuleList(
                AttentiveTransformer(n_a, in_features) for _ in range(n_steps)
            )
            self.head = nn.Linear(n_d, num_classes)

        def forward(self, x: Any) -> Any:
            x = self.initial_bn(x.float())
            prior = torch.ones_like(x)
            transformed = self.shared(x)
            attention = transformed[:, self.n_d :]
            aggregate = x.new_zeros(x.shape[0], self.n_d)
            for step in range(self.n_steps):
                mask = self.attentive[step](attention, prior)
                prior = prior * (self.gamma - mask).clamp_min(0.0)
                transformed = self.step_transformers[step](mask * x)
                decision = F.relu(transformed[:, : self.n_d])
                aggregate = aggregate + decision
                attention = transformed[:, self.n_d :]
            return self.head(aggregate)


    class MPNNLayer(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.update = DendriticGRUCell(hidden, hidden)

        def forward(self, h: Any, adjacency: Any) -> Any:
            batch, nodes, hidden = h.shape
            source = h.unsqueeze(2).expand(batch, nodes, nodes, hidden)
            target = h.unsqueeze(1).expand(batch, nodes, nodes, hidden)
            messages = self.edge_mlp(torch.cat([target, source], dim=-1))
            messages = messages * adjacency.unsqueeze(-1)
            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            aggregated = messages.sum(dim=2) / degree
            return self.update(
                aggregated.reshape(batch * nodes, hidden),
                h.reshape(batch * nodes, hidden),
            ).view(batch, nodes, hidden)


    class MPNN(nn.Module):
        def __init__(self, node_features: int = 9, hidden: int = 96, steps: int = 4):
            super().__init__()
            self.node_encoder = nn.Sequential(
                nn.Linear(node_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.layers = nn.ModuleList(MPNNLayer(hidden) for _ in range(steps))
            self.readout_gate = nn.Linear(hidden, 1)
            self.readout = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, 1),
            )

        def forward(self, node_features: Any, adjacency: Any) -> Any:
            h = F.relu(self.node_encoder(node_features))
            for layer in self.layers:
                h = layer(h, adjacency)
            node_mask = (adjacency.sum(dim=-1) > 0).float()
            gate = torch.sigmoid(self.readout_gate(h)).squeeze(-1) * node_mask
            graph_repr = (h * gate.unsqueeze(-1)).sum(dim=1) / gate.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            return self.readout(graph_repr).squeeze(-1)


    class ActorCritic(nn.Module):
        def __init__(self, obs_dim: int = 4, hidden: int = 128, action_dim: int = 2):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )
            self.policy = nn.Linear(hidden, action_dim)
            self.value = nn.Linear(hidden, 1)

        def forward(self, x: Any) -> tuple[Any, Any]:
            hidden = self.backbone(x)
            return self.policy(hidden), self.value(hidden).squeeze(-1)


    class LSTMAutoencoder(nn.Module):
        def __init__(
            self,
            input_size: int = 1,
            hidden: int = 64,
            latent: int = 32,
            num_layers: int = 2,
        ):
            super().__init__()
            self.hidden = hidden
            self.encoder_cells = nn.ModuleList(
                DendriticLSTMCell(input_size if layer == 0 else hidden, hidden)
                for layer in range(num_layers)
            )
            self.to_latent = nn.Linear(hidden, latent)
            self.from_latent = nn.Linear(latent, hidden)
            self.decoder_cell = DendriticLSTMCell(input_size, hidden)
            self.output = nn.Linear(hidden, input_size)

        def _encode(self, x: Any) -> Any:
            batch = x.shape[0]
            states = [
                (x.new_zeros(batch, self.hidden), x.new_zeros(batch, self.hidden))
                for _ in self.encoder_cells
            ]
            for timestep in range(x.shape[1]):
                step = x[:, timestep]
                next_states = []
                for index, cell in enumerate(self.encoder_cells):
                    h, c = cell(step, states[index])
                    next_states.append((h, c))
                    step = h
                states = next_states
            return torch.tanh(self.to_latent(states[-1][0]))

        def forward(self, x: Any) -> Any:
            batch, seq_len, feat = x.shape
            hidden = torch.tanh(self.from_latent(self._encode(x)))
            cell_state = torch.zeros_like(hidden)
            decoder_input = x.new_zeros(batch, feat)
            outputs = []
            for _ in range(seq_len):
                hidden, cell_state = self.decoder_cell(decoder_input, (hidden, cell_state))
                decoder_input = self.output(hidden)
                outputs.append(decoder_input)
            return torch.stack(outputs, dim=1)


    class DistilBertClassifier(nn.Module):
        def __init__(self, num_classes: int = 2):
            super().__init__()
            transformers = cast(Any, __import__("transformers"))
            self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=num_classes,
            )

        def forward(self, input_ids: Any, attention_mask: Any | None = None) -> Any:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


    class DQN(nn.Module):
        def __init__(self, obs_dim: int = 8, hidden: int = 256, action_dim: int = 4):
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
            self.actor_mean = nn.Linear(hidden, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
            self.critic = nn.Linear(hidden, 1)

        def forward(self, x: Any) -> Any:
            hidden = self.backbone(x)
            return torch.tanh(self.actor_mean(hidden))

        def value_function(self, x: Any) -> Any:
            return self.critic(self.backbone(x)).squeeze(-1)


    class AttentiveFPLayer(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden, 1),
            )
            self.message = nn.Linear(hidden, hidden)
            self.update = DendriticGRUCell(hidden, hidden)

        def forward(self, h: Any, adjacency: Any) -> Any:
            batch, nodes, hidden = h.shape
            src = h.unsqueeze(2).expand(batch, nodes, nodes, hidden)
            dst = h.unsqueeze(1).expand(batch, nodes, nodes, hidden)
            scores = self.attention(torch.cat([dst, src], dim=-1)).squeeze(-1)
            scores = scores.masked_fill(adjacency <= 0, -1.0e9)
            weights = torch.softmax(scores, dim=-1)
            messages = torch.bmm(weights, self.message(h))
            return self.update(
                messages.reshape(batch * nodes, hidden),
                h.reshape(batch * nodes, hidden),
            ).view(batch, nodes, hidden)


    class AttentiveFP(nn.Module):
        def __init__(
            self,
            node_features: int = 9,
            hidden: int = 128,
            layers: int = 3,
            readout_steps: int = 2,
        ):
            super().__init__()
            self.node_proj = nn.Sequential(
                nn.Linear(node_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.layers = nn.ModuleList(AttentiveFPLayer(hidden) for _ in range(layers))
            self.readout_steps = readout_steps
            self.readout_attn = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )
            self.readout_gru = DendriticGRUCell(hidden, hidden)
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, 1),
            )

        def forward(self, node_features: Any, adjacency: Any) -> Any:
            h = F.relu(self.node_proj(node_features))
            for layer in self.layers:
                h = layer(h, adjacency)
            node_mask = adjacency.sum(dim=-1) > 0
            graph = h.mean(dim=1)
            for _ in range(self.readout_steps):
                expanded_graph = graph.unsqueeze(1).expand_as(h)
                scores = self.readout_attn(torch.cat([h, expanded_graph], dim=-1)).squeeze(-1)
                scores = scores.masked_fill(~node_mask, -1.0e9)
                weights = torch.softmax(scores, dim=-1)
                context = (h * weights.unsqueeze(-1)).sum(dim=1)
                graph = self.readout_gru(context, graph)
            return self.head(graph).squeeze(-1)


    class GINLayer(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.eps = nn.Parameter(torch.zeros(1))
            self.mlp = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Linear(out_features, out_features),
                nn.BatchNorm1d(out_features),
            )

        def forward(self, x: Any, adjacency: Any) -> Any:
            batch, nodes, features = x.shape
            out = (1.0 + self.eps) * x + torch.bmm(adjacency, x)
            return self.mlp(out.reshape(batch * nodes, features)).view(batch, nodes, -1)


    class GIN(nn.Module):
        def __init__(self, in_features: int = 8, hidden: int = 64, num_classes: int = 2):
            super().__init__()
            self.input_proj = nn.Linear(in_features, hidden)
            self.layers = nn.ModuleList(GINLayer(hidden, hidden) for _ in range(4))
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden, num_classes),
            )

        def forward(self, x: Any, adjacency: Any) -> Any:
            x = F.relu(self.input_proj(x))
            for layer in self.layers:
                x = F.relu(layer(x, adjacency))
            return self.head(x.mean(dim=1))


    class Chomp1d(nn.Module):
        def __init__(self, chomp_size: int):
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, x: Any) -> Any:
            return x[..., : -self.chomp_size] if self.chomp_size else x


    class TemporalBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float = 0.1):
            super().__init__()
            padding = (3 - 1) * dilation
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 3, padding=padding, dilation=dilation),
                Chomp1d(padding),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(out_channels, out_channels, 3, padding=padding, dilation=dilation),
                Chomp1d(padding),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.downsample = (
                nn.Conv1d(in_channels, out_channels, 1)
                if in_channels != out_channels
                else nn.Identity()
            )

        def forward(self, x: Any) -> Any:
            return F.relu(self.net(x) + self.downsample(x))


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
            h = self.net(x.transpose(1, 2))[..., -1]
            return self.head(h).view(-1, self.horizon, self.input_size)


    class GRUForecaster(nn.Module):
        def __init__(self, input_size: int = 21, horizon: int = 24, hidden: int = 64, layers: int = 2):
            super().__init__()
            self.horizon = horizon
            self.input_size = input_size
            self.hidden = hidden
            self.cells = nn.ModuleList(
                DendriticGRUCell(input_size if layer == 0 else hidden, hidden)
                for layer in range(layers)
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, horizon * input_size),
            )

        def forward(self, x: Any) -> Any:
            batch = x.shape[0]
            states = [x.new_zeros(batch, self.hidden) for _ in self.cells]
            for timestep in range(x.shape[1]):
                step = x[:, timestep]
                for index, cell in enumerate(self.cells):
                    states[index] = cell(step, states[index])
                    step = states[index]
            return self.head(states[-1]).view(-1, self.horizon, self.input_size)


    class TransformNet(nn.Module):
        def __init__(self, k: int):
            super().__init__()
            self.k = k
            self.conv = nn.Sequential(
                nn.Conv1d(k, 64, 1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 1024, 1),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
            )
            self.fc = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, k * k),
            )

        def forward(self, x: Any) -> Any:
            batch = x.shape[0]
            init = torch.eye(self.k, device=x.device, dtype=x.dtype).view(1, self.k * self.k)
            matrix = self.fc(self.conv(x).amax(dim=-1)) + init.repeat(batch, 1)
            return matrix.view(batch, self.k, self.k)


    class PointNet(nn.Module):
        def __init__(self, num_classes: int = 40):
            super().__init__()
            self.input_transform = TransformNet(3)
            self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
            self.feature_transform = TransformNet(64)
            self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
            self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
            self.head = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )

        def forward(self, points: Any) -> Any:
            x = points.transpose(1, 2)
            x = torch.bmm(self.input_transform(x), x)
            x = self.conv1(x)
            x = torch.bmm(self.feature_transform(x), x)
            x = self.conv2(x)
            x = self.conv3(x).amax(dim=-1)
            return self.head(x)


    class VAE(nn.Module):
        def __init__(self, latent_dim: int = 32):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
            )
            self.mu = nn.Linear(256, latent_dim)
            self.logvar = nn.Linear(256, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 784),
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


    class SurrogateSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, membrane_minus_threshold: Any) -> Any:
            ctx.save_for_backward(membrane_minus_threshold)
            return (membrane_minus_threshold > 0).to(membrane_minus_threshold.dtype)

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            (grad_output,) = grad_outputs
            (input_,) = ctx.saved_tensors
            return grad_output / (1.0 + input_.abs()).pow(2)


    class SpikingConvNet(nn.Module):
        def __init__(
            self,
            num_classes: int = 10,
            time_steps: int = 10,
            beta: float = 0.9,
            threshold: float = 1.0,
        ):
            super().__init__()
            self.time_steps = time_steps
            self.beta = beta
            self.threshold = threshold
            self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc = nn.Linear(64 * 8 * 8, num_classes)

        def _lif(self, current: Any, membrane: Any) -> tuple[Any, Any]:
            membrane = self.beta * membrane + current
            spike = SurrogateSpike.apply(membrane - self.threshold)
            membrane = membrane * (1.0 - spike.detach())
            return spike, membrane

        def forward(self, x: Any) -> Any:
            batch = x.shape[0]
            mem1 = x.new_zeros(batch, 32, x.shape[-2], x.shape[-1])
            pooled_shape = F.avg_pool2d(
                x.new_zeros(batch, 1, x.shape[-2], x.shape[-1]), 2
            ).shape
            mem2 = x.new_zeros(batch, 64, pooled_shape[-2], pooled_shape[-1])
            mem3 = x.new_zeros(batch, self.fc.out_features)
            out_sum = x.new_zeros(batch, self.fc.out_features)
            for _ in range(self.time_steps):
                spike1, mem1 = self._lif(self.conv1(x), mem1)
                pooled1 = F.avg_pool2d(spike1, 2)
                spike2, mem2 = self._lif(self.conv2(pooled1), mem2)
                pooled2 = F.avg_pool2d(spike2, 2)
                logits = self.fc(pooled2.flatten(1))
                spike3, mem3 = self._lif(logits, mem3)
                out_sum = out_sum + spike3 + logits / self.time_steps
            return out_sum / self.time_steps


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
        def __init__(self, base: int = 32):
            super().__init__()
            self.enc1 = DoubleConv(3, base)
            self.enc2 = DoubleConv(base, base * 2)
            self.enc3 = DoubleConv(base * 2, base * 4)
            self.pool = nn.MaxPool2d(2)
            self.mid = DoubleConv(base * 4, base * 8)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.dec3 = DoubleConv(base * 8, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.dec2 = DoubleConv(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.dec1 = DoubleConv(base * 2, base)
            self.out = nn.Conv2d(base, 1, 1)

        def forward(self, x: Any) -> Any:
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            m = self.mid(self.pool(e3))
            d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out(d1)


    class SelfAttentionBlock(nn.Module):
        def __init__(self, hidden: int, heads: int = 4, dropout: float = 0.1):
            super().__init__()
            if hidden % heads != 0:
                raise ValueError("hidden must be divisible by heads")
            self.heads = heads
            self.head_dim = hidden // heads
            self.qkv = nn.Linear(hidden, hidden * 3)
            self.out = nn.Linear(hidden, hidden)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Any) -> Any:
            batch, tokens, hidden = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q = q.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = self.dropout(torch.softmax(scores, dim=-1))
            context = torch.matmul(attn, v).transpose(1, 2).reshape(batch, tokens, hidden)
            return self.out(context)


    class TransformerTabularBlock(nn.Module):
        def __init__(self, hidden: int, heads: int = 4, dropout: float = 0.1):
            super().__init__()
            self.attn = SelfAttentionBlock(hidden, heads, dropout)
            self.norm1 = nn.LayerNorm(hidden)
            self.ffn = nn.Sequential(
                nn.Linear(hidden, hidden * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden * 4, hidden),
            )
            self.norm2 = nn.LayerNorm(hidden)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Any) -> Any:
            x = self.norm1(x + self.dropout(self.attn(x)))
            return self.norm2(x + self.dropout(self.ffn(x)))


    class SAINT(nn.Module):
        """SAINT-style row/column transformer for tabular classification."""

        def __init__(
            self,
            in_features: int = 14,
            hidden: int = 64,
            depth: int = 2,
            heads: int = 4,
            num_classes: int = 2,
        ):
            super().__init__()
            self.feature_embed = nn.Linear(1, hidden)
            self.column_embedding = nn.Parameter(torch.randn(1, in_features, hidden) * 0.02)
            self.column_blocks = nn.ModuleList(
                TransformerTabularBlock(hidden, heads) for _ in range(depth)
            )
            self.row_blocks = nn.ModuleList(
                TransformerTabularBlock(hidden, heads) for _ in range(depth)
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_classes),
            )

        def forward(self, x: Any) -> Any:
            tokens = self.feature_embed(x.float().unsqueeze(-1)) + self.column_embedding
            for column_block, row_block in zip(self.column_blocks, self.row_blocks):
                tokens = column_block(tokens)
                row_tokens = row_block(tokens.transpose(0, 1)).transpose(0, 1)
                tokens = 0.5 * (tokens + row_tokens)
            return self.head(tokens.mean(dim=1))


    def _squash_capsules(x: Any) -> Any:
        norm_sq = (x * x).sum(dim=-1, keepdim=True)
        return norm_sq / (1.0 + norm_sq) * x / torch.sqrt(norm_sq + 1e-8)


    class PrimaryCapsules(nn.Module):
        def __init__(self, in_channels: int, capsule_dim: int, capsule_channels: int):
            super().__init__()
            self.capsule_dim = capsule_dim
            self.capsules = nn.Conv2d(
                in_channels,
                capsule_dim * capsule_channels,
                kernel_size=9,
                stride=2,
            )

        def forward(self, x: Any) -> Any:
            batch = x.shape[0]
            x = self.capsules(x)
            x = x.view(batch, self.capsule_dim, -1).transpose(1, 2)
            return _squash_capsules(x)


    class CapsNet(nn.Module):
        def __init__(
            self,
            num_classes: int = 10,
            primary_dim: int = 8,
            digit_dim: int = 16,
            routing_iters: int = 3,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.routing_iters = routing_iters
            self.conv = nn.Sequential(
                nn.Conv2d(1, 256, kernel_size=9),
                nn.ReLU(),
            )
            self.primary = PrimaryCapsules(256, primary_dim, capsule_channels=32)
            self.num_primary_caps = 32 * 6 * 6
            self.route_weights = nn.Parameter(
                0.01
                * torch.randn(
                    1,
                    self.num_primary_caps,
                    num_classes,
                    digit_dim,
                    primary_dim,
                )
            )
            self.decoder = nn.Sequential(
                nn.Linear(num_classes * digit_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 784),
                nn.Sigmoid(),
            )

        @staticmethod
        def squash(x: Any) -> Any:
            return _squash_capsules(x)

        def forward(self, x: Any) -> Any:
            batch = x.shape[0]
            primary = self.primary(self.conv(x))
            votes = torch.einsum(
                "bip,bicdp->bicd",
                primary,
                self.route_weights.expand(batch, -1, -1, -1, -1),
            )
            logits = votes.new_zeros(batch, self.num_primary_caps, self.num_classes)
            outputs = votes.new_zeros(batch, self.num_classes, votes.shape[-1])
            for iteration in range(self.routing_iters):
                coeffs = torch.softmax(logits, dim=-1)
                outputs = self.squash((coeffs.unsqueeze(-1) * votes).sum(dim=1))
                if iteration < self.routing_iters - 1:
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
        def __init__(self, hidden_channels: int = 64, horizon: int = 10):
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

    LeNet5 = M5 = LSTMForecaster = TextCNN = GCN = TabNet = MPNN = ActorCritic = object
    LSTMAutoencoder = DistilBertClassifier = DQN = PPOPolicy = AttentiveFP = GIN = object
    TCNForecaster = GRUForecaster = PointNet = VAE = SpikingConvNet = TinyUNet = object
    SAINT = CapsNet = ConvLSTM = object


LeNet5 = cast(Any, LeNet5)
M5 = cast(Any, M5)
LSTMForecaster = cast(Any, LSTMForecaster)
TextCNN = cast(Any, TextCNN)
GCN = cast(Any, GCN)
TabNet = cast(Any, TabNet)
MPNN = cast(Any, MPNN)
ActorCritic = cast(Any, ActorCritic)
LSTMAutoencoder = cast(Any, LSTMAutoencoder)
DistilBertClassifier = cast(Any, DistilBertClassifier)
DQN = cast(Any, DQN)
PPOPolicy = cast(Any, PPOPolicy)
AttentiveFP = cast(Any, AttentiveFP)
GIN = cast(Any, GIN)
TCNForecaster = cast(Any, TCNForecaster)
GRUForecaster = cast(Any, GRUForecaster)
PointNet = cast(Any, PointNet)
VAE = cast(Any, VAE)
SpikingConvNet = cast(Any, SpikingConvNet)
TinyUNet = cast(Any, TinyUNet)
SAINT = cast(Any, SAINT)
CapsNet = cast(Any, CapsNet)
ConvLSTM = cast(Any, ConvLSTM)


def _build_resnet18_cifar10(**_: Any) -> Any:
    torchvision_models = cast(Any, __import__("torchvision.models", fromlist=["models"]))
    model = torchvision_models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def _build_mobilenetv2_cifar10(**_: Any) -> Any:
    torchvision_models = cast(Any, __import__("torchvision.models", fromlist=["models"]))
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
    "tabnet": lambda num_classes=2, **_: _construct(TabNet, num_classes=num_classes),
    "mpnn": lambda **_: MPNN(),
    "actor_critic": lambda **_: ActorCritic(),
    "lstm_autoencoder": lambda **_: LSTMAutoencoder(),
    "distilbert": lambda num_classes=2, **_: _construct(DistilBertClassifier, num_classes=num_classes),
    "dqn_lunarlander": lambda **_: DQN(),
    "ppo_bipedalwalker": lambda **_: PPOPolicy(),
    "attentivefp_freesolv": lambda **_: AttentiveFP(),
    "gin_imdbb": lambda num_classes=2, **_: _construct(GIN, num_classes=num_classes),
    "tcn_forecaster": lambda **_: _construct(TCNForecaster, input_size=7),
    "gru_forecaster": lambda **_: _construct(GRUForecaster, input_size=21),
    "pointnet_modelnet40": lambda num_classes=40, **_: _construct(PointNet, num_classes=num_classes),
    "vae_mnist": lambda **_: VAE(),
    "snn_nmnist": lambda num_classes=10, **_: _construct(SpikingConvNet, num_classes=num_classes),
    "unet_isic": lambda **_: TinyUNet(),
    "resnet18_cifar10": _build_resnet18_cifar10,
    "mobilenetv2_cifar10": _build_mobilenetv2_cifar10,
    "saint_adult": lambda num_classes=2, **_: _construct(SAINT, num_classes=num_classes),
    "capsnet_mnist": lambda num_classes=10, **_: _construct(CapsNet, num_classes=num_classes),
    "convlstm_movingmnist": lambda **_: ConvLSTM(),
}


def build_model(model_key: str, **kwargs: Any) -> Any:
    if model_key not in MODEL_FACTORIES:
        raise KeyError(f"Unknown model key: {model_key}")
    _ensure_torch()
    return MODEL_FACTORIES[model_key](**kwargs)
