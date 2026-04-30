from __future__ import annotations

import csv
import os
import tarfile
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .compat import require_torch

DATA_ROOT_ENV = "DQB_DATA_ROOT"
DEFAULT_DATA_ROOT = "data"
ETTH1_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)
ADULT_URLS = {
    "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
}
CORA_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
ESOL_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
)
MITBIH_RECORDS = ["100", "101", "103", "105", "106", "108", "109", "111"]
SPEECH_COMMAND_LABELS = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "zero",
    "one",
)


@dataclass
class TaskBundle:
    train_loader: Any
    val_loader: Any
    test_loader: Any
    metric_name: str
    metric_direction: str
    input_description: str


def _data_root() -> Path:
    return Path(os.environ.get(DATA_ROOT_ENV, DEFAULT_DATA_ROOT)).expanduser()


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"Downloading {url} -> {destination}")
        urllib.request.urlretrieve(url, destination)
    return destination


def _require_dependency(import_name: str, package_name: str | None = None) -> Any:
    try:
        return __import__(import_name)
    except ImportError as exc:
        package = package_name or import_name
        raise RuntimeError(
            f"Real dataset loading for this benchmark requires `{package}`. "
            f"Install project dependencies with `uv sync` or add `{package}` to the environment."
        ) from exc


def _make_loader(
    dataset: Any, batch_size: int, shuffle: bool = False, *, num_workers: int = 2
) -> Any:
    """Build a DataLoader tuned for Apple Silicon MPS.

    ``pin_memory`` is explicitly *off*: it only benefits CUDA host-to-device
    transfers and adds unnecessary overhead on MPS, which uses unified memory.
    ``persistent_workers`` keeps worker processes alive between epochs so the
    spawn cost is paid only once per training run.
    """
    torch = require_torch()
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


def _split_dataset(
    dataset: Any, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> tuple[Any, Any, Any]:
    torch = require_torch()
    total = len(dataset)
    if total < 3:
        raise ValueError(
            "Need at least three samples to build train/validation/test splits."
        )
    train_size = max(1, int(total * train_ratio))
    val_size = max(1, int(total * val_ratio))
    test_size = max(1, total - train_size - val_size)
    overflow = train_size + val_size + test_size - total
    if overflow > 0:
        train_size = max(1, train_size - overflow)
    return torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )


def _bundle_from_splits(
    train_ds: Any,
    val_ds: Any,
    test_ds: Any,
    batch_size: int,
    metric_name: str,
    metric_direction: str,
    input_description: str,
    *,
    num_workers: int = 2,
) -> TaskBundle:
    return TaskBundle(
        _make_loader(train_ds, batch_size, shuffle=True, num_workers=num_workers),
        _make_loader(val_ds, batch_size, num_workers=num_workers),
        _make_loader(test_ds, batch_size, num_workers=num_workers),
        metric_name,
        metric_direction,
        input_description,
    )


def _bundle_from_dataset(
    dataset: Any,
    batch_size: int,
    metric_name: str,
    metric_direction: str,
    input_description: str,
    *,
    num_workers: int = 2,
) -> TaskBundle:
    return _bundle_from_splits(
        *_split_dataset(dataset),
        batch_size,
        metric_name,
        metric_direction,
        input_description,
        num_workers=num_workers,
    )


def _build_mnist(batch_size: int) -> TaskBundle:
    torchvision = _require_dependency("torchvision")
    transforms = __import__("torchvision.transforms", fromlist=["transforms"])
    root = _data_root() / "mnist"
    root.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    train_full = torchvision.datasets.MNIST(
        root=str(root), train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.MNIST(
        root=str(root), train=False, download=True, transform=transform
    )
    train_ds, val_ds = require_torch().utils.data.random_split(
        train_full,
        [55_000, 5_000],
        generator=require_torch().Generator().manual_seed(42),
    )
    return _bundle_from_splits(
        train_ds,
        val_ds,
        test_ds,
        batch_size,
        "Accuracy",
        "maximize",
        "MNIST handwritten digit images",
    )


class _SpeechCommands12:
    def __init__(self, subset: str) -> None:
        torchaudio = _require_dependency("torchaudio")
        root = _data_root() / "speechcommands"
        root.mkdir(parents=True, exist_ok=True)
        base = torchaudio.datasets.SPEECHCOMMANDS(
            str(root), download=True, subset=subset
        )
        labels = {label: index for index, label in enumerate(SPEECH_COMMAND_LABELS)}
        self.base = base
        self.labels = labels
        self.indices = [index for index, item in enumerate(base) if item[2] in labels]
        self.target_len = 16_000
        # Do NOT store self.torch — the torch module object is not picklable, which
        # would prevent DataLoader from serialising this dataset for worker processes.

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        _torch = (
            require_torch()
        )  # lightweight: just returns the already-imported module
        waveform, sample_rate, label, *_ = self.base[self.indices[index]]
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.target_len:
            waveform = _torch.nn.functional.interpolate(
                waveform.unsqueeze(0),
                size=self.target_len,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
        if waveform.shape[-1] < self.target_len:
            waveform = _torch.nn.functional.pad(
                waveform, (0, self.target_len - waveform.shape[-1])
            )
        waveform = waveform[:, : self.target_len]
        return waveform, _torch.tensor(self.labels[label], dtype=_torch.long)


def _build_speechcommands(batch_size: int) -> TaskBundle:
    train_ds = _SpeechCommands12("training")
    val_ds = _SpeechCommands12("validation")
    test_ds = _SpeechCommands12("testing")
    return _bundle_from_splits(
        train_ds,
        val_ds,
        test_ds,
        batch_size,
        "Accuracy",
        "maximize",
        "SpeechCommands 12-class keyword audio",
    )


class _TensorRowsDataset:
    def __init__(self, *tensors: Any) -> None:
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        return tuple(tensor[index] for tensor in self.tensors)


def _build_etth1(batch_size: int) -> TaskBundle:
    torch = require_torch()
    path = _download(ETTH1_URL, _data_root() / "etth1" / "ETTh1.csv")
    rows: list[float] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(float(row["OT"]))
    values = torch.tensor(rows, dtype=torch.float32)
    mean = values.mean()
    std = values.std().clamp_min(1e-6)
    values = (values - mean) / std
    seq_len = 24
    x = torch.stack(
        [values[index : index + seq_len] for index in range(len(values) - seq_len)]
    ).unsqueeze(-1)
    y = torch.stack([values[index + seq_len] for index in range(len(values) - seq_len)])
    return _bundle_from_dataset(
        _TensorRowsDataset(x, y),
        batch_size,
        "MAE",
        "minimize",
        "ETTh1 hourly oil temperature forecasting windows",
    )


def _tokenize(text: str) -> list[str]:
    return "".join(char.lower() if char.isalnum() else " " for char in text).split()


def _build_vocab(texts: Iterable[str], vocab_size: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_tokenize(text))
    return {
        token: index + 1
        for index, (token, _) in enumerate(counts.most_common(vocab_size - 1))
    }


def _encode_texts(texts: Iterable[str], vocab: dict[str, int], seq_len: int) -> Any:
    torch = require_torch()
    encoded = []
    for text in texts:
        ids = [vocab.get(token, 0) for token in _tokenize(text)[:seq_len]]
        ids.extend([0] * (seq_len - len(ids)))
        encoded.append(ids)
    return torch.tensor(encoded, dtype=torch.long)


def _hf_dataset_cache() -> str:
    cache = _data_root() / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    return str(cache)


def _build_ag_news(batch_size: int) -> TaskBundle:
    torch = require_torch()
    datasets = _require_dependency("datasets")
    loaded = datasets.load_dataset("ag_news", cache_dir=_hf_dataset_cache())
    vocab = _build_vocab(loaded["train"]["text"], 5_000)
    train_texts = loaded["train"]["text"]
    train_labels = torch.tensor(loaded["train"]["label"], dtype=torch.long)
    x_train = _encode_texts(train_texts, vocab, 64)
    train_full = _TensorRowsDataset(x_train, train_labels)
    train_ds, val_ds, _ = _split_dataset(train_full, train_ratio=0.9, val_ratio=0.1)
    x_test = _encode_texts(loaded["test"]["text"], vocab, 64)
    y_test = torch.tensor(loaded["test"]["label"], dtype=torch.long)
    return _bundle_from_splits(
        train_ds,
        val_ds,
        _TensorRowsDataset(x_test, y_test),
        batch_size,
        "Accuracy",
        "maximize",
        "AG News tokenized article titles and descriptions",
    )


def _build_sst2(batch_size: int) -> TaskBundle:
    torch = require_torch()
    datasets = _require_dependency("datasets")
    loaded = datasets.load_dataset("glue", "sst2", cache_dir=_hf_dataset_cache())
    vocab = _build_vocab(loaded["train"]["sentence"], 5_000)
    x_train = _encode_texts(loaded["train"]["sentence"], vocab, 96)
    y_train = torch.tensor(loaded["train"]["label"], dtype=torch.long)
    train_full = _TensorRowsDataset(x_train, y_train)
    train_ds, val_ds, _ = _split_dataset(train_full, train_ratio=0.9, val_ratio=0.1)
    x_test = _encode_texts(loaded["validation"]["sentence"], vocab, 96)
    y_test = torch.tensor(loaded["validation"]["label"], dtype=torch.long)
    return _bundle_from_splits(
        train_ds,
        val_ds,
        _TensorRowsDataset(x_test, y_test),
        batch_size,
        "Accuracy",
        "maximize",
        "SST-2 tokenized sentiment sentences",
    )


class _CoraEgoDataset:
    """Module-level ego-graph dataset for Cora node classification.

    Storing all data as instance attributes (rather than capturing them via a
    closure inside ``_build_cora``) makes this class picklable, which lets
    ``DataLoader`` serialise it safely for multi-process worker prefetching.
    """

    def __init__(self, adjacency: Any, x_all: Any, y_all: Any) -> None:
        self.adjacency = adjacency
        self.x_all = x_all
        self.y_all = y_all

    def __len__(self) -> int:
        return int(self.y_all.shape[0])

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        torch = require_torch()
        neighbors = self.adjacency[index].nonzero().flatten()[:50]
        if len(neighbors) < 50:
            pad = neighbors.new_full((50 - len(neighbors),), index)
            neighbors = torch.cat([neighbors, pad])
        sub_x = self.x_all[neighbors]
        sub_adj = self.adjacency[neighbors][:, neighbors]
        return sub_x, sub_adj, self.y_all[index]


def _build_cora(batch_size: int) -> TaskBundle:
    torch = require_torch()
    root = _data_root() / "cora"
    archive = _download(CORA_URL, root / "cora.tgz")
    content = root / "cora" / "cora.content"
    cites = root / "cora" / "cora.cites"
    if not content.exists() or not cites.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(root)

    paper_ids: list[str] = []
    features: list[list[float]] = []
    labels_raw: list[str] = []
    with content.open() as fh:
        for line in fh:
            parts = line.strip().split()
            paper_ids.append(parts[0])
            features.append([float(value) for value in parts[1:-1]])
            labels_raw.append(parts[-1])
    id_to_idx = {paper_id: index for index, paper_id in enumerate(paper_ids)}
    label_to_idx = {label: index for index, label in enumerate(sorted(set(labels_raw)))}
    x_all = torch.tensor(features, dtype=torch.float32)
    y_all = torch.tensor(
        [label_to_idx[label] for label in labels_raw], dtype=torch.long
    )
    adjacency = torch.eye(len(paper_ids), dtype=torch.float32)
    with cites.open() as fh:
        for line in fh:
            src, dst = line.strip().split()
            if src in id_to_idx and dst in id_to_idx:
                i, j = id_to_idx[src], id_to_idx[dst]
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0

    return _bundle_from_dataset(
        _CoraEgoDataset(adjacency, x_all, y_all),
        batch_size,
        "Accuracy",
        "maximize",
        "Cora citation-network ego graphs for node labels",
    )


def _parse_adult_file(path: Path) -> list[list[str]]:
    rows = []
    with path.open(newline="") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw or raw.startswith("|"):
                continue
            rows.append([part.strip().rstrip(".") for part in raw.split(",")])
    return rows


def _build_adult(batch_size: int) -> TaskBundle:
    torch = require_torch()
    root = _data_root() / "adult"
    for filename, url in ADULT_URLS.items():
        _download(url, root / filename)
    train_rows = _parse_adult_file(root / "adult.data")
    test_rows = _parse_adult_file(root / "adult.test")
    feature_count = 14
    encoders: list[dict[str, int]] = [{} for _ in range(feature_count)]
    numeric_columns = {0, 2, 4, 10, 11, 12}

    def encode(rows: list[list[str]]) -> tuple[list[list[float]], list[int]]:
        values = []
        labels = []
        for row in rows:
            encoded = []
            for col in range(feature_count):
                value = row[col]
                if col in numeric_columns:
                    encoded.append(float(value) if value != "?" else 0.0)
                else:
                    mapping = encoders[col]
                    if value not in mapping:
                        mapping[value] = len(mapping) + 1
                    encoded.append(float(mapping[value]))
            values.append(encoded)
            labels.append(1 if row[-1] == ">50K" else 0)
        return values, labels

    train_x_raw, train_y_raw = encode(train_rows)
    test_x_raw, test_y_raw = encode(test_rows)
    train_x = torch.tensor(train_x_raw, dtype=torch.float32)
    test_x = torch.tensor(test_x_raw, dtype=torch.float32)
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_ds, val_ds, _ = _split_dataset(
        _TensorRowsDataset(
            (train_x - mean) / std, torch.tensor(train_y_raw, dtype=torch.long)
        ),
        train_ratio=0.9,
        val_ratio=0.1,
    )
    test_ds = _TensorRowsDataset(
        (test_x - mean) / std, torch.tensor(test_y_raw, dtype=torch.long)
    )
    return _bundle_from_splits(
        train_ds,
        val_ds,
        test_ds,
        batch_size,
        "Accuracy",
        "maximize",
        "Adult Income census features",
    )


def _smiles_to_graph(smiles: str) -> tuple[Any, Any]:
    torch = require_torch()
    atoms: list[str] = []
    edges: list[tuple[int, int]] = []
    last_atom: int | None = None
    ring_open: dict[str, int] = {}
    index = 0
    while index < len(smiles):
        char = smiles[index]
        token = None
        if index + 1 < len(smiles) and smiles[index : index + 2] in {"Cl", "Br"}:
            token = smiles[index : index + 2]
            index += 2
        elif char.isalpha():
            token = char.upper()
            index += 1
        elif char.isdigit() and last_atom is not None:
            if char in ring_open:
                edges.append((ring_open.pop(char), last_atom))
            else:
                ring_open[char] = last_atom
            index += 1
            continue
        else:
            index += 1
            continue
        atom_index = len(atoms)
        atoms.append(token)
        if last_atom is not None:
            edges.append((last_atom, atom_index))
        last_atom = atom_index
    if not atoms:
        atoms = ["C"]
    atom_types = ["C", "N", "O", "S", "F", "CL", "BR", "I"]
    x = torch.zeros((24, 9), dtype=torch.float32)
    for atom_index, atom in enumerate(atoms[:24]):
        feature_index = atom_types.index(atom) if atom in atom_types else 8
        x[atom_index, feature_index] = 1.0
    adjacency = torch.eye(24, dtype=torch.float32)
    for src, dst in edges:
        if src < 24 and dst < 24:
            adjacency[src, dst] = 1.0
            adjacency[dst, src] = 1.0
    return x, adjacency


def _build_esol(batch_size: int) -> TaskBundle:
    torch = require_torch()
    path = _download(ESOL_URL, _data_root() / "esol" / "delaney-processed.csv")
    node_features = []
    adjacencies = []
    labels = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smiles = row.get("smiles") or row.get("smile") or row.get("SMILES")
            target = row.get("measured log solubility in mols per litre") or row.get(
                "ESOL predicted log solubility in mols per litre"
            )
            if smiles is None or target is None:
                continue
            x, adjacency = _smiles_to_graph(smiles)
            node_features.append(x)
            adjacencies.append(adjacency)
            labels.append(float(target))
    return _bundle_from_dataset(
        _TensorRowsDataset(
            torch.stack(node_features),
            torch.stack(adjacencies),
            torch.tensor(labels, dtype=torch.float32),
        ),
        batch_size,
        "RMSE",
        "minimize",
        "ESOL MoleculeNet molecular graphs from SMILES",
    )


def _build_cartpole(batch_size: int) -> TaskBundle:
    torch = require_torch()
    gymnasium = _require_dependency("gymnasium", "gymnasium[classic-control]")
    numpy = _require_dependency("numpy")
    cache = _data_root() / "cartpole" / "heuristic_rollouts.pt"
    if cache.exists():
        payload = torch.load(cache, map_location="cpu")
        x, y = payload["x"], payload["y"]
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        env = gymnasium.make("CartPole-v1")
        observations = []
        actions = []
        obs, _ = env.reset(seed=42)
        while len(observations) < 10_000:
            action = 1 if (obs[2] + 0.25 * obs[3]) > 0 else 0
            observations.append(obs)
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()
        x = torch.tensor(numpy.array(observations), dtype=torch.float32)
        y = torch.tensor(actions, dtype=torch.long)
        torch.save({"x": x, "y": y}, cache)
    return _bundle_from_dataset(
        _TensorRowsDataset(x, y),
        batch_size,
        "Reward",
        "maximize",
        "CartPole-v1 observations labeled by a stabilizing policy",
    )


def _build_mitbih(batch_size: int) -> TaskBundle:
    torch = require_torch()
    wfdb = _require_dependency("wfdb")
    root = _data_root() / "mit-bih"
    if not all((root / f"{record}.dat").exists() for record in MITBIH_RECORDS):
        root.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database("mitdb", dl_dir=str(root), records=MITBIH_RECORDS)
    windows = []
    labels = []
    half_width = 64
    for record in MITBIH_RECORDS:
        signal = wfdb.rdrecord(str(root / record)).p_signal[:, 0]
        annotations = wfdb.rdann(str(root / record), "atr")
        for sample, symbol in zip(annotations.sample, annotations.symbol):
            start = sample - half_width
            end = sample + half_width
            if start < 0 or end > len(signal):
                continue
            window = torch.tensor(signal[start:end], dtype=torch.float32)
            window = (window - window.mean()) / window.std().clamp_min(1e-6)
            windows.append(window.unsqueeze(-1))
            labels.append(0 if symbol == "N" else 1)
    return _bundle_from_dataset(
        _TensorRowsDataset(
            torch.stack(windows),
            torch.stack(windows),
            torch.tensor(labels, dtype=torch.long),
        ),
        batch_size,
        "AUC",
        "maximize",
        "MIT-BIH Arrhythmia ECG beat windows",
    )


# Per-model batch sizes tuned for Apple Silicon MPS throughput.
# Larger batches amortise Python-loop and host-to-device transfer overhead,
# keeping the GPU busy for longer between CPU round-trips.  Each value was
# chosen to be well within the M3 Pro's unified-memory budget while giving
# the GPU enough work per step to approach saturation.
_BATCH_SIZES: dict[str, int] = {
    "lenet5": 512,  # MNIST 28×28 greyscale — negligible per-sample cost
    "m5": 128,  # SpeechCommands 16 K-sample 1-D waveform
    "lstm_forecaster": 256,  # ETTh1 short sliding windows
    "textcnn": 256,  # AG News fixed-length token sequences
    "gcn": 32,  # Cora ego-graphs — adjacency matrix limits batch size
    "tabnet": 512,  # Adult Income 14-feature tabular rows
    "mpnn": 32,  # ESOL molecular graphs — variable topology
    "actor_critic": 1024,  # CartPole 4-D observations — negligible per-sample cost
    "lstm_autoencoder": 256,  # MIT-BIH 128-sample ECG windows
    "distilbert": 128,  # SST-2 token sequences
}


def build_task_bundle(model_key: str, batch_size: int | None = None) -> TaskBundle:
    """Build the data loaders for *model_key*.

    Pass an explicit ``batch_size`` to override the MPS-tuned default in
    ``_BATCH_SIZES`` (useful for smoke tests or ablation studies).
    """
    require_torch()
    builders = {
        "lenet5": _build_mnist,
        "m5": _build_speechcommands,
        "lstm_forecaster": _build_etth1,
        "textcnn": _build_ag_news,
        "gcn": _build_cora,
        "tabnet": _build_adult,
        "mpnn": _build_esol,
        "actor_critic": _build_cartpole,
        "lstm_autoencoder": _build_mitbih,
        "distilbert": _build_sst2,
    }
    if model_key not in builders:
        raise KeyError(f"Unknown model key: {model_key}")
    effective_batch_size = (
        batch_size if batch_size is not None else _BATCH_SIZES.get(model_key, 64)
    )
    return builders[model_key](effective_batch_size)
