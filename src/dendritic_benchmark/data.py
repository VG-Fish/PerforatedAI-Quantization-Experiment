from __future__ import annotations

import csv
import gzip
import os
import tarfile
import urllib.request
import zipfile
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
ETTM1_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"
)
WEATHER_URL = (
    "https://huggingface.co/datasets/dunzane/time-series-dataset/raw/main/weather/weather.csv"
)
ADULT_URLS = {
    "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
}
CORA_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
ESOL_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
)
FREESOLV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv"
IMDBB_URL = "https://www.chrsmrrs.com/graphkerneldatasets/IMDB-BINARY.zip"
MOVING_MNIST_URL = "https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy.gz"
ISIC_SAMPLE_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip"
)
ISIC_MASK_SAMPLE_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip"
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


def _extract_zip(archive: Path, destination: Path) -> None:
    marker = destination / ".extracted"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(destination)
    marker.write_text("ok\n")


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


def _build_cifar10(batch_size: int) -> TaskBundle:
    torchvision = _require_dependency("torchvision")
    transforms = __import__("torchvision.transforms", fromlist=["transforms"])
    root = _data_root() / "cifar10"
    root.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    train_full = torchvision.datasets.CIFAR10(
        root=str(root), train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=str(root), train=False, download=True, transform=test_transform
    )
    train_ds, val_ds, _ = _split_dataset(train_full, train_ratio=0.9, val_ratio=0.1)
    return _bundle_from_splits(
        train_ds,
        val_ds,
        test_ds,
        batch_size,
        "Accuracy",
        "maximize",
        "CIFAR-10 32x32 natural images",
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


def _build_multivariate_forecast(
    batch_size: int,
    *,
    url: str,
    subdir: str,
    filename: str,
    seq_len: int,
    horizon: int,
    input_description: str,
) -> TaskBundle:
    torch = require_torch()
    path = _download(url, _data_root() / subdir / filename)
    rows: list[list[float]] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            values = []
            for key, value in row.items():
                if key.lower() == "date":
                    continue
                try:
                    values.append(float(value))
                except ValueError:
                    pass
            if values:
                rows.append(values)
    values_t = torch.tensor(rows, dtype=torch.float32)
    mean = values_t.mean(dim=0, keepdim=True)
    std = values_t.std(dim=0, keepdim=True).clamp_min(1e-6)
    values_t = (values_t - mean) / std
    xs = []
    ys = []
    limit = len(values_t) - seq_len - horizon + 1
    for index in range(limit):
        xs.append(values_t[index : index + seq_len])
        ys.append(values_t[index + seq_len : index + seq_len + horizon])
    return _bundle_from_dataset(
        _TensorRowsDataset(torch.stack(xs), torch.stack(ys)),
        batch_size,
        "MAE",
        "minimize",
        input_description,
    )


def _build_ettm1(batch_size: int) -> TaskBundle:
    return _build_multivariate_forecast(
        batch_size,
        url=ETTM1_URL,
        subdir="ettm1",
        filename="ETTm1.csv",
        seq_len=96,
        horizon=24,
        input_description="ETTm1 15-minute multivariate transformer-temperature windows",
    )


def _build_weather(batch_size: int) -> TaskBundle:
    torch = require_torch()
    datasets = _require_dependency("datasets")
    loaded = datasets.load_dataset(
        "dunzane/time-series-dataset", "Weather", cache_dir=_hf_dataset_cache()
    )
    split = loaded["train"]
    columns = [
        name
        for name in split.column_names
        if name.lower() != "date" and split.features[name].dtype in {"float32", "float64", "int32", "int64"}
    ]
    rows = [[float(row[name]) for name in columns] for row in split]
    values_t = torch.tensor(rows, dtype=torch.float32)
    mean = values_t.mean(dim=0, keepdim=True)
    std = values_t.std(dim=0, keepdim=True).clamp_min(1e-6)
    values_t = (values_t - mean) / std
    seq_len = 96
    horizon = 24
    xs = []
    ys = []
    for index in range(len(values_t) - seq_len - horizon + 1):
        xs.append(values_t[index : index + seq_len])
        ys.append(values_t[index + seq_len : index + seq_len + horizon])
    return _bundle_from_dataset(
        _TensorRowsDataset(torch.stack(xs), torch.stack(ys)),
        batch_size,
        "MAE",
        "minimize",
        "Weather multivariate meteorological forecasting windows",
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


def _build_freesolv(batch_size: int) -> TaskBundle:
    torch = require_torch()
    path = _download(FREESOLV_URL, _data_root() / "freesolv" / "SAMPL.csv")
    node_features = []
    adjacencies = []
    labels = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smiles = row.get("smiles") or row.get("SMILES")
            target = (
                row.get("expt")
                or row.get("measured log solubility in mols per litre")
                or row.get("y")
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
        "FreeSolv molecular hydration free-energy graphs from SMILES",
    )


def _read_tu_indicator(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def _build_imdbb(batch_size: int) -> TaskBundle:
    torch = require_torch()
    root = _data_root() / "imdb_binary"
    archive = _download(IMDBB_URL, root / "IMDB-BINARY.zip")
    _extract_zip(archive, root)
    dataset_dir = root / "IMDB-BINARY"
    graph_indicator = _read_tu_indicator(dataset_dir / "IMDB-BINARY_graph_indicator.txt")
    labels_raw = _read_tu_indicator(dataset_dir / "IMDB-BINARY_graph_labels.txt")
    edges: list[tuple[int, int]] = []
    with (dataset_dir / "IMDB-BINARY_A.txt").open() as fh:
        for line in fh:
            left, right = line.replace(" ", "").strip().split(",")
            edges.append((int(left) - 1, int(right) - 1))
    graph_nodes: dict[int, list[int]] = {}
    for node_index, graph_id in enumerate(graph_indicator):
        graph_nodes.setdefault(graph_id, []).append(node_index)
    max_nodes = 96
    features = []
    adjacencies = []
    labels = []
    edge_set = set(edges) | {(b, a) for a, b in edges}
    for graph_id, nodes in sorted(graph_nodes.items()):
        nodes = nodes[:max_nodes]
        node_map = {node: i for i, node in enumerate(nodes)}
        adjacency = torch.eye(max_nodes, dtype=torch.float32)
        degree = torch.zeros(max_nodes, dtype=torch.float32)
        for src, dst in edge_set:
            if src in node_map and dst in node_map:
                i, j = node_map[src], node_map[dst]
                adjacency[i, j] = 1.0
                degree[i] += 1.0
        x = torch.zeros((max_nodes, 8), dtype=torch.float32)
        x[: len(nodes), 0] = 1.0
        x[:, 1] = degree / degree.clamp_min(1.0).max().clamp_min(1.0)
        features.append(x)
        adjacencies.append(adjacency)
        labels.append(1 if labels_raw[graph_id - 1] > 0 else 0)
    return _bundle_from_dataset(
        _TensorRowsDataset(
            torch.stack(features),
            torch.stack(adjacencies),
            torch.tensor(labels, dtype=torch.long),
        ),
        batch_size,
        "Accuracy",
        "maximize",
        "IMDB-Binary social-network graph classification",
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


def _build_lunarlander(batch_size: int) -> TaskBundle:
    torch = require_torch()
    gymnasium = _require_dependency("gymnasium", "gymnasium[box2d]")
    numpy = _require_dependency("numpy")
    cache = _data_root() / "lunarlander" / "heuristic_rollouts.pt"
    if cache.exists():
        payload = torch.load(cache, map_location="cpu")
        x, y = payload["x"], payload["y"]
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            env = gymnasium.make("LunarLander-v3")
        except Exception:
            env = gymnasium.make("LunarLander-v2")
        observations = []
        actions = []
        obs, _ = env.reset(seed=42)
        while len(observations) < 40_000:
            x_pos, y_pos, x_vel, y_vel, angle, angular_vel, left_contact, right_contact = obs
            if left_contact or right_contact:
                action = 0 if abs(x_vel) < 0.2 else (3 if x_vel < 0 else 1)
            elif abs(angle) > 0.12:
                action = 1 if angle > 0 else 3
            elif y_vel < -0.25 or y_pos < 0.6:
                action = 2
            elif abs(x_pos) > 0.15:
                action = 3 if x_pos < 0 else 1
            else:
                action = 0
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
        "LunarLander-v3 observations labeled by a stabilizing heuristic policy",
    )


def _build_bipedalwalker(batch_size: int) -> TaskBundle:
    torch = require_torch()
    gymnasium = _require_dependency("gymnasium", "gymnasium[box2d]")
    numpy = _require_dependency("numpy")
    cache = _data_root() / "bipedalwalker" / "heuristic_rollouts.pt"
    if cache.exists():
        payload = torch.load(cache, map_location="cpu")
        x, y = payload["x"], payload["y"]
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        env = gymnasium.make("BipedalWalker-v3")
        observations = []
        actions = []
        obs, _ = env.reset(seed=42)
        while len(observations) < 50_000:
            hull_angle = obs[0]
            hull_angular_velocity = obs[1]
            hip_drive = -0.6 * hull_angle - 0.2 * hull_angular_velocity
            action = numpy.array(
                [
                    numpy.clip(hip_drive + 0.35, -1.0, 1.0),
                    0.45,
                    numpy.clip(-hip_drive + 0.35, -1.0, 1.0),
                    0.45,
                ],
                dtype=numpy.float32,
            )
            observations.append(obs)
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()
        x = torch.tensor(numpy.array(observations), dtype=torch.float32)
        y = torch.tensor(numpy.array(actions), dtype=torch.float32)
        torch.save({"x": x, "y": y}, cache)
    return _bundle_from_dataset(
        _TensorRowsDataset(x, y),
        batch_size,
        "Reward",
        "maximize",
        "BipedalWalker-v3 observations labeled by a continuous-action heuristic policy",
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


class _ModelNet40Dataset:
    def __init__(self, train: bool) -> None:
        self.root = _data_root() / "modelnet40"
        archive = _download("http://modelnet.cs.princeton.edu/ModelNet40.zip", self.root / "ModelNet40.zip")
        raw_root = self.root / "raw"
        if not raw_root.exists() or not any(raw_root.iterdir()):
            _extract_zip(archive, self.root / "_extracted")
            extracted = self.root / "_extracted" / "ModelNet40"
            if extracted.exists():
                extracted.rename(raw_root)
        split = "train" if train else "test"
        categories = sorted(path.name for path in raw_root.iterdir() if path.is_dir())
        self.class_to_idx = {category: index for index, category in enumerate(categories)}
        self.samples: list[tuple[Path, int]] = []
        for category in categories:
            for path in sorted((raw_root / category / split).glob("*.off")):
                self.samples.append((path, self.class_to_idx[category]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        torch = require_torch()
        path, label = self.samples[index]
        with path.open() as fh:
            header = fh.readline().strip()
            if header != "OFF":
                counts = header[3:].strip().split()
            else:
                counts = fh.readline().strip().split()
            vertex_count = int(counts[0])
            vertices = []
            for _ in range(vertex_count):
                vertices.append([float(value) for value in fh.readline().split()[:3]])
        points = torch.tensor(vertices, dtype=torch.float32)
        points = points - points.mean(dim=0, keepdim=True)
        points = points / points.norm(dim=1).max().clamp_min(1e-6)
        if len(points) >= 1024:
            choice = torch.linspace(0, len(points) - 1, steps=1024).long()
            points = points[choice]
        else:
            pad = points[torch.arange(1024 - len(points)) % len(points)]
            points = torch.cat([points, pad], dim=0)
        return points, torch.tensor(label, dtype=torch.long)


def _build_modelnet40(batch_size: int) -> TaskBundle:
    train_full = _ModelNet40Dataset(train=True)
    train_ds, val_ds, _ = _split_dataset(train_full, train_ratio=0.9, val_ratio=0.1)
    test_ds = _ModelNet40Dataset(train=False)
    return _bundle_from_splits(
        train_ds,
        val_ds,
        test_ds,
        batch_size,
        "Accuracy",
        "maximize",
        "ModelNet40 1024-point CAD object clouds",
        num_workers=0,
    )


def _build_nmnist(batch_size: int) -> TaskBundle:
    torch = require_torch()
    tonic = _require_dependency("tonic")
    transforms = __import__("tonic.transforms", fromlist=["transforms"])
    transform = transforms.Compose(
        [
            transforms.ToFrame(
                sensor_size=tonic.datasets.NMNIST.sensor_size,
                n_time_bins=10,
            ),
            lambda frames: torch.tensor(frames, dtype=torch.float32).sum(dim=0),
        ]
    )
    root = _data_root() / "nmnist"
    train_full = tonic.datasets.NMNIST(save_to=str(root), train=True, transform=transform)
    test_ds = tonic.datasets.NMNIST(save_to=str(root), train=False, transform=transform)
    train_ds, val_ds, _ = _split_dataset(train_full, train_ratio=0.9, val_ratio=0.1)
    return _bundle_from_splits(
        train_ds,
        val_ds,
        test_ds,
        batch_size,
        "Accuracy",
        "maximize",
        "N-MNIST event-camera spike frames",
        num_workers=0,
    )


class _ISICDataset:
    def __init__(self, root: Path, image_size: int = 128) -> None:
        self.root = root
        self.image_size = image_size
        self.samples = self._discover_pairs()

    def _discover_pairs(self) -> list[tuple[Path, Path]]:
        image_files = [
            path
            for path in self.root.rglob("*.jpg")
            if "superpixel" not in path.name.lower()
        ]
        mask_files = list(self.root.rglob("*segmentation*.png")) + list(
            self.root.rglob("*Segmentation*.png")
        )
        masks_by_stem = {
            mask.name.replace("_segmentation", "").replace("_Segmentation", "").split(".")[0]: mask
            for mask in mask_files
        }
        pairs = []
        for image in image_files:
            key = image.stem
            if key in masks_by_stem:
                pairs.append((image, masks_by_stem[key]))
        if not pairs:
            raise RuntimeError(
                "ISIC files were downloaded, but image/mask pairs could not be matched."
            )
        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        torch = require_torch()
        Image = __import__("PIL.Image", fromlist=["Image"])
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size))
        image_t = torch.tensor(list(image.getdata()), dtype=torch.float32).view(
            self.image_size, self.image_size, 3
        ).permute(2, 0, 1) / 255.0
        mask_t = torch.tensor(list(mask.getdata()), dtype=torch.float32).view(
            1, self.image_size, self.image_size
        ) / 255.0
        return image_t, (mask_t > 0.5).float()


def _build_isic(batch_size: int) -> TaskBundle:
    root = _data_root() / "isic2018"
    image_archive = _download(ISIC_SAMPLE_URL, root / "images.zip")
    mask_archive = _download(ISIC_MASK_SAMPLE_URL, root / "masks.zip")
    _extract_zip(image_archive, root / "images")
    _extract_zip(mask_archive, root / "masks")
    return _bundle_from_dataset(
        _ISICDataset(root),
        batch_size,
        "Dice",
        "maximize",
        "ISIC 2018 Task 1 dermoscopy images and lesion masks",
        num_workers=0,
    )


def _build_moving_mnist(batch_size: int) -> TaskBundle:
    torch = require_torch()
    numpy = _require_dependency("numpy")
    root = _data_root() / "moving_mnist"
    gz_path = _download(MOVING_MNIST_URL, root / "mnist_test_seq.npy.gz")
    npy_path = root / "mnist_test_seq.npy"
    if not npy_path.exists():
        with gzip.open(gz_path, "rb") as src, npy_path.open("wb") as dst:
            dst.write(src.read())
    arr = numpy.load(npy_path)  # (20, 10000, 64, 64)
    arr = torch.tensor(arr, dtype=torch.float32).permute(1, 0, 2, 3).unsqueeze(2) / 255.0
    x = arr[:, :10]
    y = arr[:, 10:20]
    return _bundle_from_dataset(
        _TensorRowsDataset(x, y),
        batch_size,
        "SSIM",
        "maximize",
        "Moving MNIST two-digit video prediction sequences",
        num_workers=0,
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
    "dqn_lunarlander": 1024,
    "ppo_bipedalwalker": 1024,
    "attentivefp_freesolv": 32,
    "gin_imdbb": 32,
    "tcn_forecaster": 128,
    "gru_forecaster": 128,
    "pointnet_modelnet40": 32,
    "vae_mnist": 512,
    "snn_nmnist": 128,
    "unet_isic": 16,
    "resnet18_cifar10": 256,
    "mobilenetv2_cifar10": 256,
    "saint_adult": 512,
    "capsnet_mnist": 128,
    "convlstm_movingmnist": 16,
}


def dataset_exists(model_key: str) -> bool:
    """Return True if the primary data files for *model_key* appear to be cached on disk.

    Uses a per-model sentinel path — a file or directory whose presence indicates
    that the download and extraction steps have already completed.  A False result
    is always safe: ``build_task_bundle`` will then run and fill any gaps.
    """
    root = _data_root()
    sentinels: dict[str, list[Path]] = {
        "lenet5":               [root / "mnist"],
        "vae_mnist":            [root / "mnist"],
        "capsnet_mnist":        [root / "mnist"],
        "resnet18_cifar10":     [root / "cifar10"],
        "mobilenetv2_cifar10":  [root / "cifar10"],
        "m5":                   [root / "speechcommands"],
        "lstm_forecaster":      [root / "etth1" / "ETTh1.csv"],
        "tcn_forecaster":       [root / "ettm1" / "ETTm1.csv"],
        "textcnn":              [root / "huggingface" / "ag_news"],
        "distilbert":           [root / "huggingface" / "glue"],
        "gru_forecaster":       [root / "huggingface" / "dunzane___time-series-dataset"],
        "gcn":                  [root / "cora" / "cora" / "cora.content"],
        "tabnet":               [root / "adult" / "adult.data"],
        "saint_adult":          [root / "adult" / "adult.data"],
        "mpnn":                 [root / "esol" / "delaney-processed.csv"],
        "attentivefp_freesolv": [root / "freesolv" / "SAMPL.csv"],
        "gin_imdbb":            [root / "imdb_binary" / ".extracted"],
        "actor_critic":         [root / "cartpole" / "heuristic_rollouts.pt"],
        "dqn_lunarlander":      [root / "lunarlander" / "heuristic_rollouts.pt"],
        "ppo_bipedalwalker":    [root / "bipedalwalker" / "heuristic_rollouts.pt"],
        "lstm_autoencoder":     [root / "mit-bih" / "100.dat"],
        "pointnet_modelnet40":  [root / "modelnet40" / "raw"],
        "snn_nmnist":           [root / "nmnist"],
        "unet_isic":            [root / "isic2018" / "images" / ".extracted"],
        "convlstm_movingmnist": [root / "moving_mnist" / "mnist_test_seq.npy"],
    }
    paths = sentinels.get(model_key)
    if paths is None:
        return False
    return all(p.exists() for p in paths)


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
        "dqn_lunarlander": _build_lunarlander,
        "ppo_bipedalwalker": _build_bipedalwalker,
        "attentivefp_freesolv": _build_freesolv,
        "gin_imdbb": _build_imdbb,
        "tcn_forecaster": _build_ettm1,
        "gru_forecaster": _build_weather,
        "pointnet_modelnet40": _build_modelnet40,
        "vae_mnist": _build_mnist,
        "snn_nmnist": _build_nmnist,
        "unet_isic": _build_isic,
        "resnet18_cifar10": _build_cifar10,
        "mobilenetv2_cifar10": _build_cifar10,
        "saint_adult": _build_adult,
        "capsnet_mnist": _build_mnist,
        "convlstm_movingmnist": _build_moving_mnist,
    }
    if model_key not in builders:
        raise KeyError(f"Unknown model key: {model_key}")
    effective_batch_size = (
        batch_size if batch_size is not None else _BATCH_SIZES.get(model_key, 64)
    )
    return builders[model_key](effective_batch_size)
