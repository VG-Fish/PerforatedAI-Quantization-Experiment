import csv
import math
import os
import tarfile
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Callable

import torch

from .models import MOLECULE_NODE_FEATURES, SOCIAL_GRAPH_NODE_FEATURES

DATA_ROOT_ENV: str = "DQB_DATA_ROOT"
DEFAULT_DATA_ROOT: str = "data"
EXTRACTED_MARKER: str = ".extracted"
HEURISTIC_ROLLOUTS_FILENAME: str = "heuristic_rollouts.pt"
ADULT_DATA_FILENAME: str = "adult.data"
ETTH1_URL: str = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)
ETTM1_URL: str = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"
)
ADULT_URLS: dict[str, str] = {
    ADULT_DATA_FILENAME: "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
}
CORA_URL: str = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
ESOL_URL: str = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
)
FREESOLV_URL: str = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv"
IMDBB_URL: str = "https://www.chrsmrrs.com/graphkerneldatasets/IMDB-BINARY.zip"
ISIC_SAMPLE_URL: str = (
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip"
)
ISIC_MASK_SAMPLE_URL: str = (
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip"
)
MITBIH_RECORDS: list[str] = ["100", "101", "103", "105", "106", "108", "109", "111"]
# Elements that get their own one-hot slot in the molecular featuriser; anything
# else falls into a shared "other" slot. Order is load-bearing: it fixes the
# feature layout that MOLECULE_NODE_FEATURES counts.
_ATOM_TYPES: tuple[str, ...] = (
    "C", "N", "O", "S", "F", "CL", "BR", "I", "P", "B", "SI", "SE",
)
_TWO_LETTER_ELEMENTS: frozenset[str] = frozenset({"Cl", "Br", "Si", "Se"})
_BOND_ORDERS: dict[str, float] = {"-": 1.0, "=": 2.0, "#": 3.0, ":": 1.5, "$": 4.0}
# Node capacity per molecular dataset, sized from the data: ESOL's heavy-atom
# count reaches 55 with a 99th percentile of 31, FreeSolv's maximum is 24. The
# dense [N,N] adjacency makes this quadratic in cost, so the cap tracks each
# dataset rather than taking the larger of the two.
ESOL_MAX_ATOMS: int = 40
FREESOLV_MAX_ATOMS: int = 24
# Degree buckets for the IMDB-BINARY featuriser: channel 0 is the real-node
# indicator and channels 1..8 are a one-hot over log2 degree, matching the
# degree-as-feature convention Xu et al. use for label-free social graphs.
_SOCIAL_DEGREE_BUCKETS: int = SOCIAL_GRAPH_NODE_FEATURES - 2
# Kept in step with TextCNN's `vocab_size` default in models.py.
AG_NEWS_VOCAB_SIZE: int = 20_000
AG_NEWS_SEQ_LEN: int = 128
SPEECH_COMMAND_LABELS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class _ParsedAtom:
    element: str
    aromatic: bool
    charge: int
    hydrogens: int


@dataclass(frozen=True)
class _Bond:
    source: int
    target: int
    order: float
    ring_closure: bool


@dataclass
class TaskBundle:
    train_loader: Any
    val_loader: Any
    test_loader: Any
    metric_name: str
    metric_direction: str
    input_description: str
    # Set when a regression task standardises its targets for training. The
    # loaders then yield z-scores, so metrics have to be mapped back through
    # ``value * target_scale + target_offset`` to stay in the dataset's own
    # units (log-solubility, kcal/mol, ...) and comparable to published numbers.
    # Identity by default; every classification task leaves it alone.
    target_offset: float = 0.0
    target_scale: float = 1.0

# Making the following into static methods is not necessary since every domain class 
# after calls them. Adding inheritance by making some shared base class is pointless.
def _data_root() -> Path:
    return Path(os.environ.get(DATA_ROOT_ENV, DEFAULT_DATA_ROOT)).expanduser()


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    print(f"Downloading {url} -> {destination}")
    tmp_path: Path = destination.with_suffix(destination.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        with urllib.request.urlopen(url) as response:
            expected_size: int | None = None
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                expected_size = int(content_length)
            with open(tmp_path, "wb") as out:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
        actual_size: int = tmp_path.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            raise IOError(
                f"Download size mismatch for {url}: got {actual_size} bytes, "
                f"expected {expected_size} bytes (likely interrupted; check disk space)."
            )
        tmp_path.rename(destination)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return destination


def _extract_zip(archive: Path, destination: Path) -> None:
    marker: Path = destination / EXTRACTED_MARKER
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
        package: str = package_name or import_name
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
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


def _split_dataset(
    dataset: Any, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> tuple[Any, Any, Any]:
    total: int = len(dataset)
    if total < 3:
        raise ValueError(
            "Need at least three samples to build train/validation/test splits."
        )
    train_size: int = max(1, int(total * train_ratio))
    val_size: int = max(1, int(total * val_ratio))
    test_size: int = max(1, total - train_size - val_size)
    overflow: int = train_size + val_size + test_size - total
    if overflow > 0:
        train_size: int = max(1, train_size - overflow)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    return train_ds, val_ds, test_ds


def _split_anomaly_dataset(
    dataset: Any,
    labels: Any,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[Any, Any, Any]:
    """Split a dataset so only normal rows reach the training split.

    A reconstruction autoencoder flags anomalies by failing to reconstruct
    them, which only works if it never learned to reconstruct them.  Under a
    plain random split the anomalies land in training too, the model learns
    them alongside the normal rows, and reconstruction error stops separating
    the classes — the detector gets worse the longer it trains.

    Normal rows are divided across all three splits.  Anomalous rows are held
    out of training entirely and divided evenly between validation and test, so
    both keep a mixed population to score AUC against.
    """
    label_values = labels.long().flatten()
    normal: list[int] = (label_values == 0).nonzero(as_tuple=True)[0].tolist()
    anomalous: list[int] = (label_values != 0).nonzero(as_tuple=True)[0].tolist()
    if len(normal) < 3 or len(anomalous) < 2:
        raise ValueError(
            "Need at least three normal and two anomalous samples to build "
            f"anomaly splits; got {len(normal)} normal and {len(anomalous)} "
            "anomalous."
        )

    generator = torch.Generator().manual_seed(42)

    def shuffled(indices: list[int]) -> list[int]:
        order = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[position] for position in order]

    normal = shuffled(normal)
    anomalous = shuffled(anomalous)

    # Clamp so validation and test always keep at least one normal row each.
    normal_train: int = min(max(int(len(normal) * train_ratio), 1), len(normal) - 2)
    normal_val: int = min(
        max(int(len(normal) * val_ratio), 1), len(normal) - normal_train - 1
    )
    anomalous_val: int = min(max(len(anomalous) // 2, 1), len(anomalous) - 1)

    subset = torch.utils.data.Subset
    return (
        subset(dataset, normal[:normal_train]),
        subset(
            dataset,
            normal[normal_train : normal_train + normal_val]
            + anomalous[:anomalous_val],
        ),
        subset(
            dataset,
            normal[normal_train + normal_val :] + anomalous[anomalous_val:],
        ),
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


def _standardized_regression_bundle(
    inputs: tuple[Any, ...],
    targets: Any,
    batch_size: int,
    metric_name: str,
    metric_direction: str,
    input_description: str,
    *,
    num_workers: int = 2,
) -> TaskBundle:
    """Split, then z-score the regression target using training rows only.

    FreeSolv's hydration energies span roughly -25..+4 kcal/mol and ESOL's log
    solubilities -11.6..+1.6. Trained on those raw values the models spent their
    opening epochs just learning the offset — AttentiveFP's train MSE sat at
    ~14.8 (FreeSolv's target variance) for its first ten epochs and the run
    finished at RMSE 2.14 against MoleculeNet's ~1.15.

    Statistics come from the training split alone so the validation and test
    rows contribute nothing to the transform. The offset and scale ride along on
    the bundle, and ``_compute_all_metrics`` maps predictions back through them,
    so reported RMSE/MAE stay in the dataset's own units.
    """
    dataset = _TensorRowsDataset(*inputs, targets)
    train_ds, val_ds, test_ds = _split_dataset(dataset)
    train_targets = targets[torch.tensor(list(train_ds.indices), dtype=torch.long)]
    offset = float(train_targets.mean().item())
    scale = float(train_targets.std().clamp_min(1e-6).item())
    standardized = _TensorRowsDataset(*inputs, (targets - offset) / scale)
    subset = torch.utils.data.Subset
    bundle = _bundle_from_splits(
        subset(standardized, list(train_ds.indices)),
        subset(standardized, list(val_ds.indices)),
        subset(standardized, list(test_ds.indices)),
        batch_size,
        metric_name,
        metric_direction,
        input_description,
        num_workers=num_workers,
    )
    bundle.target_offset = offset
    bundle.target_scale = scale
    return bundle


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

def _hf_dataset_cache() -> str:
    cache: Path = _data_root() / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    return str(cache)


# Each following class groups build functions for a single data type/domain as a static
# method. As mentioned above, helpers reused across domains stay outside.

class VisionDatasets:
    @staticmethod
    def mnist(batch_size: int, *, augment: bool = False) -> TaskBundle:
        """MNIST, optionally with the 2-pixel-shift augmentation.

        ``augment`` is on for the two MNIST *classifiers* (LeNet-5, CapsNet) and
        off for the VAE, whose ELBO is measured against the canonical images.
        Sabour et al. train CapsNet on "MNIST shifted by up to 2 pixels in each
        direction with zero padding" and nothing else, which is what the small
        random translation below reproduces; LeNet-5 benefits from the same.
        """
        torchvision = _require_dependency("torchvision")
        transforms = __import__("torchvision.transforms", fromlist=["transforms"])
        root: Path = _data_root() / "mnist"
        root.mkdir(parents=True, exist_ok=True)
        eval_transform = transforms.Compose([transforms.ToTensor()])
        train_transform = (
            transforms.Compose(
                [
                    transforms.RandomAffine(
                        degrees=0, translate=(2 / 28, 2 / 28), fill=0
                    ),
                    transforms.ToTensor(),
                ]
            )
            if augment
            else eval_transform
        )
        train_full = torchvision.datasets.MNIST(
            root=str(root), train=True, download=True, transform=train_transform
        )
        test_ds = torchvision.datasets.MNIST(
            root=str(root), train=False, download=True, transform=eval_transform
        )
        train_ds, val_ds = torch.utils.data.random_split(
            train_full,
            [55_000, 5_000],
            generator=torch.Generator().manual_seed(42),
        )
        if augment:
            # val_ds is a Subset of train_full and would otherwise inherit the
            # random shift, adding augmentation noise to the score PerforatedAI
            # reads when deciding whether to add dendrites. Same indices, clean
            # transform — mirrors what cifar10() does below.
            eval_view = torchvision.datasets.MNIST(
                root=str(root), train=True, download=True, transform=eval_transform
            )
            val_ds = torch.utils.data.Subset(eval_view, val_ds.indices)
        return _bundle_from_splits(
            train_ds,
            val_ds,
            test_ds,
            batch_size,
            "Accuracy",
            "maximize",
            "MNIST handwritten digit images",
        )
    @staticmethod
    def mnist_augmented(batch_size: int) -> TaskBundle:
        return VisionDatasets.mnist(batch_size, augment=True)

    @staticmethod
    def cifar10(batch_size: int) -> TaskBundle:
        torchvision = _require_dependency("torchvision")
        transforms = __import__("torchvision.transforms", fromlist=["transforms"])
        root: Path = _data_root() / "cifar10"
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
        # val_ds is a Subset of train_full, so without this it inherits
        # train_full's RandomCrop/RandomHorizontalFlip transform — every
        # validation read (and therefore every PAI dendrite-switch decision,
        # which watches validation score) sees a randomly cropped/flipped
        # image instead of the canonical one, adding pure augmentation noise
        # to the signal PAI uses to decide when to add dendrites. Same
        # indices, clean (no-augmentation) transform.
        eval_view = torchvision.datasets.CIFAR10(
            root=str(root), train=True, download=True, transform=test_transform
        )
        val_ds = torch.utils.data.Subset(eval_view, val_ds.indices)
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
        root: Path = _data_root() / "speechcommands"
        root.mkdir(parents=True, exist_ok=True)
        base = torchaudio.datasets.SPEECHCOMMANDS(
            str(root), download=True, subset=subset
        )
        labels: dict[str, int] = {label: index for index, label in enumerate(SPEECH_COMMAND_LABELS)}
        self.base = base
        self.labels: dict[str, int] = labels
        self.indices: list[int] = [index for index, item in enumerate(base) if item[2] in labels]
        self.target_len = 16_000
        # Do NOT store self.torch — the torch module object is not picklable, which
        # would prevent DataLoader from serialising this dataset for worker processes.

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        _torch = (
            torch
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

class AudioDatasets:
    @staticmethod
    def speechcommands(batch_size: int) -> TaskBundle:
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
        self.tensors: tuple[Any, ...] = tensors

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        return tuple(tensor[index] for tensor in self.tensors)

class TimeSeriesDatasets:
    @staticmethod
    def etth1(batch_size: int) -> TaskBundle:
        path: Path = _download(ETTH1_URL, _data_root() / "etth1" / "ETTh1.csv")
        rows: list[float] = []
        with path.open(newline="") as fh:
            reader: csv.DictReader[str] = csv.DictReader(fh)
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

    @staticmethod
    def multivariate_forecast(
        batch_size: int,
        *,
        url: str,
        subdir: str,
        filename: str,
        seq_len: int,
        horizon: int,
        input_description: str,
    ) -> TaskBundle:
        path: Path = _download(url, _data_root() / subdir / filename)
        rows: list[list[float]] = []
        with path.open(newline="") as fh:
            reader: csv.DictReader[str] = csv.DictReader(fh)
            for row in reader:
                values: list[float] = []
                for key, value in row.items():
                    if key.lower() == "date":
                        continue
                    try:
                        values.append(float(value))
                    except ValueError:
                        # skip non-numeric columns
                        pass
                if values:
                    rows.append(values)
        values_t = torch.tensor(rows, dtype=torch.float32)
        mean = values_t.mean(dim=0, keepdim=True)
        std = values_t.std(dim=0, keepdim=True).clamp_min(1e-6)
        values_t = (values_t - mean) / std
        xs = []
        ys = []
        limit: int = len(values_t) - seq_len - horizon + 1
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
    @staticmethod
    def ettm1(batch_size: int) -> TaskBundle:
        return TimeSeriesDatasets.multivariate_forecast(
            batch_size,
            url=ETTM1_URL,
            subdir="ettm1",
            filename="ETTm1.csv",
            seq_len=96,
            horizon=24,
            input_description="ETTm1 15-minute multivariate transformer-temperature windows",
        )
    @staticmethod
    def weather(batch_size: int) ->TaskBundle:
        datasets = _require_dependency("datasets")
        loaded = datasets.load_dataset(
            "dunzane/time-series-dataset", "Weather", cache_dir=_hf_dataset_cache()
        )
        split = loaded["train"]
        columns: list[Any] = [
            name
            for name in split.column_names
            if name.lower() != "date" and split.features[name].dtype in {"float32", "float64", "int32", "int64"}
        ]
        rows: list[list[float]] = [[float(row[name]) for name in columns] for row in split]
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

class TextDataSets:
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return "".join(char.lower() if char.isalnum() else " " for char in text).split()

    @staticmethod
    def _build_vocab(texts: Iterable[str], vocab_size: int) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(TextDataSets._tokenize(text))
        return {
            token: index + 1
            for index, (token, _) in enumerate(counts.most_common(vocab_size - 1))
        }

    @staticmethod
    def _encode_texts(texts: Iterable[str], vocab: dict[str, int], seq_len: int) -> Any:
        encoded: list[list[int]] = []
        for text in texts:
            ids: list[int] = [vocab.get(token, 0) for token in TextDataSets._tokenize(text)[:seq_len]]
            ids.extend([0] * (seq_len - len(ids)))
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    @staticmethod
    def ag_news(batch_size: int) -> TaskBundle:
        datasets = _require_dependency("datasets")
        loaded = datasets.load_dataset("ag_news", cache_dir=_hf_dataset_cache())
        # AG News rows are a title plus a full description sentence, averaging
        # ~38 tokens and running past 100. Truncating at 64 tokens with a 5k
        # vocabulary threw away the tail of the longer rows and mapped a large
        # share of content words to the OOV id; Kim's TextCNN results rely on
        # covering the vocabulary. AG_NEWS_VOCAB_SIZE must stay in step with
        # TextCNN's `vocab_size` default in models.py.
        vocab: dict[str, int] = TextDataSets._build_vocab(
            loaded["train"]["text"], AG_NEWS_VOCAB_SIZE
        )
        train_texts = loaded["train"]["text"]
        train_labels = torch.tensor(loaded["train"]["label"], dtype=torch.long)
        x_train = TextDataSets._encode_texts(train_texts, vocab, AG_NEWS_SEQ_LEN)
        train_full = _TensorRowsDataset(x_train, train_labels)
        train_ds, val_ds, _ = _split_dataset(train_full, train_ratio=0.9, val_ratio=0.1)
        x_test = TextDataSets._encode_texts(loaded["test"]["text"], vocab, AG_NEWS_SEQ_LEN)
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

    @staticmethod
    def sst2(batch_size: int) -> TaskBundle:
        datasets = _require_dependency("datasets")
        transformers = _require_dependency("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
        loaded = datasets.load_dataset("glue", "sst2", cache_dir=_hf_dataset_cache())

        def _tokenize(sentences: list[str]) -> tuple[Any, Any]:
            encoding = tokenizer(
                list(sentences),
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            return encoding["input_ids"], encoding["attention_mask"]

        train_ids, train_mask = _tokenize(loaded["train"]["sentence"])
        y_train = torch.tensor(list(loaded["train"]["label"]), dtype=torch.long)
        train_full = _TensorRowsDataset(train_ids, train_mask, y_train)
        train_ds, val_ds, _ = _split_dataset(train_full, train_ratio=0.9, val_ratio=0.1)
        test_ids, test_mask = _tokenize(loaded["validation"]["sentence"])
        y_test = torch.tensor(list(loaded["validation"]["label"]), dtype=torch.long)
        return _bundle_from_splits(
            train_ds,
            val_ds,
            _TensorRowsDataset(test_ids, test_mask, y_test),
            batch_size,
            "Accuracy",
            "maximize",
            "SST-2 sentences tokenized with distilbert-base-uncased tokenizer",
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
        neighbors = self.adjacency[index].nonzero().flatten()
        neighbors = torch.cat(
            [
                torch.tensor([index], dtype=neighbors.dtype, device=neighbors.device),
                neighbors[neighbors != index],
            ]
        )[:50]
        if len(neighbors) < 50:
            pad = neighbors.new_full((50 - len(neighbors),), index)
            neighbors = torch.cat([neighbors, pad])
        sub_x = self.x_all[neighbors]
        sub_adj = self.adjacency[neighbors][:, neighbors]
        return sub_x, sub_adj, self.y_all[index]

def _parse_adult_file(path: Path) -> list[list[str]]:
    rows = []
    with path.open(newline="") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw or raw.startswith("|"):
                continue
            rows.append([part.strip().rstrip(".") for part in raw.split(",")])
    return rows


def _encode_adult_row(
    row: list[str],
    encoders: list[dict[str, int]],
    numeric_columns: set[int],
    feature_count: int,
) -> list[float]:
    encoded: list[float] = []
    for col in range(feature_count):
        value: str = row[col]
        if col in numeric_columns:
            encoded.append(float(value) if value != "?" else 0.0)
        else:
            mapping: dict[str, int] = encoders[col]
            if value not in mapping:
                mapping[value] = len(mapping) + 1
            encoded.append(float(mapping[value]))
    return encoded

def _encode_adult_rows(
    rows: list[list[str]],
    encoders: list[dict[str, int]],
    numeric_columns: set[int],
    feature_count: int,
) -> tuple[list[list[float]], list[int]]:
    values: list[list[float]] = []
    labels: list[int] = []
    for row in rows:
        values.append(_encode_adult_row(row, encoders, numeric_columns, feature_count))
        labels.append(1 if row[-1] == ">50K" else 0)
    return values, labels

def _build_adult(batch_size: int) -> TaskBundle:
    root: Path = _data_root() / "adult"
    for filename, url in ADULT_URLS.items():
        _download(url, root / filename)
    train_rows: list[list[str]] = _parse_adult_file(root / ADULT_DATA_FILENAME)
    test_rows: list[list[str]] = _parse_adult_file(root / "adult.test")
    feature_count = 14
    encoders: list[dict[str, int]] = [{} for _ in range(feature_count)]
    numeric_columns: set[int] = {0, 2, 4, 10, 11, 12}

    train_x_raw, train_y_raw = _encode_adult_rows(train_rows, encoders, numeric_columns, feature_count)
    test_x_raw, test_y_raw = _encode_adult_rows(test_rows, encoders, numeric_columns, feature_count)
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



class GraphDatasets:
    @staticmethod
    def cora(batch_size: int) -> TaskBundle:
        root: Path = _data_root() / "cora"
        archive: Path = _download(CORA_URL, root / "cora.tgz")
        content: Path = root / "cora" / "cora.content"
        cites: Path = root / "cora" / "cora.cites"
        if not content.exists() or not cites.exists():
            with tarfile.open(archive) as tar:
                tar.extractall(root)

        paper_ids: list[str] = []
        features: list[list[float]] = []
        labels_raw: list[str] = []
        with content.open() as fh:
            for line in fh:
                parts: list[str] = line.strip().split()
                paper_ids.append(parts[0])
                features.append([float(value) for value in parts[1:-1]])
                labels_raw.append(parts[-1])
        id_to_idx: dict[str, int] = {paper_id: index for index, paper_id in enumerate(paper_ids)}
        label_to_idx: dict[str, int] = {label: index for index, label in enumerate(sorted(set(labels_raw)))}
        x_all = torch.tensor(features, dtype=torch.float32)
        # Row-normalise the bag-of-words, as in Kipf & Welling's reference
        # implementation. Without it a node's feature magnitude scales with how
        # many vocabulary words its abstract happens to contain, so the GCN's
        # first layer sees inputs whose norm varies several-fold across nodes
        # for reasons unrelated to the class.
        x_all = x_all / x_all.sum(dim=1, keepdim=True).clamp_min(1.0)
        y_all = torch.tensor([label_to_idx[label] for label in labels_raw], dtype=torch.long)
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

    @staticmethod
    def _parse_bracket_atom(body: str) -> _ParsedAtom:
        """Parse the inside of a SMILES bracket atom, e.g. ``nH``, ``O-``, ``N+``."""
        index = 0
        while index < len(body) and body[index].isdigit():  # strip the isotope
            index += 1
        body = body[index:]
        if not body:
            return _ParsedAtom("C", False, 0, 0)
        element: str = body[0]
        rest: str = body[1:]
        if rest and rest[0].islower() and (element + rest[0]).upper() in _ATOM_TYPES:
            element += rest[0]
            rest = rest[1:]
        aromatic: bool = element[0].islower()
        charge = 0
        hydrogens = 0
        index = 0
        while index < len(rest):
            char = rest[index]
            index += 1
            if char == "H":
                digits = ""
                while index < len(rest) and rest[index].isdigit():
                    digits += rest[index]
                    index += 1
                hydrogens = int(digits) if digits else 1
                continue
            if char in "+-":
                sign = 1 if char == "+" else -1
                digits = ""
                while index < len(rest) and rest[index].isdigit():
                    digits += rest[index]
                    index += 1
                if digits:
                    charge += sign * int(digits)
                else:
                    repeats = 1
                    while index < len(rest) and rest[index] == char:
                        repeats += 1
                        index += 1
                    charge += sign * repeats
        return _ParsedAtom(element.upper(), aromatic, charge, hydrogens)

    @staticmethod
    def _read_atom(smiles: str, index: int) -> tuple["_ParsedAtom | None", int]:
        """Return (atom, next_index); atom is None for a non-atom character."""
        char: str = smiles[index]
        if char == "[":
            end: int = smiles.find("]", index)
            if end == -1:
                return None, index + 1
            return GraphDatasets._parse_bracket_atom(smiles[index + 1 : end]), end + 1
        pair: str = smiles[index : index + 2]
        if pair in _TWO_LETTER_ELEMENTS:
            return _ParsedAtom(pair.upper(), False, 0, 0), index + 2
        if char.isalpha():
            return _ParsedAtom(char.upper(), char.islower(), 0, 0), index + 1
        return None, index + 1

    @staticmethod
    def _parse_smiles(smiles: str) -> tuple[list["_ParsedAtom"], list[_Bond]]:
        """Parse SMILES into atoms and bonds, honouring branches and ring closures.

        The previous parser skipped every non-alphabetic character, so ``(`` and
        ``)`` silently vanished and a branched molecule came out as one long
        chain — wrong topology for 76% of ESOL and 70% of FreeSolv. It also
        dropped bond orders and bracket atoms entirely.
        """
        atoms: list[_ParsedAtom] = []
        bonds: list[_Bond] = []
        branch_stack: list[int | None] = []
        ring_open: dict[str, tuple[int, float]] = {}
        previous: int | None = None
        pending_order: float | None = None
        index = 0
        while index < len(smiles):
            char: str = smiles[index]
            if char == "(":
                branch_stack.append(previous)
                index += 1
                continue
            if char == ")":
                previous = branch_stack.pop() if branch_stack else None
                index += 1
                continue
            if char in _BOND_ORDERS:
                pending_order = _BOND_ORDERS[char]
                index += 1
                continue
            if char in "/\\":  # directional bonds are single bonds
                index += 1
                continue
            if char == ".":  # disconnected component
                previous = None
                pending_order = None
                index += 1
                continue
            if previous is not None and (char == "%" or char.isdigit()):
                if char == "%":
                    label: str = smiles[index + 1 : index + 3]
                    index += 3
                else:
                    label = char
                    index += 1
                order: float = pending_order or 1.0
                pending_order = None
                if label in ring_open:
                    partner, partner_order = ring_open.pop(label)
                    bonds.append(
                        _Bond(partner, previous, max(order, partner_order), True)
                    )
                else:
                    ring_open[label] = (previous, order)
                continue
            atom, index = GraphDatasets._read_atom(smiles, index)
            if atom is None:
                continue
            current: int = len(atoms)
            atoms.append(atom)
            if previous is not None:
                bonds.append(_Bond(previous, current, pending_order or 1.0, False))
            pending_order = None
            previous = current
        if not atoms:
            atoms = [_ParsedAtom("C", False, 0, 0)]
        return atoms, bonds

    @staticmethod
    def _build_graph_tensors(
        atoms: list["_ParsedAtom"], bonds: list[_Bond], max_nodes: int
    ) -> tuple[Any, Any]:
        """Featurise a parsed molecule into (node_features, dense adjacency).

        Padded rows are left all-zero, which is what the molecular models use to
        tell real atoms from padding (see ``MPNN.forward``/``AttentiveFP.forward``);
        the adjacency's self-loops cannot serve that purpose.
        """
        width: int = MOLECULE_NODE_FEATURES
        x = torch.zeros((max_nodes, width), dtype=torch.float32)
        adjacency = torch.eye(max_nodes, dtype=torch.float32)
        kept: int = min(len(atoms), max_nodes)
        degree = [0.0] * kept
        double_bond = [0.0] * kept
        triple_bond = [0.0] * kept
        in_ring = [0.0] * kept
        for bond in bonds:
            if bond.source >= kept or bond.target >= kept or bond.source == bond.target:
                continue
            adjacency[bond.source, bond.target] = 1.0
            adjacency[bond.target, bond.source] = 1.0
            for end in (bond.source, bond.target):
                degree[end] += 1.0
                if bond.order >= 3.0:
                    triple_bond[end] = 1.0
                elif bond.order >= 2.0:
                    double_bond[end] = 1.0
                if bond.ring_closure:
                    in_ring[end] = 1.0
        offset: int = len(_ATOM_TYPES)
        for position in range(kept):
            atom = atoms[position]
            if atom.element in _ATOM_TYPES:
                x[position, _ATOM_TYPES.index(atom.element)] = 1.0
            else:
                x[position, offset] = 1.0
            x[position, offset + 1] = 1.0 if atom.aromatic else 0.0
            x[position, offset + 2] = atom.charge / 2.0
            x[position, offset + 3] = degree[position] / 4.0
            x[position, offset + 4] = atom.hydrogens / 4.0
            x[position, offset + 5] = double_bond[position]
            x[position, offset + 6] = triple_bond[position]
            x[position, offset + 7] = in_ring[position]
        return x, adjacency

    @staticmethod
    def _smiles_to_graph(smiles: str, max_nodes: int) -> tuple[Any, Any]:
        atoms, bonds = GraphDatasets._parse_smiles(smiles)
        return GraphDatasets._build_graph_tensors(atoms, bonds, max_nodes)

    @staticmethod
    def _molecular_regression_bundle(
        path: Path,
        *,
        target_keys: tuple[str, ...],
        max_nodes: int,
        batch_size: int,
        input_description: str,
    ) -> TaskBundle:
        node_features: list[Any] = []
        adjacencies: list[Any] = []
        labels: list[float] = []
        with path.open(newline="") as fh:
            reader: csv.DictReader[str] = csv.DictReader(fh)
            for row in reader:
                smiles: str | None = (
                    row.get("smiles") or row.get("smile") or row.get("SMILES")
                )
                target: str | None = next(
                    (row[key] for key in target_keys if row.get(key)), None
                )
                if smiles is None or target is None:
                    continue
                x, adjacency = GraphDatasets._smiles_to_graph(smiles, max_nodes)
                node_features.append(x)
                adjacencies.append(adjacency)
                labels.append(float(target))
        return _standardized_regression_bundle(
            (torch.stack(node_features), torch.stack(adjacencies)),
            torch.tensor(labels, dtype=torch.float32),
            batch_size,
            "RMSE",
            "minimize",
            input_description,
        )

    @staticmethod
    def esol(batch_size: int) -> TaskBundle:
        return GraphDatasets._molecular_regression_bundle(
            _download(ESOL_URL, _data_root() / "esol" / "delaney-processed.csv"),
            target_keys=(
                "measured log solubility in mols per litre",
                "ESOL predicted log solubility in mols per litre",
            ),
            max_nodes=ESOL_MAX_ATOMS,
            batch_size=batch_size,
            input_description="ESOL MoleculeNet molecular graphs from SMILES",
        )

    @staticmethod
    def freesolv(batch_size: int) -> TaskBundle:
        return GraphDatasets._molecular_regression_bundle(
            _download(FREESOLV_URL, _data_root() / "freesolv" / "SAMPL.csv"),
            target_keys=("expt", "measured log solubility in mols per litre", "y"),
            max_nodes=FREESOLV_MAX_ATOMS,
            batch_size=batch_size,
            input_description=(
                "FreeSolv molecular hydration free-energy graphs from SMILES"
            ),
        )

    @staticmethod
    def _read_tu_indicator(path: Path) -> list[int]:
        return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]

    @staticmethod
    def imdbb(batch_size: int) -> TaskBundle:
        root: Path = _data_root() / "imdb_binary"
        archive: Path = _download(IMDBB_URL, root / "IMDB-BINARY.zip")
        _extract_zip(archive, root)
        dataset_dir: Path = root / "IMDB-BINARY"
        graph_indicator: list[int] = GraphDatasets._read_tu_indicator(dataset_dir / "IMDB-BINARY_graph_indicator.txt")
        labels_raw: list[int] = GraphDatasets._read_tu_indicator(dataset_dir / "IMDB-BINARY_graph_labels.txt")
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
        edge_set: set[tuple[int, int]] = set(edges) | {(b, a) for a, b in edges}
        for graph_id, nodes in sorted(graph_nodes.items()):
            nodes = nodes[:max_nodes]
            node_map: dict[int, int] = {node: i for i, node in enumerate(nodes)}
            adjacency = torch.eye(max_nodes, dtype=torch.float32)
            degree = torch.zeros(max_nodes, dtype=torch.float32)
            for src, dst in edge_set:
                if src in node_map and dst in node_map:
                    i, j = node_map[src], node_map[dst]
                    adjacency[i, j] = 1.0
                    degree[i] += 1.0
            # IMDB-BINARY nodes carry no labels, so the degree *is* the feature.
            # Xu et al. one-hot it for GIN; the previous single normalised-degree
            # scalar collapsed every node into one nearly-constant channel, which
            # is most of why train loss moved only 0.705 -> 0.622 in 100 epochs.
            x = torch.zeros((max_nodes, SOCIAL_GRAPH_NODE_FEATURES), dtype=torch.float32)
            x[: len(nodes), 0] = 1.0  # real-node indicator; GIN pools on this
            buckets = torch.log2(degree.clamp_min(1.0)).floor().long()
            buckets = buckets.clamp(0, _SOCIAL_DEGREE_BUCKETS - 1)
            for node_index in range(len(nodes)):
                x[node_index, 1 + int(buckets[node_index].item())] = 1.0
            x[: len(nodes), -1] = degree[: len(nodes)] / float(max_nodes)
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
    gymnasium = _require_dependency("gymnasium", "gymnasium[classic-control]")
    numpy = _require_dependency("numpy")
    cache: Path = _data_root() / "cartpole" / HEURISTIC_ROLLOUTS_FILENAME
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
            action: int = 1 if (obs[2] + 0.25 * obs[3]) > 0 else 0
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


def _lunarlander_heuristic_action(
    x_pos: float,
    y_pos: float,
    x_vel: float,
    y_vel: float,
    angle: float,
    left_contact: bool,
    right_contact: bool,
) -> int:
    if left_contact or right_contact:
        if abs(x_vel) < 0.2:
            return 0
        return 3 if x_vel < 0 else 1
    if abs(angle) > 0.12:
        return 1 if angle > 0 else 3
    if y_vel < -0.25 or y_pos < 0.6:
        return 2
    if abs(x_pos) > 0.15:
        return 3 if x_pos < 0 else 1
    return 0


def _build_lunarlander(batch_size: int) -> TaskBundle:
    gymnasium = _require_dependency("gymnasium", "gymnasium[box2d]")
    numpy = _require_dependency("numpy")
    cache: Path = _data_root() / "lunarlander" / HEURISTIC_ROLLOUTS_FILENAME
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
            x_pos, y_pos, x_vel, y_vel, angle, _, left_contact, right_contact = obs
            action = _lunarlander_heuristic_action(
                x_pos, y_pos, x_vel, y_vel, angle, left_contact, right_contact
            )
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
    gymnasium = _require_dependency("gymnasium", "gymnasium[box2d]")
    numpy = _require_dependency("numpy")
    cache: Path = _data_root() / "bipedalwalker" / HEURISTIC_ROLLOUTS_FILENAME
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


class _ModelNet40Dataset:
    def __init__(self, train: bool) -> None:
        self.root: Path = _data_root() / "modelnet40"
        archive: Path = _download("http://modelnet.cs.princeton.edu/ModelNet40.zip", self.root / "ModelNet40.zip")
        raw_root: Path = self.root / "raw"
        if not raw_root.exists() or not any(raw_root.iterdir()):
            _extract_zip(archive, self.root / "_extracted")
            extracted: Path = self.root / "_extracted" / "ModelNet40"
            if extracted.exists():
                extracted.rename(raw_root)
        split: str = "train" if train else "test"
        categories: list[str] = sorted(path.name for path in raw_root.iterdir() if path.is_dir())
        self.class_to_idx: dict[str, int] = {category: index for index, category in enumerate(categories)}
        self.samples: list[tuple[Path, int]] = []
        for category in categories:
            for path in sorted((raw_root / category / split).glob("*.off")):
                self.samples.append((path, self.class_to_idx[category]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        path, label = self.samples[index]
        with path.open() as fh:
            header: str = fh.readline().strip()
            if header != "OFF":
                counts: list[str] = header[3:].strip().split()
            else:
                counts: list[str] = fh.readline().strip().split()
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


class _ModelNet40TrainAugment:
    """Random rotation + jitter, applied only to the training split.

    ``_ModelNet40Dataset.__getitem__`` downsamples each mesh to the same 1024
    vertex indices (``torch.linspace``) every time it's called, so without
    this wrapper the model sees the exact same points in the exact same pose
    on every one of its ~60 epochs — effectively zero example diversity,
    which is a big part of why train accuracy reached ~92% while val sat
    near random-guess (~8%) with val_loss ballooning to double digits.
    This mirrors the augmentation from the reference PointNet implementation
    (Qi et al., https://github.com/charlesq34/pointnet, provider.py
    ``rotate_point_cloud`` + ``jitter_point_cloud``): a random rotation about
    the up axis plus small per-point Gaussian jitter, clipped to bound
    outliers. Only wraps the train subset — val/test stay deterministic.
    """

    def __init__(self, base: Any, jitter_std: float = 0.01, jitter_clip: float = 0.05):
        self.base = base
        self.jitter_std = jitter_std
        self.jitter_clip = jitter_clip

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        points, label = self.base[index]
        angle = float(torch.rand(1).item()) * 2.0 * math.pi
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotation = torch.tensor(
            [[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]],
            dtype=points.dtype,
        )
        points = points @ rotation.T
        jitter = (torch.randn_like(points) * self.jitter_std).clamp(
            -self.jitter_clip, self.jitter_clip
        )
        return points + jitter, label


def _build_modelnet40(batch_size: int) -> TaskBundle:
    train_full = _ModelNet40Dataset(train=True)
    train_ds, val_ds, _ = _split_dataset(train_full, train_ratio=0.9, val_ratio=0.1)
    train_ds = _ModelNet40TrainAugment(train_ds)
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
    root: Path = _data_root() / "nmnist"
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
        self.root: Path = root
        self.image_size: int = image_size
        self.samples: list[tuple[Path, Path]] = self._discover_pairs()

    def _discover_pairs(self) -> list[tuple[Path, Path]]:
        image_files: list[Path] = [
            path
            for path in self.root.rglob("*.jpg")
            if "superpixel" not in path.name.lower()
        ]
        mask_files: list[Path] = list(self.root.rglob("*segmentation*.png")) + list(
            self.root.rglob("*Segmentation*.png")
        )
        masks_by_stem: dict[str, Path] = {
            mask.name.replace("_segmentation", "").replace("_Segmentation", "").split(".")[0]: mask
            for mask in mask_files
        }
        pairs = []
        for image in image_files:
            key: str = image.stem
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
        pil_image = __import__("PIL.Image", fromlist=["Image"])
        image_path, mask_path = self.samples[index]
        image = pil_image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        mask = pil_image.open(mask_path).convert("L").resize((self.image_size, self.image_size))
        image_t = torch.tensor(list(image.getdata()), dtype=torch.float32).view(
            self.image_size, self.image_size, 3
        ).permute(2, 0, 1) / 255.0
        mask_t = torch.tensor(list(mask.getdata()), dtype=torch.float32).view(
            1, self.image_size, self.image_size
        ) / 255.0
        return image_t, (mask_t > 0.5).float()

class MedicalDatasets:
    @staticmethod
    def isic(batch_size: int) -> TaskBundle:
        root: Path = _data_root() / "isic2018"
        image_archive: Path = _download(ISIC_SAMPLE_URL, root / "images.zip")
        mask_archive: Path = _download(ISIC_MASK_SAMPLE_URL, root / "masks.zip")
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

    @staticmethod
    def mitbih(batch_size: int) -> TaskBundle:
        wfdb = _require_dependency("wfdb")
        root: Path = _data_root() / "mit-bih"
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
        stacked = torch.stack(windows)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        # Anomalous beats are ~32% of this corpus, and two of the eight records
        # (109, 111) are entirely abnormal, so a random split would put their
        # morphology in training and the autoencoder would reconstruct them as
        # readily as a normal beat.  Train on normal beats only.
        return _bundle_from_splits(
            *_split_anomaly_dataset(
                _TensorRowsDataset(stacked, stacked, label_tensor), label_tensor
            ),
            batch_size,
            "AUC",
            "maximize",
            "MIT-BIH Arrhythmia ECG beat windows (autoencoder trained on normal beats)",
        )



# Per-model batch sizes tuned for Apple Silicon MPS throughput.
# Larger batches amortise Python-loop and host-to-device transfer overhead,
# keeping the GPU busy for longer between CPU round-trips.  Each value was
# chosen to be well within the M3 Pro's unified-memory budget while giving
# the GPU enough work per step to approach saturation.
# Fallback batch sizes for callers that build a bundle without one — chiefly
# `dqb download_data` and ad-hoc use. `BenchmarkRunner._run_condition` always
# passes the batch size from that model's ModelTrainingRecipe, so training never
# reads this table; the values are kept in step with the recipes anyway, since a
# table that silently disagreed with them is a trap for anyone reading either one.
_BATCH_SIZES: dict[str, int] = {
    "lenet5": 128,  # MNIST 28×28 greyscale — negligible per-sample cost
    "m5": 128,  # SpeechCommands 16 K-sample 1-D waveform
    "lstm_forecaster": 256,  # ETTh1 short sliding windows
    "textcnn": 128,  # AG News fixed-length token sequences
    "gcn": 32,  # Cora ego-graphs — adjacency matrix limits batch size
    "tabnet": 1024,  # Adult Income 14-feature tabular rows
    "mpnn": 32,  # ESOL molecular graphs — variable topology
    "actor_critic": 512,  # CartPole 4-D observations — negligible per-sample cost
    "lstm_autoencoder": 128,  # MIT-BIH 128-sample ECG windows
    "distilbert": 32,  # DistilBERT 128-token sequences — larger model requires smaller batches
    "dqn_lunarlander": 128,  # Matches tuned SB3 RL Zoo DQN batch size.
    "ppo_bipedalwalker": 64,  # Matches tuned SB3 RL Zoo PPO minibatch size.
    "attentivefp_freesolv": 32,
    "gin_imdbb": 32,
    "tcn_forecaster": 128,
    "gru_forecaster": 24,  # Forecasting horizon-sized batches stabilize GRU training.
    "pointnet_modelnet40": 32,  # PointNet reference batch size (Qi et al.).
    "vae_mnist": 128,  # PyTorch MNIST VAE example default.
    "snn_nmnist": 16,  # N-MNIST SNN literature setting.
    "unet_isic": 8,  # ISIC lesion segmentation studies favor small batches.
    "resnet18_cifar10": 128,  # CIFAR SGD recipe batch size.
    "mobilenetv2_cifar10": 128,  # CIFAR SGD recipe batch size.
    "saint_adult": 256,  # Official SAINT implementation default.
    "capsnet_mnist": 128,  # CapsNet MNIST recipe.
}


def dataset_exists(model_key: str) -> bool:
    """Return True if the primary data files for *model_key* appear to be cached on disk.

    Uses a per-model sentinel path — a file or directory whose presence indicates
    that the download and extraction steps have already completed.  A False result
    is a:lways safe: ``build_task_bundle`` will then run and fill any gaps.
    """
    root: Path = _data_root()
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
        "tabnet":               [root / "adult" / ADULT_DATA_FILENAME],
        "saint_adult":          [root / "adult" / ADULT_DATA_FILENAME],
        "mpnn":                 [root / "esol" / "delaney-processed.csv"],
        "attentivefp_freesolv": [root / "freesolv" / "SAMPL.csv"],
        "gin_imdbb":            [root / "imdb_binary" / EXTRACTED_MARKER],
        "actor_critic":         [root / "cartpole" / HEURISTIC_ROLLOUTS_FILENAME],
        "dqn_lunarlander":      [root / "lunarlander" / HEURISTIC_ROLLOUTS_FILENAME],
        "ppo_bipedalwalker":    [root / "bipedalwalker" / HEURISTIC_ROLLOUTS_FILENAME],
        "lstm_autoencoder":     [root / "mit-bih" / "100.dat"],
        "pointnet_modelnet40":  [root / "modelnet40" / "raw"],
        "snn_nmnist":           [root / "nmnist"],
        "unet_isic":            [root / "isic2018" / "images" / EXTRACTED_MARKER],
    }
    paths: list[Path] | None = sentinels.get(model_key)
    if paths is None:
        return False
    return all(p.exists() for p in paths)


def build_task_bundle(model_key: str, batch_size: int | None = None) -> TaskBundle:
    """Build the data loaders for *model_key*.

    Pass an explicit ``batch_size`` to override the MPS-tuned default in
    ``_BATCH_SIZES`` (useful for smoke tests or ablation studies).
    """
    builders: dict[str, Callable[..., TaskBundle]] = {
        "lenet5": VisionDatasets.mnist_augmented,
        "m5": AudioDatasets.speechcommands,
        "lstm_forecaster": TimeSeriesDatasets.etth1,
        "textcnn": TextDataSets.ag_news,
        "gcn": GraphDatasets.cora,
        "tabnet": _build_adult,
        "mpnn": GraphDatasets.esol,
        "actor_critic": _build_cartpole,
        "lstm_autoencoder": MedicalDatasets.mitbih,
        "distilbert": TextDataSets.sst2,
        "dqn_lunarlander": _build_lunarlander,
        "ppo_bipedalwalker": _build_bipedalwalker,
        "attentivefp_freesolv": GraphDatasets.freesolv,
        "gin_imdbb": GraphDatasets.imdbb,
        "tcn_forecaster": TimeSeriesDatasets.ettm1,
        "gru_forecaster": TimeSeriesDatasets.weather,
        "pointnet_modelnet40": _build_modelnet40,
        "vae_mnist": VisionDatasets.mnist,
        "snn_nmnist": _build_nmnist,
        "unet_isic": MedicalDatasets.isic,
        "resnet18_cifar10": VisionDatasets.cifar10,
        "mobilenetv2_cifar10": VisionDatasets.cifar10,
        "saint_adult": _build_adult,
        "capsnet_mnist": VisionDatasets.mnist_augmented,
    }
    if model_key not in builders:
        raise KeyError(f"Unknown model key: {model_key}")
    effective_batch_size: int = (
        batch_size if batch_size is not None else _BATCH_SIZES.get(model_key, 64)
    )
    return builders[model_key](effective_batch_size)
