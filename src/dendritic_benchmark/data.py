import csv
import importlib
import math
import os
import tarfile
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import torch

from .models import (
    ADULT_CATEGORICAL_CARDINALITIES,
    ADULT_FEATURES,
    ADULT_NUMERIC_COLUMNS,
    EDGE_AROMATIC,
    EDGE_DOUBLE,
    EDGE_IN_RING,
    EDGE_SELF_LOOP,
    EDGE_SINGLE,
    EDGE_TRIPLE,
    ETT_FORECAST_HORIZON,
    FORECAST_SEQ_LEN,
    MOLECULE_EDGE_FEATURES,
    MOLECULE_NODE_FEATURES,
    SOCIAL_GRAPH_NODE_FEATURES,
    WEATHER_FORECAST_HORIZON,
    RunningObsNorm,
)

DATA_ROOT_ENV: str = "DQB_DATA_ROOT"
DEFAULT_DATA_ROOT: str = "data"
EXTRACTED_MARKER: str = ".extracted"
# Versioned: a cached rollout file records whichever heuristic produced it and
# whatever payload schema was current when it was written. Reusing one after
# changing either would train on stale labels or fail on a missing key, so the
# version is bumped for both kinds of change. v3 added per-step episode ids.
HEURISTIC_ROLLOUTS_FILENAME: str = "heuristic_rollouts_v3.pt"
# Safety net only; each environment's own TimeLimit truncates first.
_RL_ROLLOUT_STEP_CAP: int = 2000
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
# Cora is trained transductively on the whole graph — every node's features and
# edges are present in every forward pass, and only the labels are split. See
# _CoraTransductiveDataset for why the previous fixed-width 2-hop ego graphs
# were not the task Kipf & Welling's 81.5% measures.
CORA_NODES: int = 2708
CORA_NODE_FEATURES: int = 1433
# The Planetoid semi-supervised split that every published Cora/Citeseer/Pubmed
# number is measured on: 20 labelled nodes per class, then 500 validation and
# 1000 test nodes. See _planetoid_style_split.
PLANETOID_LABELS_PER_CLASS: int = 20
PLANETOID_VAL_NODES: int = 500
PLANETOID_TEST_NODES: int = 1000
# ETT's published train/validation/test division is calendar-based rather than a
# ratio: 12 months / 4 / 4, counted in each file's own sampling period. ETTh1 is
# hourly and ETTm1 is the same series resampled to 15 minutes.
_ETT_SPLIT_HOURLY: tuple[int, int, int] = (12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24)
_ETT_SPLIT_15MIN: tuple[int, int, int] = (
    _ETT_SPLIT_HOURLY[0] * 4,
    _ETT_SPLIT_HOURLY[1] * 4,
    _ETT_SPLIT_HOURLY[2] * 4,
)
# Weather has no calendar convention attached to it; Autoformer and everything
# downstream of it splits 70/10/20 chronologically.
_WEATHER_SPLIT_RATIOS: tuple[float, float] = (0.7, 0.1)
# Degree buckets for the IMDB-BINARY featuriser: channel 0 is the real-node
# indicator and channels 1..8 are a one-hot over log2 degree, matching the
# degree-as-feature convention Xu et al. use for label-free social graphs.
_SOCIAL_DEGREE_BUCKETS: int = SOCIAL_GRAPH_NODE_FEATURES - 2
# Kept in step with TextCNN's `vocab_size` default in models.py.
AG_NEWS_VOCAB_SIZE: int = 20_000
AG_NEWS_SEQ_LEN: int = 128
# Fraction of GLUE's 872-row SST-2 dev set held out as validation, leaving the
# rest as test. SST-2's *train* split is phrase-level, so a validation set drawn
# from it shares constituency subtrees with training rows; see TextDataSets.sst2.
# 0.3 keeps ~610 test rows (standard error ~1.2%) while giving the plateau
# detector ~262 rows, enough that one flipped prediction moves it by 0.38%.
SST2_DEV_VALIDATION_RATIO: float = 0.3
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
    # Set only for on-policy reinforcement learning, where the training data is
    # a function of the current weights rather than a fixed file. Holds a
    # ``PPORolloutSource``; ``_run_epoch_batches`` calls ``collect(model, device)``
    # at the top of every epoch and replaces ``train_loader`` with the result.
    # ``ppo_bipedalwalker`` is the only model in the suite that sets it, and it
    # is the one property that makes that model structurally unlike the other
    # 21 — worth checking before assuming a bundle's loaders are static.
    on_policy: Any | None = None

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
    """Import *import_name*, or raise with an install hint naming *package_name*.

    ``importlib.import_module`` rather than ``__import__``: the latter hands
    back the *top-level* package for a dotted name, so asking for
    "gymnasium.envs.box2d.lunar_lander" would return bare ``gymnasium`` and the
    attribute lookup would fail somewhere far from here.
    """
    try:
        return importlib.import_module(import_name)
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


def _planetoid_style_split(
    labels: Any, *, num_classes: int, seed: int = 42
) -> tuple[list[int], list[int], list[int]]:
    """Planetoid's semi-supervised node split: 20 per class / 500 val / 1000 test.

    Cora was previously split 70/15/15 at random, which trains on 1895 labelled
    nodes. Kipf & Welling's 81.5% — and every number quoted against it — is
    measured with **140**. Training on 13x the labels and then reading the
    result against the published figure compares two different problems, so the
    87.47% that setup produced was not evidence of a stronger GCN.

    The node identities here differ from Planetoid's own fixed index file, which
    is ordered differently from ``cora.content``. Shchur et al. ("Pitfalls of
    Graph Neural Network Evaluation") re-draw splits of exactly these sizes at
    random and still place GCN at 81.5 +/- 1.3, so it is the sizes that carry the
    comparison rather than the particular nodes. The draw is seeded, so it is
    fixed across runs and identical in the baseline and dendritic arms.
    """
    generator = torch.Generator().manual_seed(seed)
    order: list[int] = torch.randperm(len(labels), generator=generator).tolist()
    train_idx: list[int] = []
    remaining: list[int] = []
    per_class: Counter[int] = Counter()
    for index in order:
        label = int(labels[index])
        if per_class[label] < PLANETOID_LABELS_PER_CLASS:
            per_class[label] += 1
            train_idx.append(index)
        else:
            remaining.append(index)
    expected: int = PLANETOID_LABELS_PER_CLASS * num_classes
    if len(train_idx) != expected:
        raise ValueError(
            f"expected {expected} training nodes ({PLANETOID_LABELS_PER_CLASS} per "
            f"class across {num_classes} classes), got {len(train_idx)}"
        )
    needed: int = PLANETOID_VAL_NODES + PLANETOID_TEST_NODES
    if len(remaining) < needed:
        raise ValueError(
            f"need {needed} nodes beyond the training set, only "
            f"{len(remaining)} available"
        )
    return (
        train_idx,
        remaining[:PLANETOID_VAL_NODES],
        remaining[PLANETOID_VAL_NODES:needed],
    )


def _stratified_holdout(
    labels: Any, *, holdout_ratio: float, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Split row indices into (holdout, remainder), keeping the label mix in both.

    Used to cut a validation set out of an evaluation split that is too small
    for the class balance to survive an unstratified draw — GLUE's SST-2 dev set
    is 872 rows, where a plain random 30% can drift the positive rate by several
    points and move the accuracy it reports for reasons that have nothing to do
    with the model. Seeded, so the two splits are fixed across runs and
    identical in the baseline and dendritic arms.
    """
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError(f"holdout_ratio must be in (0, 1), got {holdout_ratio}")
    generator = torch.Generator().manual_seed(seed)
    holdout: list[int] = []
    remainder: list[int] = []
    for label in torch.unique(labels).tolist():
        rows = (labels == label).nonzero(as_tuple=True)[0]
        order = torch.randperm(len(rows), generator=generator)
        shuffled = rows[order].tolist()
        # Clamped so neither side of a small class can come out empty.
        count = min(max(1, round(len(shuffled) * holdout_ratio)), len(shuffled) - 1)
        holdout.extend(shuffled[:count])
        remainder.extend(shuffled[count:])
    return sorted(holdout), sorted(remainder)


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


class _TensorRowsDataset(torch.utils.data.Dataset[tuple[Any, ...]]):
    """Row-wise view over a set of equal-length tensors.

    Subclasses ``Dataset`` rather than standing alone so that it type-checks
    where torch expects a ``Dataset`` — chiefly ``Subset``, which
    ``_standardized_regression_bundle`` wraps it in.
    """

    def __init__(self, *tensors: Any) -> None:
        self.tensors: tuple[Any, ...] = tensors

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        return tuple(tensor[index] for tensor in self.tensors)


def _forecast_windows(
    series: Any, start: int, stop: int, seq_len: int, horizon: int
) -> tuple[Any, Any]:
    """Sliding (lookback, horizon) pairs whose *targets* lie inside [start, stop).

    The first lookback window opens ``seq_len`` steps before ``start``, so a
    validation or test window may read timesteps belonging to the split before
    it. That is the Informer/Autoformer convention and it is not leakage: those
    steps are model inputs, never targets, and every predicted timestep falls
    strictly inside ``[start, stop)``. Starting each split cold instead would
    silently drop its first ``seq_len`` targets.
    """
    first: int = max(0, start - seq_len)
    last: int = stop - seq_len - horizon
    if last < first:
        raise ValueError(
            f"forecast split [{start}, {stop}) cannot hold a {seq_len}-step "
            f"lookback plus a {horizon}-step horizon"
        )
    indices = range(first, last + 1)
    xs = torch.stack([series[index : index + seq_len] for index in indices])
    ys = torch.stack(
        [series[index + seq_len : index + seq_len + horizon] for index in indices]
    )
    return xs, ys


def _chronological_forecast_bundle(
    values: Any,
    *,
    seq_len: int,
    horizon: int,
    split_sizes: tuple[int, int, int],
    batch_size: int,
    input_description: str,
    univariate: bool = False,
    num_workers: int = 2,
) -> TaskBundle:
    """Window a ``[T, C]`` series into chronologically disjoint forecasting splits.

    Two properties here are what let the reported MAE be read against published
    forecasting results:

    * **The splits are contiguous in time.** These datasets were previously fed
      through ``_split_dataset``, which calls ``random_split`` over the *windows*.
      Window ``i`` and window ``i+1`` share ``seq_len - 1`` of their ``seq_len``
      timesteps, so a random assignment routinely put one in train and its
      near-duplicate in test — and a test window's target timestep was usually
      also sitting inside some training window's lookback. Every forecasting
      benchmark splits chronologically for exactly this reason.
    * **Normalisation is fitted on the training span alone.** The previous code
      z-scored using statistics over the whole file, which hands the model the
      test period's level and scale before training starts.

    Metrics stay on the z-scored scale, which is also what Informer, Autoformer
    and their successors report, so no inverse transform is applied here.
    """
    num_train, num_val, num_test = split_sizes
    total: int = num_train + num_val + num_test
    if total > len(values):
        raise ValueError(
            f"split sizes sum to {total} rows but the series has {len(values)}"
        )
    train_span = values[:num_train]
    mean = train_span.mean(dim=0, keepdim=True)
    std = train_span.std(dim=0, keepdim=True).clamp_min(1e-6)
    series = (values - mean) / std

    borders: tuple[tuple[int, int], ...] = (
        (0, num_train),
        (num_train, num_train + num_val),
        (num_train + num_val, total),
    )
    splits: list[Any] = []
    for start, stop in borders:
        x, y = _forecast_windows(series, start, stop, seq_len, horizon)
        # A univariate task keeps the channel axis on the input so the model
        # still sees [B, seq_len, 1], but drops it from the target to match a
        # head that predicts a single series over the horizon.
        splits.append(_TensorRowsDataset(x, y.squeeze(-1) if univariate else y))
    return _bundle_from_splits(
        splits[0],
        splits[1],
        splits[2],
        batch_size,
        "MAE",
        "minimize",
        input_description,
        num_workers=num_workers,
    )


class TimeSeriesDatasets:
    @staticmethod
    def _numeric_columns(path: Path) -> Any:
        """Read every numeric, non-date column of a CSV into a ``[T, C]`` tensor."""
        rows: list[list[float]] = []
        with path.open(newline="") as fh:
            reader: csv.DictReader[str] = csv.DictReader(fh)
            for row in reader:
                values: list[float] = []
                for key, value in row.items():
                    if key is None or key.lower() == "date":
                        continue
                    try:
                        values.append(float(value))
                    except (TypeError, ValueError):
                        continue  # skip non-numeric columns
                if values:
                    rows.append(values)
        return torch.tensor(rows, dtype=torch.float32)

    @staticmethod
    def etth1(batch_size: int) -> TaskBundle:
        """ETTh1 univariate (the OT column), 96-step lookback, 24-step horizon.

        This is Informer's ETTh1 univariate setting, reported there at MSE 0.098
        / MAE 0.247 with recurrent baselines in the high 0.2s. The previous setup
        predicted one step ahead from a 24-step lookback, which no published
        result uses — and under the random window split that made it easier
        still, it scored MAE 0.069 against numbers that are not its own.
        """
        path: Path = _download(ETTH1_URL, _data_root() / "etth1" / "ETTh1.csv")
        rows: list[float] = []
        with path.open(newline="") as fh:
            reader: csv.DictReader[str] = csv.DictReader(fh)
            for row in reader:
                rows.append(float(row["OT"]))
        return _chronological_forecast_bundle(
            torch.tensor(rows, dtype=torch.float32).unsqueeze(-1),
            seq_len=FORECAST_SEQ_LEN,
            horizon=ETT_FORECAST_HORIZON,
            split_sizes=_ETT_SPLIT_HOURLY,
            batch_size=batch_size,
            input_description=(
                "ETTh1 hourly oil temperature, 96-step lookback to 24-step "
                "univariate horizon (Informer protocol)"
            ),
            univariate=True,
        )

    @staticmethod
    def ettm1(batch_size: int) -> TaskBundle:
        """ETTm1 multivariate, 96-step lookback, 24-step horizon.

        Informer's ETTm1 multivariate setting: MSE 0.323 / MAE 0.369 there. The
        window geometry was already right; the split and the normalisation were
        not.
        """
        return _chronological_forecast_bundle(
            TimeSeriesDatasets._numeric_columns(
                _download(ETTM1_URL, _data_root() / "ettm1" / "ETTm1.csv")
            ),
            seq_len=FORECAST_SEQ_LEN,
            horizon=ETT_FORECAST_HORIZON,
            split_sizes=_ETT_SPLIT_15MIN,
            batch_size=batch_size,
            input_description=(
                "ETTm1 15-minute 7-variate windows, 96-step lookback to 24-step "
                "horizon (Informer protocol)"
            ),
        )

    @staticmethod
    def weather(batch_size: int) -> TaskBundle:
        """Weather 21-variate, 96-step lookback, 96-step horizon.

        Horizon 96 rather than 24: the 21-variable Weather set entered the
        literature with Autoformer, which reports it at {96, 192, 336, 720} and
        scores 0.266 MSE / 0.336 MAE at the shortest. Horizon 24 has no
        published counterpart on this dataset.
        """
        datasets = _require_dependency("datasets")
        loaded = datasets.load_dataset(
            "dunzane/time-series-dataset", "Weather", cache_dir=_hf_dataset_cache()
        )
        split = loaded["train"]
        columns: list[Any] = [
            name
            for name in split.column_names
            if name.lower() != "date"
            and split.features[name].dtype
            in {"float32", "float64", "int32", "int64"}
        ]
        rows: list[list[float]] = [
            [float(row[name]) for name in columns] for row in split
        ]
        values = torch.tensor(rows, dtype=torch.float32)
        total: int = len(values)
        train_ratio, val_ratio = _WEATHER_SPLIT_RATIOS
        num_train: int = int(total * train_ratio)
        num_val: int = int(total * val_ratio)
        return _chronological_forecast_bundle(
            values,
            seq_len=FORECAST_SEQ_LEN,
            horizon=WEATHER_FORECAST_HORIZON,
            split_sizes=(num_train, num_val, total - num_train - num_val),
            batch_size=batch_size,
            input_description=(
                "Weather 21-variate meteorological windows, 96-step lookback to "
                "96-step horizon (Autoformer protocol)"
            ),
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
        """SST-2 with validation and test both carved out of the GLUE dev set.

        Validation used to be a random 10% of GLUE's ``train`` split. That split
        is **phrase-level**: Stanford parsed each sentence into a constituency
        tree and labelled every subtree, so one sentence contributes the full
        sentence plus many overlapping sub-phrases. A random row-wise draw
        therefore routinely put a phrase in validation whose parent sentence sat
        in training, and the measured effect was large — validation read 95.19%
        against a test accuracy of 90.48%.

        The reported *test* number was never wrong (it is the real GLUE dev set,
        and 90.48% sits on the published ~91%). The reason to care is
        PerforatedAI: its switch logic reads the validation metric, so a signal
        inflated by 4.7 points and moving for the wrong reasons is a poor
        plateau detector, and dendrite insertion timing depends on it.

        Both evaluation splits now come from the 872-row GLUE dev set, which is
        one row per sentence with no phrase overlap against anything. The
        trade-off is deliberate and has to be stated wherever the number is
        read: test is ~610 rows rather than 872, so its standard error rises
        from roughly 1.0% to 1.2%, and it is no longer the whole standard dev
        set. Training still uses every phrase-level train row — phrase
        augmentation is what SST-2's train split is *for*, and it only becomes a
        leak when the same tree lands on both sides of an evaluation boundary.
        """
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
        dev_ids, dev_mask = _tokenize(loaded["validation"]["sentence"])
        y_dev = torch.tensor(list(loaded["validation"]["label"]), dtype=torch.long)
        dev_dataset = _TensorRowsDataset(dev_ids, dev_mask, y_dev)
        val_idx, test_idx = _stratified_holdout(
            y_dev, holdout_ratio=SST2_DEV_VALIDATION_RATIO
        )
        subset = torch.utils.data.Subset
        return _bundle_from_splits(
            _TensorRowsDataset(train_ids, train_mask, y_train),
            subset(dev_dataset, val_idx),
            subset(dev_dataset, test_idx),
            batch_size,
            "Accuracy",
            "maximize",
            "SST-2 sentences tokenized with distilbert-base-uncased tokenizer; "
            "validation and test are disjoint halves of the GLUE dev set",
        )

class _CoraTransductiveDataset:
    """The whole Cora graph as a single item, plus the node indices to score.

    Cora node classification used to be cut into one fixed-width 2-hop ego graph
    per labelled node, 64 nodes wide, batched 32 at a time. That is a legitimate
    task — inductive GraphSAGE-style classification — but it is not the task
    Kipf & Welling's 81.5% measures, and the difference is not a detail:

    | | Kipf transductive | ego-graph |
    |---|---|---|
    | receptive field | the full 2708-node graph | 2 hops, capped at 64 nodes |
    | batching | one graph, one optimiser step | 32 independent subgraphs |
    | unlabelled nodes | propagate information | only where a subgraph reaches |
    | high-degree node | every neighbour present | truncated to 64 |

    Cora's 2-hop closed neighbourhood has a 75th percentile of 40 nodes but a
    long tail, so the 64-slot cap silently truncated the best-connected nodes —
    exactly the ones a graph convolution has most to say about. And because
    every subgraph was scored independently, a test node's 2-hop neighbourhood
    was re-encoded from scratch rather than sharing the representation the rest
    of the graph had built.

    The transductive setup restores the reference mechanism. One item holds the
    full feature matrix and the full normalised-input adjacency; the split
    supplies which node indices its loss and metrics are computed over. Every
    node's features and edges are visible in every forward pass, of every split
    — that is what "transductive" means and it is not a leak, because only the
    *labels* are partitioned. ``_planetoid_style_split`` already partitions
    those 20-per-class / 500 / 1000.

    One item per split means one optimiser step per epoch, which is also the
    reference setup: Kipf & Welling run 200 epochs of full-batch descent, i.e.
    200 steps, which the 200-epoch recipe now reproduces exactly.

    Attributes are stored on the instance rather than captured in a closure so
    the class stays picklable for DataLoader worker processes.
    """

    def __init__(self, adjacency: Any, x_all: Any, node_indices: list[int], y_all: Any) -> None:
        self.adjacency = adjacency
        self.x_all = x_all
        self.node_indices = torch.tensor(node_indices, dtype=torch.long)
        self.labels = y_all[self.node_indices]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[Any, Any, Any, Any]:
        if index != 0:
            raise IndexError(
                "the transductive Cora dataset holds one item (the whole graph); "
                f"got index {index}"
            )
        return self.x_all, self.adjacency, self.node_indices, self.labels

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
                # 0-based: these codes index the models' embedding tables.
                mapping[value] = len(mapping)
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
    """Adult census rows: standardised numerics, raw codes for nominal columns.

    Every column used to be z-scored together, categorical codes included, which
    presented eight nominal variables to the models as though they were ordered
    measurements. TabNet and SAINT both embed categoricals instead, so the codes
    now pass through untouched and ``TabularColumnEmbedding`` looks them up.
    """
    root: Path = _data_root() / "adult"
    for filename, url in ADULT_URLS.items():
        _download(url, root / filename)
    train_rows: list[list[str]] = _parse_adult_file(root / ADULT_DATA_FILENAME)
    test_rows: list[list[str]] = _parse_adult_file(root / "adult.test")
    feature_count = ADULT_FEATURES
    encoders: list[dict[str, int]] = [{} for _ in range(feature_count)]
    numeric_columns: set[int] = set(ADULT_NUMERIC_COLUMNS)

    train_x_raw, train_y_raw = _encode_adult_rows(train_rows, encoders, numeric_columns, feature_count)
    test_x_raw, test_y_raw = _encode_adult_rows(test_rows, encoders, numeric_columns, feature_count)
    train_x = torch.tensor(train_x_raw, dtype=torch.float32)
    test_x = torch.tensor(test_x_raw, dtype=torch.float32)

    # The models size their embedding tables from the constant schema, so an
    # unexpected category would index past the end of a table. Adult is closed
    # and this holds today; catch it here rather than in a CUDA/MPS assert.
    for column, cardinality in ADULT_CATEGORICAL_CARDINALITIES.items():
        observed: int = int(
            max(train_x[:, column].max().item(), test_x[:, column].max().item())
        )
        if observed >= cardinality:
            raise ValueError(
                f"Adult column {column} produced code {observed}, beyond the "
                f"declared cardinality {cardinality}; update "
                "ADULT_CATEGORICAL_CARDINALITIES in models.py."
            )

    # Standardise the numeric columns only, on training rows only. Categorical
    # codes stay as integers for the embedding lookup.
    numeric_index = torch.tensor(sorted(numeric_columns), dtype=torch.long)
    mean = train_x[:, numeric_index].mean(dim=0, keepdim=True)
    std = train_x[:, numeric_index].std(dim=0, keepdim=True).clamp_min(1e-6)
    for matrix in (train_x, test_x):
        matrix[:, numeric_index] = (matrix[:, numeric_index] - mean) / std

    train_ds, val_ds, _ = _split_dataset(
        _TensorRowsDataset(train_x, torch.tensor(train_y_raw, dtype=torch.long)),
        train_ratio=0.9,
        val_ratio=0.1,
    )
    test_ds = _TensorRowsDataset(test_x, torch.tensor(test_y_raw, dtype=torch.long))
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
        node_count = len(paper_ids)
        # A-hat = A + I, symmetric. The self-loop is Kipf & Welling's renormalisation
        # trick, so every node keeps its own features through the convolution.
        adjacency = torch.eye(node_count, dtype=torch.float32)
        with cites.open() as fh:
            for line in fh:
                src, dst = line.strip().split()
                if src in id_to_idx and dst in id_to_idx:
                    i, j = id_to_idx[src], id_to_idx[dst]
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0

        splits = _planetoid_style_split(y_all, num_classes=len(label_to_idx))
        # Each split is one item carrying the whole graph and its own node
        # indices, so all three see identical features and edges and differ only
        # in which labels are scored. batch_size is ignored on purpose: a
        # transductive epoch is one full-graph step, which is the reference
        # setup, and _BATCH_SIZES pins the recipe to 1 to keep that visible.
        train_ds, val_ds, test_ds = (
            _CoraTransductiveDataset(adjacency, x_all, node_indices, y_all)
            for node_indices in splits
        )
        return _bundle_from_splits(
            train_ds,
            val_ds,
            test_ds,
            1,
            "Accuracy",
            "maximize",
            "Cora full citation graph, transductive Planetoid split "
            f"({len(splits[0])} train / {len(splits[1])} val / {len(splits[2])} test labels)",
            # The whole graph is one already-materialised tensor pair; a worker
            # process would only add a 44 MB pickle round-trip per epoch.
            num_workers=0,
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
    def _ring_bond_flags(atom_count: int, bonds: list[_Bond]) -> list[bool]:
        """Mark every bond that lies on a cycle, by finding the graph's bridges.

        ``_Bond.ring_closure`` marks only the one bond that *closes* a SMILES
        ring, so the previous featuriser set ``in_ring`` on two atoms of a
        benzene and left the other four at zero — aromaticity and ring
        membership are among the strongest signals for solubility, and the
        models were being shown a corrupted version of both.

        A bond lies on a cycle exactly when it is not a bridge, so one
        bridge-finding pass labels every ring bond correctly, fused and
        spiro systems included. Iterative rather than recursive: the DFS depth
        is the molecule's atom count, and edge ids (not endpoint pairs) mark the
        parent so parallel bonds are handled.
        """
        neighbours: list[list[tuple[int, int]]] = [[] for _ in range(atom_count)]
        for edge_id, bond in enumerate(bonds):
            if bond.source == bond.target:
                continue
            if not (0 <= bond.source < atom_count and 0 <= bond.target < atom_count):
                continue
            neighbours[bond.source].append((bond.target, edge_id))
            neighbours[bond.target].append((bond.source, edge_id))

        discovery: list[int] = [-1] * atom_count
        low: list[int] = [0] * atom_count
        on_cycle: list[bool] = [False] * len(bonds)
        timer: int = 0
        for root in range(atom_count):
            if discovery[root] != -1:
                continue
            discovery[root] = low[root] = timer
            timer += 1
            # Each frame is [node, edge id we arrived by, next neighbour index].
            stack: list[list[int]] = [[root, -1, 0]]
            while stack:
                node, parent_edge, cursor = stack[-1]
                if cursor < len(neighbours[node]):
                    stack[-1][2] += 1
                    neighbour, edge_id = neighbours[node][cursor]
                    if edge_id == parent_edge:
                        continue
                    if discovery[neighbour] == -1:
                        discovery[neighbour] = low[neighbour] = timer
                        timer += 1
                        stack.append([neighbour, edge_id, 0])
                    else:
                        low[node] = min(low[node], discovery[neighbour])
                    continue
                stack.pop()
                if not stack:
                    continue
                parent = stack[-1][0]
                low[parent] = min(low[parent], low[node])
                # low[node] > discovery[parent] means nothing below this edge can
                # reach back above it: a bridge. Anything else closes a cycle.
                if low[node] <= discovery[parent]:
                    on_cycle[parent_edge] = True
        return on_cycle

    @staticmethod
    def _bond_type_channel(bond: _Bond, atoms: list["_ParsedAtom"]) -> int:
        """Pick the one-hot bond-order channel for *bond*.

        Aromaticity is read off the endpoints rather than the bond order:
        ``c1ccccc1`` writes no explicit bond symbol, so its ring bonds parse at
        order 1.0 and only the lowercase atoms record the aromaticity.
        """
        if atoms[bond.source].aromatic and atoms[bond.target].aromatic:
            return EDGE_AROMATIC
        if bond.order >= 3.0:
            return EDGE_TRIPLE
        if bond.order >= 2.0:
            return EDGE_DOUBLE
        if bond.order == 1.5:
            return EDGE_AROMATIC
        return EDGE_SINGLE

    @staticmethod
    def _build_graph_tensors(
        atoms: list["_ParsedAtom"], bonds: list[_Bond], max_nodes: int
    ) -> tuple[Any, Any, Any]:
        """Featurise a parsed molecule into (node features, adjacency, edge features).

        Padded rows are left all-zero, which is what the molecular models use to
        tell real atoms from padding (see ``MPNN.forward``/``AttentiveFP.forward``);
        the adjacency's self-loops cannot serve that purpose.
        """
        width: int = MOLECULE_NODE_FEATURES
        x = torch.zeros((max_nodes, width), dtype=torch.float32)
        adjacency = torch.eye(max_nodes, dtype=torch.float32)
        edges = torch.zeros(
            (max_nodes, max_nodes, MOLECULE_EDGE_FEATURES), dtype=torch.float32
        )
        # Flag the self-loops the adjacency carries so the message function can
        # tell "this atom, no bond" from "padding".
        diagonal = torch.arange(max_nodes)
        edges[diagonal, diagonal, EDGE_SELF_LOOP] = 1.0
        kept: int = min(len(atoms), max_nodes)
        on_cycle: list[bool] = GraphDatasets._ring_bond_flags(len(atoms), bonds)
        degree = [0.0] * kept
        double_bond = [0.0] * kept
        triple_bond = [0.0] * kept
        in_ring = [0.0] * kept
        for edge_id, bond in enumerate(bonds):
            if bond.source >= kept or bond.target >= kept or bond.source == bond.target:
                continue
            adjacency[bond.source, bond.target] = 1.0
            adjacency[bond.target, bond.source] = 1.0
            feature = torch.zeros(MOLECULE_EDGE_FEATURES, dtype=torch.float32)
            feature[GraphDatasets._bond_type_channel(bond, atoms)] = 1.0
            if on_cycle[edge_id]:
                feature[EDGE_IN_RING] = 1.0
            edges[bond.source, bond.target] = feature
            edges[bond.target, bond.source] = feature
            for end in (bond.source, bond.target):
                degree[end] += 1.0
                if bond.order >= 3.0:
                    triple_bond[end] = 1.0
                elif bond.order >= 2.0:
                    double_bond[end] = 1.0
                if on_cycle[edge_id]:
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
        return x, adjacency, edges

    @staticmethod
    def _smiles_to_graph(smiles: str, max_nodes: int) -> tuple[Any, Any, Any]:
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
        edge_features: list[Any] = []
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
                x, adjacency, edges = GraphDatasets._smiles_to_graph(smiles, max_nodes)
                node_features.append(x)
                adjacencies.append(adjacency)
                edge_features.append(edges)
                labels.append(float(target))
        return _standardized_regression_bundle(
            (
                torch.stack(node_features),
                torch.stack(adjacencies),
                torch.stack(edge_features),
            ),
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


def _collect_heuristic_rollouts(
    env_id: str,
    make_policy: Callable[[Any], Callable[[Any], Any]],
    *,
    samples: int,
    cache: Path,
    discrete: bool,
    explore_probability: float = 0.15,
    seed: int = 42,
    package_hint: str = "gymnasium[box2d]",
) -> tuple[Any, Any, Any]:
    """Cache (observation, heuristic action, episode id) triples for cloning.

    Used by ``actor_critic`` and ``dqn_lunarlander``, which stay behaviour
    cloning. ``ppo_bipedalwalker`` no longer comes through here: cloning a
    *stateful* gait controller with a feedforward policy is ill-posed, so it
    trains on policy instead — see ``PPORolloutSource``.

    ``make_policy(env)`` returns a *fresh* per-episode callable, so a stateful
    heuristic is reset at every episode boundary rather than carrying state
    across resets.

    Two properties here decide how well the cloned policy actually flies, as
    opposed to how well it matches labels:

    * **Every episode gets its own seed.** The previous collector seeded only
      the first reset, so the entire cache came from one chain of near-identical
      starts and covered a narrow slice of the state space.
    * **A fraction of steps take a random action while still recording what the
      heuristic would have done.** Pure on-policy cloning only ever sees states
      the heuristic itself reaches, so the network never learns to recover from
      its own small errors and they compound: the CartPole clone returned 292
      against a heuristic that scores a perfect 500. Labelling off-policy states
      with the heuristic's correct response is the standard fix.
    """
    gymnasium = _require_dependency("gymnasium", package_hint)
    numpy = _require_dependency("numpy")
    if cache.exists():
        # weights_only=True: this cache only ever holds the {"x", "y", "episode"}
        # tensor dict saved a few lines below, so the restricted unpickler is
        # safe here and closes off arbitrary code execution from a malicious
        # or corrupted cache file.
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        return payload["x"], payload["y"], payload["episode"]

    cache.parent.mkdir(parents=True, exist_ok=True)
    env = gymnasium.make(env_id)
    env.action_space.seed(seed)
    rng = numpy.random.default_rng(seed)
    observations: list[Any] = []
    actions: list[Any] = []
    episodes: list[int] = []
    episode: int = 0
    try:
        while len(observations) < samples:
            observation, _ = env.reset(seed=seed + episode)
            policy = make_policy(env)
            for _ in range(_RL_ROLLOUT_STEP_CAP):
                label = policy(observation)
                observations.append(observation)
                actions.append(label)
                episodes.append(episode)
                step_action = (
                    env.action_space.sample()
                    if rng.random() < explore_probability
                    else label
                )
                observation, _, terminated, truncated, _ = env.step(step_action)
                if terminated or truncated or len(observations) >= samples:
                    break
            episode += 1
    finally:
        env.close()

    x = torch.tensor(numpy.array(observations), dtype=torch.float32)
    y = torch.tensor(
        numpy.array(actions), dtype=torch.long if discrete else torch.float32
    )
    episode_ids = torch.tensor(episodes, dtype=torch.long)
    torch.save({"x": x, "y": y, "episode": episode_ids}, cache)
    return x, y, episode_ids


def _split_by_episode(
    dataset: Any,
    episode_ids: Any,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Any, Any, Any]:
    """Split trajectory data so that no episode spans two splits.

    A rollout cache is a sequence, not a bag of independent rows: consecutive
    timesteps in one episode differ by a single simulator tick and are nearly
    identical. Splitting those rows at random — which is what ``_split_dataset``
    does — puts step *t* in training and step *t+1* in test, so the reported
    action accuracy is substantially a measure of how well the network
    interpolates between two frames of the same episode. It is the same defect
    the forecasting windows had.

    Whole episodes are assigned instead. Episodes are independent of one another
    (each starts from its own seeded reset), so a random assignment *of
    episodes* is sound; it is only the within-episode correlation that has to be
    kept on one side of the split.
    """
    unique = torch.unique(episode_ids).tolist()
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(unique), generator=generator).tolist()
    shuffled = [unique[position] for position in order]

    train_count = max(1, int(len(shuffled) * train_ratio))
    val_count = max(1, int(len(shuffled) * val_ratio))
    if train_count + val_count >= len(shuffled):
        raise ValueError(
            f"{len(shuffled)} episodes cannot fill a "
            f"{train_ratio}/{val_ratio} train/validation split with a test "
            "remainder; collect more samples"
        )
    assignment = {
        "train": set(shuffled[:train_count]),
        "val": set(shuffled[train_count : train_count + val_count]),
        "test": set(shuffled[train_count + val_count :]),
    }
    episode_list: list[int] = episode_ids.tolist()
    subset = torch.utils.data.Subset

    def split_for(name: str) -> Any:
        return subset(
            dataset,
            [
                index
                for index, episode in enumerate(episode_list)
                if episode in assignment[name]
            ],
        )

    # Built as a literal rather than tuple(... for name in ...): a generator
    # gives tuple[Subset, ...] of unknown length, which does not satisfy the
    # three-tuple this returns into.
    return split_for("train"), split_for("val"), split_for("test")


def _build_cartpole(batch_size: int) -> TaskBundle:
    x, y, episode_ids = _collect_heuristic_rollouts(
        "CartPole-v1",
        # Scores a perfect 500 in the environment, so this one stays as written.
        lambda _env: (lambda obs: 1 if (obs[2] + 0.25 * obs[3]) > 0 else 0),
        samples=20_000,
        cache=_data_root() / "cartpole" / HEURISTIC_ROLLOUTS_FILENAME,
        discrete=True,
        package_hint="gymnasium[classic-control]",
    )
    # Not a reward: no environment is stepped during training or validation.
    # This is behaviour cloning, and the score is the fraction of held-out
    # observations where the network picks the same discrete action as the
    # heuristic. An episodic return *is* now measured once at the end of
    # training — see _evaluate_episodic_return in training.py — and that is the
    # number to read against published CartPole results.
    return _bundle_from_splits(
        *_split_by_episode(_TensorRowsDataset(x, y), episode_ids),
        batch_size,
        "Action Accuracy",
        "maximize",
        "CartPole-v1 observations labeled by a stabilizing policy",
    )


def _build_lunarlander(batch_size: int) -> TaskBundle:
    """LunarLander observations labelled by Gymnasium's own reference heuristic.

    The hand-written policy this used to clone returns **-519** in the
    environment — it crashes the lander on essentially every episode. The model
    reproduced it faithfully (98.8% action agreement, -523 return), so the
    headline metric looked excellent while the policy was worthless, and no
    episodic number here could be read against LunarLander's 200-point solved
    threshold. ``gymnasium.envs.box2d.lunar_lander.heuristic`` is the reference
    policy shipped with the environment and returns roughly +230, above solved.
    """

    def make_policy(env: Any) -> Callable[[Any], int]:
        lunar_lander = _require_dependency(
            "gymnasium.envs.box2d.lunar_lander", "gymnasium[box2d]"
        )
        unwrapped = env.unwrapped
        return lambda observation: int(lunar_lander.heuristic(unwrapped, observation))

    x, y, episode_ids = _collect_heuristic_rollouts(
        "LunarLander-v3",
        make_policy,
        samples=40_000,
        cache=_data_root() / "lunarlander" / HEURISTIC_ROLLOUTS_FILENAME,
        discrete=True,
    )
    # Behaviour cloning, not reinforcement learning — see _build_cartpole.
    return _bundle_from_splits(
        *_split_by_episode(_TensorRowsDataset(x, y), episode_ids),
        batch_size,
        "Action Accuracy",
        "maximize",
        "LunarLander-v3 observations labeled by Gymnasium's reference heuristic",
    )


# ----------------------------------------------------------------- PPO ------
# BipedalWalker is the one model in the suite trained by reinforcement learning
# rather than on a fixed dataset. Values follow Stable-Baselines3's RL Zoo entry
# for BipedalWalker-v3 + PPO, which is the tuned reference configuration.
BIPEDAL_PPO_ROLLOUT_STEPS: int = 2048
BIPEDAL_PPO_GAMMA: float = 0.999
BIPEDAL_PPO_GAE_LAMBDA: float = 0.95
# Passes over each rollout buffer per PPO iteration (SB3's `n_epochs`). One
# benchmark epoch is one PPO iteration, so a "batch" count per epoch is
# ceil(BIPEDAL_PPO_ROLLOUT_STEPS * this / minibatch).
BIPEDAL_PPO_UPDATE_PASSES: int = 10
# Rewards are divided by the running standard deviation of the discounted return
# before advantages are computed, then clipped. This is CleanRL's NormalizeReward
# wrapper. It only rescales the critic's targets — the reported episodic return
# is always the raw environment reward, so published comparisons are unaffected.
BIPEDAL_PPO_REWARD_CLIP: float = 10.0


class _RunningScalarNorm:
    """Running mean/variance of a scalar stream, for reward scaling.

    Deliberately *not* an ``nn.Module``: unlike the observation statistics, this
    never touches the network's forward pass, so it belongs to the collector and
    not to ``model.pt``. A checkpoint resume therefore re-warms it over the first
    couple of iterations, which is harmless — it only rescales the value target.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = 1e-4
        self.epsilon = epsilon

    def update(self, values: Any) -> None:
        batch_count = int(values.numel())
        if batch_count == 0:
            return
        batch_mean = float(values.mean().item())
        batch_var = float(values.var(unbiased=False).item())
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.var = (
            self.var * self.count
            + batch_var * batch_count
            + delta * delta * self.count * batch_count / total
        ) / total
        self.mean += delta * batch_count / total
        self.count = total

    @property
    def std(self) -> float:
        return math.sqrt(self.var + self.epsilon)


class PPORolloutSource:
    """Generates PPO training batches by running the current policy in the env.

    This is what makes ``ppo_bipedalwalker`` unlike every other model here: its
    training data is a function of its own weights, so there is no cached file
    and no train/validation/test split of one. Each call to ``collect`` runs
    ``n_steps`` of the live policy, computes GAE(lambda) advantages against the
    critic's own value estimates, and hands back a fresh ``DataLoader`` over
    ``(observation, action, old_log_prob, advantage, return)``.

    Why this replaced behaviour cloning. The previous setup cloned
    ``BipedalWalkerHeuristics``, a *stateful* gait controller that carries the
    swing leg and phase between steps. One observation therefore maps to
    different actions depending on hidden state a feedforward policy cannot
    observe, so the clone was fitting an ill-posed function: it reached -80
    against the heuristic's +90 and could not have closed that gap by training
    longer. Its reported metric was also not a return at all but the negated
    mean absolute error against the heuristic's action vector, which has no
    published counterpart.

    Rollout continuity. The environment is created once and its observation
    carried across calls, so iteration *k+1* resumes the episode iteration *k*
    ended mid-way through. Truncating an episode at the buffer boundary and
    bootstrapping its value there is standard PPO and is why ``truncated`` and
    ``terminated`` are handled differently below: a terminated state has value
    0 by definition, a truncated one does not.

    Statistics timing. ``obs_norm`` is folded forward with the *previous*
    iteration's observations, before this iteration's rollout begins, and then
    left alone. Updating it mid-iteration would mean the log-probabilities
    stored during collection were computed under different normalisation than
    the ones the surrogate objective recomputes, so the importance ratio would
    not start at 1 and the clipped objective would be measuring the wrong thing.
    """

    def __init__(
        self,
        env_id: str,
        *,
        n_steps: int,
        minibatch_size: int,
        gamma: float,
        gae_lambda: float,
        update_passes: int,
        seed: int = 0,
        package_hint: str = "gymnasium[box2d]",
    ) -> None:
        self.env_id = env_id
        self.n_steps = n_steps
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_passes = update_passes
        self.seed = seed
        self.package_hint = package_hint
        self._env: Any | None = None
        self._observation: Any | None = None
        self._episode_return: float = 0.0
        self._episode_length: int = 0
        self._pending_obs: Any | None = None
        self._return_norm = _RunningScalarNorm()
        self._discounted_return: float = 0.0
        self._iterations: int = 0

    # ------------------------------------------------------------ env setup --
    def _ensure_env(self) -> Any:
        if self._env is None:
            gymnasium = _require_dependency("gymnasium", self.package_hint)
            self._env = gymnasium.make(self.env_id)
            self._env.action_space.seed(self.seed)
            self._observation, _ = self._env.reset(seed=self.seed)
        return self._env

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    @property
    def action_low(self) -> Any:
        env = self._ensure_env()
        return env.action_space.low

    @property
    def action_high(self) -> Any:
        env = self._ensure_env()
        return env.action_space.high

    # ------------------------------------------------------------- rollout --
    def _fold_pending_observation_statistics(self, model: Any) -> None:
        normalizer = _find_obs_normalizer(model)
        if normalizer is None or self._pending_obs is None:
            return
        normalizer.update(self._pending_obs)
        self._pending_obs = None

    def _scaled_rewards(self, rewards: Any, dones: Any) -> Any:
        """Divide rewards by the running std of the discounted return."""
        discounted: list[float] = []
        for reward, done in zip(rewards.tolist(), dones.tolist()):
            self._discounted_return = self._discounted_return * self.gamma + reward
            discounted.append(self._discounted_return)
            if done:
                self._discounted_return = 0.0
        self._return_norm.update(torch.tensor(discounted, dtype=torch.float32))
        return (rewards / self._return_norm.std).clamp(
            -BIPEDAL_PPO_REWARD_CLIP, BIPEDAL_PPO_REWARD_CLIP
        )

    def _advantages_and_returns(
        self, rewards: Any, values: Any, dones: Any, last_value: float
    ) -> tuple[Any, Any]:
        """GAE(lambda), computed backwards over the buffer.

        ``dones`` marks *terminal* states only. A step truncated by the time
        limit, or one merely sitting at the buffer boundary, still has future
        reward, so its next-state value is bootstrapped rather than zeroed.
        """
        advantages = torch.zeros_like(rewards)
        running = 0.0
        next_value = last_value
        for index in range(len(rewards) - 1, -1, -1):
            non_terminal = 0.0 if bool(dones[index]) else 1.0
            delta = rewards[index] + self.gamma * next_value * non_terminal - values[index]
            running = delta + self.gamma * self.gae_lambda * non_terminal * running
            advantages[index] = running
            next_value = float(values[index])
        return advantages, advantages + values

    def collect(self, model: Any, device: Any) -> tuple[Any, dict[str, float]]:
        """Run one PPO iteration's worth of environment steps.

        Returns the training ``DataLoader`` and the summary statistics of any
        episodes that finished inside this rollout — the latter is what the
        training loop reports as the epoch's train-side episodic return.
        """
        env = self._ensure_env()
        self._fold_pending_observation_statistics(model)

        was_training = model.training
        model.eval()
        observations = torch.zeros(self.n_steps, env.observation_space.shape[0])
        actions = torch.zeros(self.n_steps, env.action_space.shape[0])
        log_probs = torch.zeros(self.n_steps)
        values = torch.zeros(self.n_steps)
        rewards = torch.zeros(self.n_steps)
        dones = torch.zeros(self.n_steps, dtype=torch.bool)
        completed: list[float] = []
        lengths: list[int] = []
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32)

        try:
            with torch.no_grad():
                for step in range(self.n_steps):
                    observation = torch.as_tensor(
                        self._observation, dtype=torch.float32
                    )
                    mean, log_std, value = model(observation.unsqueeze(0).to(device))
                    distribution = torch.distributions.Normal(
                        mean.squeeze(0).cpu(), log_std.squeeze(0).cpu().exp()
                    )
                    action = distribution.sample()
                    observations[step] = observation
                    actions[step] = action
                    # Log-probability of the *unclipped* sample: clipping happens
                    # only on the way into the environment, and pretending the
                    # clipped value was the sample would put a point mass at the
                    # bounds that this density does not have.
                    log_probs[step] = distribution.log_prob(action).sum()
                    values[step] = value.squeeze(0).cpu()

                    stepped = env.step(
                        torch.clamp(action, low, high).numpy()
                    )
                    self._observation, reward, terminated, truncated, _ = stepped
                    rewards[step] = float(reward)
                    dones[step] = bool(terminated)
                    self._episode_return += float(reward)
                    self._episode_length += 1
                    if terminated or truncated:
                        completed.append(self._episode_return)
                        lengths.append(self._episode_length)
                        self._episode_return = 0.0
                        self._episode_length = 0
                        self._observation, _ = env.reset()

                final = torch.as_tensor(self._observation, dtype=torch.float32)
                last_value = float(model(final.unsqueeze(0).to(device))[2].squeeze(0))
        finally:
            model.train(was_training)

        scaled_rewards = self._scaled_rewards(rewards, dones)
        advantages, returns = self._advantages_and_returns(
            scaled_rewards, values, dones, last_value
        )
        self._pending_obs = observations
        self._iterations += 1

        dataset = _TensorRowsDataset(
            observations, actions, log_probs, advantages, returns
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.minibatch_size,
            sampler=_RepeatedPermutationSampler(
                len(dataset), self.update_passes, seed=self.seed + self._iterations
            ),
            # The buffer is already a handful of in-memory tensors; a worker
            # process would only add a pickle round-trip per epoch.
            num_workers=0,
            pin_memory=False,
        )
        return loader, _rollout_statistics(completed, lengths, values, returns)


class _RepeatedPermutationSampler(torch.utils.data.Sampler[int]):
    """``update_passes`` independent shuffles of the buffer, back to back.

    PPO runs several passes over each rollout before discarding it, and the
    benchmark's epoch loop iterates a DataLoader exactly once per epoch. Rather
    than restructure the loop, the passes are folded into the sampler: one
    epoch's iteration yields K full permutations, so every sample is seen
    exactly K times and each pass is its own shuffle. Concatenating K copies of
    the dataset and shuffling globally would instead let one minibatch hold the
    same transition twice.
    """

    def __init__(self, length: int, repeats: int, seed: int) -> None:
        self.length = length
        self.repeats = repeats
        self.seed = seed

    # Iterator, not Iterable: this is a generator function, and Sampler.__iter__
    # is declared to return an Iterator — widening it here makes the sampler
    # unusable as a DataLoader `sampler=` argument.
    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.repeats):
            yield from torch.randperm(self.length, generator=generator).tolist()

    def __len__(self) -> int:
        return self.length * self.repeats


def _find_obs_normalizer(model: Any) -> Any | None:
    """Locate the model's ``RunningObsNorm``, wherever perforation left it.

    Attribute access would be enough today, but PerforatedAI rebuilds a model's
    module tree when it inserts dendrites, and the benchmark also hands around
    ``torch.compile`` wrappers. A type scan over ``modules()`` survives both.
    """
    for module in model.modules():
        if isinstance(module, RunningObsNorm):
            return module
    return None


def _rollout_statistics(
    completed: list[float], lengths: list[int], values: Any, returns: Any
) -> dict[str, float]:
    stats: dict[str, float] = {
        "rollout_episodes": float(len(completed)),
        # Explained variance of the critic over the buffer: 1.0 means the value
        # head predicts the discounted return exactly, <= 0 means it is no
        # better than predicting the mean. The single most useful number for
        # telling "PPO is learning slowly" apart from "the critic is broken".
        "rollout_explained_variance": _explained_variance(values, returns),
    }
    if completed:
        episode_returns = torch.tensor(completed, dtype=torch.float32)
        stats["episodic_return"] = float(episode_returns.mean())
        stats["rollout_return_min"] = float(episode_returns.min())
        stats["rollout_return_max"] = float(episode_returns.max())
        stats["rollout_episode_length"] = float(
            torch.tensor(lengths, dtype=torch.float32).mean()
        )
    return stats


def _explained_variance(predictions: Any, targets: Any) -> float:
    target_variance = float(targets.var(unbiased=False))
    if target_variance < 1e-12:
        return 0.0
    return 1.0 - float((targets - predictions).var(unbiased=False)) / target_variance


def _build_bipedalwalker(batch_size: int) -> TaskBundle:
    """BipedalWalker-v3 trained with PPO, on policy.

    The bundle's loaders are placeholders with the right shapes and length;
    ``train_loader`` is replaced at the top of every epoch by
    ``PPORolloutSource.collect``, and validation and test are episodic rollouts
    rather than dataset passes (see ``_rollout_evaluation`` in training.py).
    The placeholders exist because the pipeline reads ``len(train_loader)`` to
    size PerforatedAI's correlation window and runs one batch through the model
    to infer per-module output dimensions, both before any training starts.

    The metric is the mean episodic return, which *is* comparable to the
    published 300-point solved threshold — unlike the negated action MAE this
    model used to report, which measured agreement with a scripted gait
    controller and had no published counterpart. For reference points on the
    same environment: Gymnasium's own ``BipedalWalkerHeuristics`` scores about
    +90, and the behaviour-cloning setup this replaced reached about -80.
    """
    # Fail here rather than three layers into the first epoch if Box2D is absent.
    gymnasium = _require_dependency("gymnasium", "gymnasium[box2d]")
    probe = gymnasium.make("BipedalWalker-v3")
    obs_dim = int(probe.observation_space.shape[0])
    action_dim = int(probe.action_space.shape[0])
    probe.close()

    source = PPORolloutSource(
        "BipedalWalker-v3",
        n_steps=BIPEDAL_PPO_ROLLOUT_STEPS,
        minibatch_size=batch_size,
        gamma=BIPEDAL_PPO_GAMMA,
        gae_lambda=BIPEDAL_PPO_GAE_LAMBDA,
        update_passes=BIPEDAL_PPO_UPDATE_PASSES,
    )
    placeholder = _TensorRowsDataset(
        torch.zeros(BIPEDAL_PPO_ROLLOUT_STEPS, obs_dim),
        torch.zeros(BIPEDAL_PPO_ROLLOUT_STEPS, action_dim),
        torch.zeros(BIPEDAL_PPO_ROLLOUT_STEPS),
        torch.zeros(BIPEDAL_PPO_ROLLOUT_STEPS),
        torch.zeros(BIPEDAL_PPO_ROLLOUT_STEPS),
    )
    bundle = TaskBundle(
        torch.utils.data.DataLoader(
            placeholder,
            batch_size=batch_size,
            sampler=_RepeatedPermutationSampler(
                len(placeholder), BIPEDAL_PPO_UPDATE_PASSES, seed=0
            ),
            num_workers=0,
        ),
        # Never iterated: _eval_on_loader routes on-policy models to rollouts.
        # Kept non-empty so anything that probes a loader's length or first item
        # finds a correctly shaped batch instead of raising.
        torch.utils.data.DataLoader(placeholder, batch_size=batch_size, num_workers=0),
        torch.utils.data.DataLoader(placeholder, batch_size=batch_size, num_workers=0),
        "Episodic Return",
        "maximize",
        "BipedalWalker-v3 on-policy PPO rollouts "
        f"({BIPEDAL_PPO_ROLLOUT_STEPS} steps per iteration)",
    )
    bundle.on_policy = source
    return bundle


# Points sampled once per mesh and cached, and the number actually fed to the
# network. Caching a superset and drawing from it every epoch is the reference
# protocol: Qi et al. distribute `modelnet40_ply_hdf5_2048` and train on a
# 1024-point subsample of it.
MODELNET40_CACHE_POINTS: int = 2048
MODELNET40_POINTS: int = 1024
# Bumped whenever the sampling above changes in a way that makes an existing
# cache wrong; it is part of the cache filename, so a stale cache is rebuilt
# rather than silently reused.
MODELNET40_CACHE_VERSION: int = 1


def _read_off_mesh(path: Path) -> tuple[Any, Any]:
    """Parse an OFF file into ``(vertices [V, 3], faces [F, 3])``.

    Tokenised rather than read line by line because a good number of
    ModelNet40's files have the vertex/face counts glued onto the ``OFF``
    keyword (``OFF6 8 0``) instead of on the following line, and because faces
    are not all triangles — quads and larger polygons are fan-triangulated
    here so the sampler downstream only has to handle triangles.
    """
    tokens = path.read_text().split()
    if tokens[0] == "OFF":
        cursor = 1
    else:  # "OFF<v> <f> <e>" with no separator after the keyword
        tokens[0] = tokens[0][3:]
        cursor = 0
    vertex_count = int(tokens[cursor])
    face_count = int(tokens[cursor + 1])
    cursor += 3  # counts are vertices, faces, edges (edges is always 0 here)

    stop = cursor + 3 * vertex_count
    vertices = torch.tensor(
        [float(value) for value in tokens[cursor:stop]], dtype=torch.float32
    ).view(vertex_count, 3)
    cursor = stop

    faces: list[tuple[int, int, int]] = []
    for _ in range(face_count):
        sides = int(tokens[cursor])
        corners = [int(value) for value in tokens[cursor + 1 : cursor + 1 + sides]]
        cursor += 1 + sides
        for offset in range(1, sides - 1):
            faces.append((corners[0], corners[offset], corners[offset + 1]))
    return vertices, torch.tensor(faces, dtype=torch.long).view(-1, 3)


def _sample_mesh_surface(vertices: Any, faces: Any, count: int, seed: int) -> Any:
    """Draw ``count`` points uniformly over the *surface* of a triangle mesh.

    This is the substantive difference from reading vertices straight out of
    the file. ModelNet40 is CAD geometry, so vertex density tracks how the
    model was authored, not how big the surface is: a tabletop can be four
    vertices while a moulding detail is several thousand. Taking evenly spaced
    *vertex indices* — what this loader used to do — therefore samples the
    authoring order, and it has no way to describe a mesh with fewer vertices
    than the network's input width, which is 24% of ModelNet40 (8% have under
    256 vertices); those were padded by repeating the same handful of corner
    points until the tensor was full.

    Picking triangles with probability proportional to area and a uniform
    barycentric point inside each gives a genuine surface sample at any
    density, for every mesh. It is what Qi et al.'s released point clouds are.
    """
    generator = torch.Generator().manual_seed(seed)
    triangles = vertices[faces]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * torch.cross(edge_a, edge_b, dim=-1).norm(dim=-1)
    if not torch.isfinite(areas).all() or float(areas.sum()) <= 0.0:
        # Fully degenerate mesh (zero area, or NaN coordinates). Nothing to
        # sample a surface from, so fall back to the vertices.
        index = torch.randint(len(vertices), (count,), generator=generator)
        return vertices[index]

    chosen = torch.multinomial(areas, count, replacement=True, generator=generator)
    corner = triangles[chosen, 0]
    span_a = edge_a[chosen]
    span_b = edge_b[chosen]
    u = torch.rand(count, 1, generator=generator)
    v = torch.rand(count, 1, generator=generator)
    # Reflect the half of the unit square that falls outside the triangle back
    # into it, which keeps the result uniform.
    outside = (u + v) > 1.0
    u = torch.where(outside, 1.0 - u, u)
    v = torch.where(outside, 1.0 - v, v)
    return corner + u * span_a + v * span_b


def _normalize_unit_sphere(points: Any) -> Any:
    points = points - points.mean(dim=0, keepdim=True)
    return points / points.norm(dim=1).max().clamp_min(1e-6)


def _modelnet40_cache_path(root: Path, split: str) -> Path:
    """Where a split's sampled clouds live. Shared with ``dataset_exists``."""
    return (
        root
        / f"surface_points_v{MODELNET40_CACHE_VERSION}"
        f"_{split}_{MODELNET40_CACHE_POINTS}.pt"
    )


class _ModelNet40Dataset:
    """ModelNet40 as cached surface point clouds.

    Parsing the OFF meshes costs about 19 ms each, which over 9.8k training
    meshes is ~3 minutes of single-threaded text parsing *per epoch* — it was
    the dominant cost of this model's runs, ahead of the network itself. The
    sampled clouds are therefore written once to a tensor file and reused;
    ``__getitem__`` becomes an index into a resident tensor.

    Each entry holds ``MODELNET40_CACHE_POINTS`` points; callers take the
    ``MODELNET40_POINTS`` the network wants from it (randomly, per epoch, for
    training — see ``_ModelNet40TrainAugment``).
    """

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
        self.points, self.labels = self._load_or_build_cache(split)

    def _load_or_build_cache(self, split: str) -> tuple[Any, Any]:
        cache = _modelnet40_cache_path(self.root, split)
        if cache.exists():
            # weights_only=True: same rationale as the cache load above — this
            # file only ever holds a {"points", "labels"} tensor dict.
            payload = torch.load(cache, weights_only=True)
            if len(payload["points"]) == len(self.samples):
                return payload["points"], payload["labels"]
        print(
            f"[modelnet40] sampling {MODELNET40_CACHE_POINTS} surface points from "
            f"{len(self.samples)} {split} meshes (one time, then cached)"
        )
        clouds = torch.zeros(
            len(self.samples), MODELNET40_CACHE_POINTS, 3, dtype=torch.float32
        )
        labels = torch.zeros(len(self.samples), dtype=torch.long)
        for index, (path, label) in enumerate(self.samples):
            vertices, faces = _read_off_mesh(path)
            # Seeded by position so the cache is reproducible.
            sampled = _sample_mesh_surface(
                vertices, faces, MODELNET40_CACHE_POINTS, seed=index
            )
            clouds[index] = _normalize_unit_sphere(sampled)
            labels[index] = label
            if (index + 1) % 1000 == 0:
                print(f"[modelnet40]   {index + 1}/{len(self.samples)}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"points": clouds, "labels": labels}, cache)
        return clouds, labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return self.points[index], self.labels[index]


class _ModelNet40FixedView:
    """The first ``MODELNET40_POINTS`` of each cached cloud, unaugmented.

    Used for validation and test. The cached points are already in random
    surface order, so taking a prefix is an unbiased subsample and — unlike
    drawing a fresh one every call — gives the same input on every epoch, which
    is what an evaluation split has to do.
    """

    def __init__(self, base: Any, points: int = MODELNET40_POINTS):
        self.base = base
        self.points = points

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        cloud, label = self.base[index]
        return cloud[: self.points], label


class _ModelNet40TrainAugment:
    """Resample, rotate, and jitter — the training view of a cached cloud.

    Mirrors the reference PointNet implementation (Qi et al.,
    https://github.com/charlesq34/pointnet, provider.py ``rotate_point_cloud``
    + ``jitter_point_cloud``), plus the random 1024-of-2048 subsample its data
    pipeline does: a fresh subset, a random rotation about the up axis, and
    small clipped Gaussian jitter, every epoch.

    The rotation is about **Z**. ModelNet40's raw OFF meshes are Z-up — chairs,
    bottles, lamps, and people all have their largest extent on that axis — and
    the axis is consistent across the dataset, so an object's orientation is a
    usable cue and the evaluation splits are never rotated. Spinning about Y,
    as this wrapper previously did, tips objects onto their sides during
    training only, which discards that cue and trains on a pose distribution
    the model is never evaluated on. (The distributed HDF5 clouds are Y-up
    because the conversion rotated them; the OFF files here are not.)
    """

    def __init__(
        self,
        base: Any,
        points: int = MODELNET40_POINTS,
        jitter_std: float = 0.01,
        jitter_clip: float = 0.05,
    ):
        self.base = base
        self.points = points
        self.jitter_std = jitter_std
        self.jitter_clip = jitter_clip

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        cloud, label = self.base[index]
        choice = torch.randperm(cloud.shape[0])[: self.points]
        points = cloud[choice]
        angle = float(torch.rand(1).item()) * 2.0 * math.pi
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotation = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
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
    return _bundle_from_splits(
        _ModelNet40TrainAugment(train_ds),
        _ModelNet40FixedView(val_ds),
        _ModelNet40FixedView(_ModelNet40Dataset(train=False)),
        batch_size,
        "Accuracy",
        "maximize",
        f"ModelNet40 {MODELNET40_POINTS}-point mesh-surface samples",
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
    "gcn": 1,  # Transductive Cora — one full-graph step per epoch, as in Kipf
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
    # Was 24, on the theory that horizon-sized batches stabilise GRU training.
    # It bought nothing measurable and cost 1534 batches/epoch at 2.4 batch/s —
    # 4.2 hours, a fifth of the entire base sweep, for a 74k-parameter model.
    "gru_forecaster": 128,
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
        # No entry for ppo_bipedalwalker: it trains on policy, so it has nothing
        # to cache. Reporting it as "not cached" makes `dqb download_data` build
        # its bundle, which is a cheap Box2D availability check and exactly the
        # thing worth failing early on.
        "lstm_autoencoder":     [root / "mit-bih" / "100.dat"],
        # Both the extracted meshes and the sampled-cloud caches. Without the
        # caches listed, `dqb download-data` would report the dataset ready and
        # the first training run would still stall ~4 minutes building them.
        "pointnet_modelnet40":  [
            root / "modelnet40" / "raw",
            _modelnet40_cache_path(root / "modelnet40", "train"),
            _modelnet40_cache_path(root / "modelnet40", "test"),
        ],
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
