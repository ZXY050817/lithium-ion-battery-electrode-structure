#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import random
import site
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict, Union, cast

# Work around duplicated Intel OpenMP runtimes commonly seen in Anaconda + PyTorch on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires matplotlib. Install it with: pip install matplotlib"
    ) from exc

try:
    np = importlib.import_module("numpy")
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires numpy. Install it with: pip install numpy"
    ) from exc

NDArray = Any
TorchTensor = Any
TorchDevice = Any


def add_torch_dll_directory() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    candidate_dirs = []
    try:
        for site_dir in site.getsitepackages():
            candidate_dirs.append(Path(site_dir) / "torch" / "lib")
    except Exception:
        pass

    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = None
    if user_site:
        candidate_dirs.append(Path(user_site) / "torch" / "lib")

    for candidate in candidate_dirs:
        if candidate.exists():
            os.add_dll_directory(str(candidate))


add_torch_dll_directory()

try:
    torch = cast(Any, importlib.import_module("torch"))
    nn = cast(Any, importlib.import_module("torch.nn"))
    nn_functional = cast(Any, importlib.import_module("torch.nn.functional"))
    torch_utils_data = importlib.import_module("torch.utils.data")
    DataLoader = cast(Any, torch_utils_data.DataLoader)
    Dataset = cast(Any, torch_utils_data.Dataset)
except ImportError as exc:  # pragma: no cover
    torch = cast(Any, None)
    nn = cast(Any, None)
    nn_functional = cast(Any, None)
    DataLoader = cast(Any, None)
    Dataset = cast(Any, object)
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

AutoencoderBase: type = object
if nn is not None:
    AutoencoderBase = cast(type, nn.Module)
else:  # pragma: no cover
    AutoencoderBase = object


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = DEFAULT_PROJECT_ROOT / "Structure_Data"
DEFAULT_TRAIN_ROOT = DEFAULT_DATA_ROOT / "train"
DEFAULT_VALIDATION_ROOT = DEFAULT_DATA_ROOT / "verify"
DEFAULT_TEST_ROOT = DEFAULT_DATA_ROOT / "test"
DEFAULT_OUTPUT_ROOT = DEFAULT_PROJECT_ROOT / "CNNtrain" / "CNNOutputs"
EXPECTED_INPUT_SHAPE = (64, 64, 64, 3)
DEFAULT_VERIFY_ROOT = DEFAULT_DATA_ROOT / "verify"
CBD_CHANNEL_INDEX = 2
DEFAULT_DENOISING_MASK_RATIO = 0.10
DENOISING_EXAMPLE_COUNT = 3
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1.0e-8
LEARNING_RATE_DECAY_EPOCHS = 10
LEARNING_RATE_DECAY_FACTOR = 0.95
PHASE_DISPLAY_COLORS = np.asarray(
    [
        [0.20, 0.45, 0.85],  # electrolyte
        [0.95, 0.55, 0.15],  # active
        [0.10, 0.70, 0.25],  # cbd
    ],
    dtype=np.float32,
)
MASK_DISPLAY_COLOR = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
MODEL_ARCHITECTURE_NAME = "deep_residual_3d_autoencoder_multitask_v1"
DEFAULT_RECONSTRUCTION_LOSS_WEIGHT = 0.8
DEFAULT_FACTOR_LOSS_WEIGHT = 0.2
STRUCTURE_FACTOR_JSON_SUFFIX = "_structure_factors.json"
STRUCTURE_FACTOR_NAMES: Tuple[str, ...] = (
    "particle_volume_fraction",
    "cbd_volume_fraction",
    "specific_surface_area_inv_um",
    "electronic_connectivity",
    "electrolyte_connectivity",
    "equivalent_particle_diameter_um",
)


class PhaseChannelStats(TypedDict):
    channel_counts: List[int]
    cbd_vector_present: bool
    cbd_positive_loss_weight: float


class AverageLossBreakdown(TypedDict):
    total_loss: float
    reconstruction_loss: float
    factor_loss: float


def collect_phase_channel_files(
    root: Path,
    excluded_subdirs: Optional[Sequence[str]] = None,
) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    excluded = {name.lower() for name in excluded_subdirs or ()}
    file_paths: List[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_file() and child.name.endswith("_phase_channels.npy"):
            file_paths.append(child)
            continue
        if not child.is_dir():
            continue
        if child.name.lower() in excluded:
            continue
        file_paths.extend(sorted(child.glob("*_phase_channels.npy")))
    return file_paths


def infer_structure_factor_json_path(phase_channel_path: Path) -> Path:
    phase_channel_name = phase_channel_path.name
    if phase_channel_name.endswith("_phase_channels.npy"):
        factor_name = phase_channel_name.replace(
            "_phase_channels.npy",
            STRUCTURE_FACTOR_JSON_SUFFIX,
        )
        return phase_channel_path.with_name(factor_name)
    return phase_channel_path.with_suffix(".json")


def load_structure_factor_vector(
    factor_path: Path,
    factor_names: Sequence[str] = STRUCTURE_FACTOR_NAMES,
) -> NDArray:
    if not factor_path.exists():
        raise FileNotFoundError(
            "Structure-factor JSON was not found for the phase-channel sample: "
            f"{factor_path}"
        )
    try:
        with factor_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in structure-factor file: {factor_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Structure-factor JSON must contain an object: {factor_path}")

    values: List[float] = []
    for factor_name in factor_names:
        raw_value = payload.get(factor_name)
        if raw_value is None:
            raise ValueError(
                f"Missing structure-factor key '{factor_name}' in {factor_path}"
            )
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Structure-factor key '{factor_name}' must be numeric in {factor_path}"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"Structure-factor key '{factor_name}' must be finite in {factor_path}"
            )
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def compute_factor_normalization_stats(factor_values: NDArray) -> Tuple[NDArray, NDArray]:
    if factor_values.ndim != 2:
        raise ValueError("factor_values must be a 2D array.")
    factor_mean = factor_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    factor_std = factor_values.std(axis=0, dtype=np.float64).astype(np.float32)
    factor_std = np.where(factor_std < 1.0e-6, 1.0, factor_std).astype(np.float32)
    return factor_mean, factor_std


class PhaseChannelDataset(Dataset):
    def __init__(self, data_root: Path, max_files: Optional[int] = None) -> None:
        file_paths = collect_phase_channel_files(data_root)
        if max_files is not None:
            if max_files <= 0:
                raise ValueError("max_files must be positive when provided.")
            file_paths = file_paths[:max_files]
        if not file_paths:
            raise FileNotFoundError(
                "No '*_phase_channels.npy' files were found in the first-level "
                f"subdirectories of: {data_root}"
            )

        self.file_paths = file_paths
        self.factor_names = STRUCTURE_FACTOR_NAMES
        self.factor_paths = [
            infer_structure_factor_json_path(path) for path in self.file_paths
        ]
        self.factor_values = np.stack(
            [
                load_structure_factor_vector(factor_path, self.factor_names)
                for factor_path in self.factor_paths
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        self._validate_shape(self.file_paths[0])

    def _validate_shape(self, path: Path) -> None:
        array = np.load(path, mmap_mode="r")
        if tuple(array.shape) != EXPECTED_INPUT_SHAPE:
            raise ValueError(
                f"Expected input shape {EXPECTED_INPUT_SHAPE}, got {tuple(array.shape)} for {path}"
            )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Tuple[TorchTensor, TorchTensor]:
        path = self.file_paths[index]
        array = np.load(path, allow_pickle=False)
        if tuple(array.shape) != EXPECTED_INPUT_SHAPE:
            raise ValueError(
                f"Expected input shape {EXPECTED_INPUT_SHAPE}, got {tuple(array.shape)} for {path}"
            )
        tensor = torch.from_numpy(array.astype(np.float32, copy=False))
        factor_tensor = torch.from_numpy(self.factor_values[index])
        return tensor.permute(3, 0, 1, 2).contiguous(), factor_tensor


def inspect_dataset_phase_channels(file_paths: Sequence[Path]) -> PhaseChannelStats:
    channel_counts = np.zeros(EXPECTED_INPUT_SHAPE[3], dtype=np.int64)
    cbd_one_hot_vector = np.asarray([0, 0, 1], dtype=np.uint8)
    cbd_vector_present = False

    for path in file_paths:
        array = np.load(path, mmap_mode="r")
        if tuple(array.shape) != EXPECTED_INPUT_SHAPE:
            raise ValueError(
                f"Expected input shape {EXPECTED_INPUT_SHAPE}, got {tuple(array.shape)} for {path}"
            )
        flat = array.reshape(-1, EXPECTED_INPUT_SHAPE[3])
        channel_counts += flat.sum(axis=0, dtype=np.int64)
        if not cbd_vector_present:
            cbd_vector_present = bool(np.any(np.all(flat == cbd_one_hot_vector, axis=1)))

    total_voxels = int(channel_counts.sum())
    cbd_voxels = int(channel_counts[CBD_CHANNEL_INDEX])
    cbd_positive_loss_weight = 1.0
    if cbd_vector_present and cbd_voxels > 0 and total_voxels > cbd_voxels:
        cbd_positive_loss_weight = max(
            1.0, float(total_voxels - cbd_voxels) / float(cbd_voxels)
        )

    return {
        "channel_counts": [int(value) for value in channel_counts.tolist()],
        "cbd_vector_present": cbd_vector_present,
        "cbd_positive_loss_weight": float(cbd_positive_loss_weight),
    }


def compute_reconstruction_loss(
    reconstruction_logits: TorchTensor,
    inputs: TorchTensor,
    cbd_pos_weight_tensor: Optional[TorchTensor],
) -> TorchTensor:
    if nn_functional is None:  # pragma: no cover
        raise RuntimeError("PyTorch functional API is unavailable.")
    if cbd_pos_weight_tensor is None:
        return nn_functional.binary_cross_entropy_with_logits(
            reconstruction_logits, inputs
        )
    return nn_functional.binary_cross_entropy_with_logits(
        reconstruction_logits,
        inputs,
        pos_weight=cbd_pos_weight_tensor,
    )


def compute_factor_loss(
    factor_predictions: TorchTensor,
    normalized_factor_targets: TorchTensor,
) -> TorchTensor:
    if nn_functional is None:  # pragma: no cover
        raise RuntimeError("PyTorch functional API is unavailable.")
    return nn_functional.smooth_l1_loss(
        factor_predictions,
        normalized_factor_targets,
    )


def normalize_factor_targets(
    factor_targets: TorchTensor,
    factor_mean_tensor: TorchTensor,
    factor_std_tensor: TorchTensor,
) -> TorchTensor:
    return (factor_targets - factor_mean_tensor) / factor_std_tensor


def combine_multitask_losses(
    reconstruction_loss: TorchTensor,
    factor_loss: TorchTensor,
    reconstruction_loss_weight: float,
    factor_loss_weight: float,
) -> TorchTensor:
    return (
        reconstruction_loss_weight * reconstruction_loss
        + factor_loss_weight * factor_loss
    )


def apply_random_voxel_denoising_mask(
    inputs: TorchTensor,
    mask_ratio: float,
    generator: Optional[Any] = None,
) -> TorchTensor:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required to apply denoising masks.")
    if mask_ratio <= 0.0:
        return inputs
    voxel_mask = (
        torch.rand(
            (inputs.shape[0], 1, inputs.shape[2], inputs.shape[3], inputs.shape[4]),
            device=inputs.device,
            generator=generator,
        )
        < mask_ratio
    )
    return inputs * (~voxel_mask).to(inputs.dtype)


class ResidualConvBlock3D(AutoencoderBase):
    def __init__(self, channels: int) -> None:
        if nn is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required to construct the model.")
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: TorchTensor) -> TorchTensor:
        return self.activation(inputs + self.layers(inputs))


def downsample_block_3d(in_channels: int, out_channels: int) -> Any:
    if nn is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required to construct the model.")
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        ResidualConvBlock3D(out_channels),
    )


def upsample_block_3d(in_channels: int, out_channels: int) -> Any:
    if nn is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required to construct the model.")
    return nn.Sequential(
        nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        ResidualConvBlock3D(out_channels),
    )


class StructureAutoencoder(AutoencoderBase):
    def __init__(
        self,
        latent_dim: int = 128,
        factor_output_dim: int = len(STRUCTURE_FACTOR_NAMES),
    ) -> None:
        if nn is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required to construct the model.")
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if factor_output_dim <= 0:
            raise ValueError("factor_output_dim must be positive.")
        self.latent_dim = latent_dim
        self.factor_output_dim = factor_output_dim
        self.architecture_name = MODEL_ARCHITECTURE_NAME
        encoder_channels = (32, 64, 128, 192, 256)
        decoder_channels = (192, 128, 64, 32, 16)
        bottleneck_channels = encoder_channels[-1]

        self.encoder_cnn = nn.Sequential(
            downsample_block_3d(3, encoder_channels[0]),
            downsample_block_3d(encoder_channels[0], encoder_channels[1]),
            downsample_block_3d(encoder_channels[1], encoder_channels[2]),
            downsample_block_3d(encoder_channels[2], encoder_channels[3]),
            downsample_block_3d(encoder_channels[3], encoder_channels[4]),
        )
        self.encoder_fc = nn.Linear(bottleneck_channels * 2 * 2 * 2, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, bottleneck_channels * 2 * 2 * 2)
        self.decoder_cnn = nn.Sequential(
            upsample_block_3d(bottleneck_channels, decoder_channels[0]),
            upsample_block_3d(decoder_channels[0], decoder_channels[1]),
            upsample_block_3d(decoder_channels[1], decoder_channels[2]),
            upsample_block_3d(decoder_channels[2], decoder_channels[3]),
            upsample_block_3d(decoder_channels[3], decoder_channels[4]),
            nn.Conv3d(decoder_channels[4], 3, kernel_size=3, stride=1, padding=1),
        )
        self.factor_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, factor_output_dim),
        )

    def encode(self, inputs: TorchTensor) -> TorchTensor:
        encoded = self.encoder_cnn(inputs)
        return self.encoder_fc(encoded.flatten(start_dim=1))

    def decode(self, latent: TorchTensor) -> TorchTensor:
        decoded = self.decoder_fc(latent).view(-1, 256, 2, 2, 2)
        return self.decoder_cnn(decoded)

    def predict_structure_factors(self, latent: TorchTensor) -> TorchTensor:
        return self.factor_head(latent)

    def forward(
        self,
        inputs: TorchTensor,
    ) -> Tuple[TorchTensor, TorchTensor, TorchTensor]:
        latent = self.encode(inputs)
        reconstruction_logits = self.decode(latent)
        factor_predictions = self.predict_structure_factors(latent)
        return reconstruction_logits, factor_predictions, latent


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a multitask 3D convolutional autoencoder on 64x64x64x3 "
            "phase-channel structures together with sidecar structure-factor JSON files "
            f"saved under {DEFAULT_DATA_ROOT}."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("train", "resume", "verify"),
        help="Run mode: train a new model, resume an existing one, or reconstruct the verify set.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_TRAIN_ROOT,
        help=(
            "Root directory containing training structures. Each "
            "'*_phase_channels.npy' sample must have a matching "
            f"'{STRUCTURE_FACTOR_JSON_SUFFIX}' file. Default: {DEFAULT_TRAIN_ROOT}"
        ),
    )
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=DEFAULT_VALIDATION_ROOT,
        help=(
            "Root directory containing validation structures used for early stopping "
            "and best-model selection. Validation samples must use the same JSON sidecar "
            f"format. Default: {DEFAULT_VALIDATION_ROOT}"
        ),
    )
    parser.add_argument(
        "--varify-root",
        dest="validation_root",
        type=Path,
        help="Alias of --validation-root (compatibility with 'varify' naming).",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=DEFAULT_TEST_ROOT,
        help=(
            "Root directory containing held-out test structures. The test set is "
            "evaluated only after the best model is selected. Test samples must use "
            f"the same JSON sidecar format. Default: {DEFAULT_TEST_ROOT}"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for training outputs. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs. Default: 20",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Mini-batch size for training. Default: 2",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-3,
        help="Learning rate for Adam. Default: 1e-3",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Latent vector dimension. Default: 128",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=50,
        help="Print averaged total/reconstruction/factor losses after this many structures. Default: 50",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader workers. Default: 0",
    )
    parser.add_argument(
        "--denoising-mask-ratio",
        type=float,
        default=DEFAULT_DENOISING_MASK_RATIO,
        help=(
            "Fraction of voxel positions randomly zeroed to (0,0,0) during training, "
            "while BCE still targets the original clean structure. Default: 0.10"
        ),
    )
    parser.add_argument(
        "--reconstruction-loss-weight",
        type=float,
        default=DEFAULT_RECONSTRUCTION_LOSS_WEIGHT,
        help=(
            "Relative weight for the voxel-structure reconstruction loss. "
            f"Default: {DEFAULT_RECONSTRUCTION_LOSS_WEIGHT}"
        ),
    )
    parser.add_argument(
        "--factor-loss-weight",
        type=float,
        default=DEFAULT_FACTOR_LOSS_WEIGHT,
        help=(
            "Relative weight for the structure-factor regression loss. "
            f"Default: {DEFAULT_FACTOR_LOSS_WEIGHT}"
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Optional limit on the number of NPY files for debugging.",
    )
    parser.add_argument(
        "--validation-max-files",
        type=int,
        help="Optional limit on the number of validation NPY files for debugging.",
    )
    parser.add_argument(
        "--test-max-files",
        type=int,
        help="Optional limit on the number of test NPY files for debugging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260409,
        help="Random seed for training reproducibility. Default: 20260409",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training even if CUDA is available.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "Optional checkpoint path to resume from. When provided, --epochs "
            "means additional epochs to train on top of the saved state."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint path used for verify-set reconstruction export mode.",
    )
    parser.add_argument(
        "--verify-root",
        type=Path,
        default=DEFAULT_VERIFY_ROOT,
        help=f"Root directory for verify structures. Default: {DEFAULT_VERIFY_ROOT}",
    )
    parser.add_argument(
        "--reconstruction-output-dir",
        type=Path,
        help=(
            "Optional output directory for verify-set TXT exports. "
            "Default: <checkpoint_dir>\\verify_txt_reconstructions"
        ),
    )
    return parser.parse_args(argv)


MODE_PROMPT_OPTIONS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("train", "新训练 / New training", ("新训练", "训练", "new", "new training")),
    ("resume", "续训 / Resume training", ("续训", "继续训练", "恢复训练", "resume training")),
    ("verify", "验证 / Verify reconstruction", ("验证", "验证集", "重建验证", "validation")),
)


def prompt_choice(
    prompt: str,
    choices: Sequence[Tuple[str, str, Sequence[str]]],
    default: str,
) -> str:
    alias_to_value: Dict[str, str] = {}
    default_label = default
    display_items: List[str] = []
    for value, label, aliases in choices:
        display_items.append(f"{value} ({label})")
        if value == default:
            default_label = f"{value} ({label})"
        for alias in (value, *aliases):
            alias_to_value[alias.strip().lower()] = value

    choice_text = " | ".join(display_items)
    while True:
        raw = input(f"{prompt} [{default_label}]: ").strip().lower()
        if not raw:
            return default
        mapped = alias_to_value.get(raw)
        if mapped is not None:
            return mapped
        print(f"请输入以下选项之一 / Please choose one of: {choice_text}")


def prompt_text(prompt: str, default: str = "") -> str:
    if default:
        raw = input(f"{prompt} [{default}]: ").strip()
        return raw if raw else default
    return input(f"{prompt}: ").strip()


def prompt_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return int(default)
        try:
            return int(raw)
        except ValueError:
            print("请输入有效整数。 / Please enter a valid integer.")


def prompt_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("请输入有效数值。 / Please enter a valid number.")


def prompt_optional_int(prompt: str, default: Optional[int]) -> Optional[int]:
    default_text = "" if default is None else str(default)
    while True:
        if default is None:
            raw = input(f"{prompt} [留空表示全部 / Blank for all]: ").strip()
        else:
            raw = input(
                f"{prompt} [{default_text}, 留空表示全部 / Blank for all]: "
            ).strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("请输入有效整数。 / Please enter a valid integer.")
            continue
        if value <= 0:
            print("请输入正整数。 / Please enter a positive integer.")
            continue
        return value


def prompt_bool(prompt: str, default: bool) -> bool:
    default_text = "是 / yes" if default else "否 / no"
    true_values = {"y", "yes", "1", "true", "是", "好", "对"}
    false_values = {"n", "no", "0", "false", "否", "不", "不要"}
    while True:
        raw = input(
            f"{prompt} [是/yes(y) | 否/no(n), 默认={default_text} / default={default_text}]: "
        ).strip().lower()
        if not raw:
            return default
        if raw in true_values:
            return True
        if raw in false_values:
            return False
        print("请输入 是/否 或 y/n。 / Please enter yes/no or y/n.")


def prompt_mode_choice(default: str = "train") -> str:
    menu_items: Tuple[Tuple[str, str, str], ...] = (
        ("1", "train", "新训练 / New training"),
        ("2", "resume", "续训 / Resume training"),
        ("3", "verify", "验证 / Verify reconstruction"),
    )
    alias_to_value: Dict[str, str] = {}
    default_number = "1"
    print("可选运行模式 / Available run modes:")
    for number, value, label in menu_items:
        print(f"  {number}. {value} - {label}")
        alias_to_value[number] = value
        alias_to_value[value] = value
        if value == default:
            default_number = number
    while True:
        raw = input(
            f"请输入选项编号或名称 / Enter option number or name [{default_number}/{default}]: "
        ).strip().lower()
        if not raw:
            return default
        mapped = alias_to_value.get(raw)
        if mapped is not None:
            return mapped
        print("请输入 1/2/3 或 train/resume/verify。 / Please enter 1/2/3 or train/resume/verify.")


def configure_train_interactively(args: argparse.Namespace) -> argparse.Namespace:
    print("进入新训练参数设置。 / Configure parameters for a new training run.")
    args.data_root = Path(
        prompt_text(
            "训练集目录 / Training data root",
            str(args.data_root),
        )
    )
    args.validation_root = Path(
        prompt_text(
            "验证集目录 / Validation data root",
            str(args.validation_root),
        )
    )
    args.test_root = Path(
        prompt_text(
            "测试集目录 / Test data root",
            str(args.test_root),
        )
    )
    args.output_root = Path(
        prompt_text(
            "训练输出目录 / Training output root",
            str(args.output_root),
        )
    )
    args.epochs = prompt_int("训练轮数 / Epochs", int(args.epochs))
    args.batch_size = prompt_int("批大小 / Batch size", int(args.batch_size))
    args.learning_rate = prompt_float(
        "学习率 / Learning rate", float(args.learning_rate)
    )
    args.denoising_mask_ratio = prompt_float(
        "去噪遮挡比例 / Denoising mask ratio",
        float(args.denoising_mask_ratio),
    )
    args.reconstruction_loss_weight = prompt_float(
        "重构损失权重 / Reconstruction loss weight",
        float(args.reconstruction_loss_weight),
    )
    args.factor_loss_weight = prompt_float(
        "结构参量损失权重 / Structure-factor loss weight",
        float(args.factor_loss_weight),
    )
    args.latent_dim = prompt_int("隐向量维度 / Latent dimension", int(args.latent_dim))
    args.report_every = prompt_int(
        "误差输出间隔（结构数） / Report every how many structures",
        int(args.report_every),
    )
    args.num_workers = prompt_int(
        "DataLoader 进程数 / DataLoader workers", int(args.num_workers)
    )
    args.max_files = prompt_optional_int(
        "最多读取多少个结构 / Max files to read",
        args.max_files,
    )
    args.seed = prompt_int("随机种子 / Random seed", int(args.seed))
    args.cpu = prompt_bool("是否强制使用 CPU / Force CPU only", bool(args.cpu))
    args.validation_max_files = prompt_optional_int(
        "最多读取多少个验证结构 / Max validation files to read",
        args.validation_max_files,
    )
    args.test_max_files = prompt_optional_int(
        "最多读取多少个测试结构 / Max test files to read",
        args.test_max_files,
    )
    return args


def infer_data_root_for_resume(
    checkpoint_path: Path, fallback: Path
) -> Path:
    summary_path = checkpoint_path.parent / "training_summary.json"
    if not summary_path.exists():
        return fallback
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except Exception:
        return fallback
    if not isinstance(summary, dict):
        return fallback
    data_root_value = summary.get("data_root")
    if isinstance(data_root_value, str) and data_root_value.strip():
        return Path(data_root_value)
    return fallback


def infer_validation_root_for_resume(
    checkpoint_path: Path, fallback: Path
) -> Path:
    summary_path = checkpoint_path.parent / "training_summary.json"
    if not summary_path.exists():
        return fallback
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except Exception:
        return fallback
    if not isinstance(summary, dict):
        return fallback
    validation_root_value = summary.get("validation_root")
    if isinstance(validation_root_value, str) and validation_root_value.strip():
        return Path(validation_root_value)
    return fallback


def infer_test_root_for_resume(
    checkpoint_path: Path, fallback: Path
) -> Path:
    summary_path = checkpoint_path.parent / "training_summary.json"
    if not summary_path.exists():
        return fallback
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except Exception:
        return fallback
    if not isinstance(summary, dict):
        return fallback
    test_root_value = summary.get("test_root")
    if isinstance(test_root_value, str) and test_root_value.strip():
        return Path(test_root_value)
    return fallback


def configure_resume_interactively(args: argparse.Namespace) -> argparse.Namespace:
    print("进入续训参数设置。 / Configure parameters for resume training.")
    resume_text = prompt_text(
        "请输入续训 checkpoint 路径 / Enter checkpoint path for resume",
        str(args.resume_from) if args.resume_from is not None else "",
    )
    if not resume_text:
        raise ValueError(
            "续训模式必须提供 checkpoint 路径。 / A checkpoint path is required for resume mode."
        )
    args.resume_from = Path(resume_text)

    suggested_data_root = infer_data_root_for_resume(args.resume_from, args.data_root)
    args.data_root = Path(
        prompt_text(
            "续训数据目录 / Resume data root",
            str(suggested_data_root),
        )
    )
    args.validation_root = Path(
        prompt_text(
            "验证集目录 / Validation data root",
            str(infer_validation_root_for_resume(args.resume_from, args.validation_root)),
        )
    )
    args.test_root = Path(
        prompt_text(
            "测试集目录 / Test data root",
            str(infer_test_root_for_resume(args.resume_from, args.test_root)),
        )
    )
    args.epochs = prompt_int(
        "额外训练轮数 / Additional epochs to train",
        int(args.epochs),
    )
    args.batch_size = prompt_int("批大小 / Batch size", int(args.batch_size))
    args.learning_rate = prompt_float(
        "学习率 / Learning rate", float(args.learning_rate)
    )
    args.denoising_mask_ratio = prompt_float(
        "去噪遮挡比例 / Denoising mask ratio",
        float(args.denoising_mask_ratio),
    )
    args.reconstruction_loss_weight = prompt_float(
        "重构损失权重 / Reconstruction loss weight",
        float(args.reconstruction_loss_weight),
    )
    args.factor_loss_weight = prompt_float(
        "结构参量损失权重 / Structure-factor loss weight",
        float(args.factor_loss_weight),
    )
    args.report_every = prompt_int(
        "误差输出间隔（结构数） / Report every how many structures",
        int(args.report_every),
    )
    args.num_workers = prompt_int(
        "DataLoader 进程数 / DataLoader workers", int(args.num_workers)
    )
    args.max_files = prompt_optional_int(
        "最多读取多少个结构 / Max files to read",
        args.max_files,
    )
    args.seed = prompt_int("随机种子 / Random seed", int(args.seed))
    args.cpu = prompt_bool("是否强制使用 CPU / Force CPU only", bool(args.cpu))
    args.validation_max_files = prompt_optional_int(
        "最多读取多少个验证结构 / Max validation files to read",
        args.validation_max_files,
    )
    args.test_max_files = prompt_optional_int(
        "最多读取多少个测试结构 / Max test files to read",
        args.test_max_files,
    )
    return args


def configure_verify_interactively(args: argparse.Namespace) -> argparse.Namespace:
    print("进入验证参数设置。 / Configure parameters for verification.")
    checkpoint_text = prompt_text(
        "请输入验证所用 checkpoint 路径 / Enter checkpoint path for verify",
        str(args.checkpoint) if args.checkpoint is not None else "",
    )
    if not checkpoint_text:
        raise ValueError(
            "验证模式必须提供 checkpoint 路径。 / A checkpoint path is required for verify mode."
        )
    args.checkpoint = Path(checkpoint_text)
    args.verify_root = Path(
        prompt_text("验证集目录 / Verify data root", str(args.verify_root))
    )
    output_default = (
        str(args.reconstruction_output_dir)
        if args.reconstruction_output_dir is not None
        else str(args.checkpoint.parent / "verify_txt_reconstructions")
    )
    output_text = prompt_text(
        "验证结果输出目录 / Verify reconstruction output directory",
        output_default,
    )
    args.reconstruction_output_dir = Path(output_text) if output_text else None
    args.cpu = prompt_bool("是否强制使用 CPU / Force CPU only", bool(args.cpu))
    return args


def configure_mode_interactively(args: argparse.Namespace) -> argparse.Namespace:
    interactive_mode_selection = args.mode is None
    if interactive_mode_selection:
        print("请选择运行模式。 / Please choose the run mode.")
        args.mode = prompt_mode_choice("train")

    if args.mode == "train" and interactive_mode_selection:
        args = configure_train_interactively(args)

    if args.mode == "resume" and interactive_mode_selection:
        args = configure_resume_interactively(args)

    if args.mode == "verify" and interactive_mode_selection:
        args = configure_verify_interactively(args)
    elif args.mode == "verify" and args.checkpoint is None:
        checkpoint_text = prompt_text(
            "请输入验证所用 checkpoint 路径 / Enter checkpoint path for verify"
        )
        if not checkpoint_text:
            raise ValueError(
                "验证模式必须提供 checkpoint 路径。 / A checkpoint path is required for verify mode."
            )
        args.checkpoint = Path(checkpoint_text)

    return args


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_output_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    date_prefix = datetime.now().strftime("%Y%m%d")
    index = 1
    while True:
        candidate = output_root / f"CNNtrain_{date_prefix}_{index:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        index += 1


def load_checkpoint(path: Path, device: TorchDevice) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary or model-state mapping.")
    return cast(Dict[str, Any], checkpoint)


def format_elapsed_time(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def coerce_float(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def normalize_loss_weights(
    reconstruction_loss_weight: float,
    factor_loss_weight: float,
) -> Tuple[float, float]:
    if reconstruction_loss_weight < 0.0:
        raise ValueError("reconstruction_loss_weight must be non-negative.")
    if factor_loss_weight < 0.0:
        raise ValueError("factor_loss_weight must be non-negative.")
    weight_sum = reconstruction_loss_weight + factor_loss_weight
    if weight_sum <= 0.0:
        raise ValueError("At least one multitask loss weight must be positive.")
    return (
        reconstruction_loss_weight / weight_sum,
        factor_loss_weight / weight_sum,
    )


def compute_scheduled_learning_rate(
    base_learning_rate: float, completed_epoch_count: int
) -> float:
    decay_steps = max(0, int(completed_epoch_count) // LEARNING_RATE_DECAY_EPOCHS)
    return float(base_learning_rate) * (LEARNING_RATE_DECAY_FACTOR ** decay_steps)


def set_optimizer_learning_rate(optimizer: Any, learning_rate: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(learning_rate)


def trim_history_to_epoch(
    history: List[Dict[str, Union[float, int]]], max_epoch: int
) -> List[Dict[str, Union[float, int]]]:
    trimmed: List[Dict[str, Union[float, int]]] = []
    for row in history:
        try:
            row_epoch = int(row.get("epoch", 0))
        except (TypeError, ValueError):
            row_epoch = 0
        if row_epoch <= max_epoch:
            trimmed.append(dict(row))
    return trimmed


def evaluate_average_multitask_losses(
    model: StructureAutoencoder,
    data_loader: Any,
    device: TorchDevice,
    cbd_pos_weight_tensor: Optional[TorchTensor],
    factor_mean_tensor: TorchTensor,
    factor_std_tensor: TorchTensor,
    reconstruction_loss_weight: float,
    factor_loss_weight: float,
) -> AverageLossBreakdown:
    model.eval()
    total_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    factor_loss_sum = 0.0
    total_structures = 0
    with torch.no_grad():
        for inputs, factor_targets in data_loader:
            inputs = inputs.to(device, non_blocking=(device.type == "cuda"))
            factor_targets = factor_targets.to(
                device,
                non_blocking=(device.type == "cuda"),
            )
            reconstruction_logits, factor_predictions, _ = model(inputs)
            reconstruction_loss = compute_reconstruction_loss(
                reconstruction_logits,
                inputs,
                cbd_pos_weight_tensor,
            )
            normalized_factor_targets = normalize_factor_targets(
                factor_targets,
                factor_mean_tensor,
                factor_std_tensor,
            )
            factor_loss = compute_factor_loss(
                factor_predictions,
                normalized_factor_targets,
            )
            total_loss = combine_multitask_losses(
                reconstruction_loss,
                factor_loss,
                reconstruction_loss_weight,
                factor_loss_weight,
            )
            batch_size = int(inputs.shape[0])
            total_loss_sum += float(total_loss.item()) * batch_size
            reconstruction_loss_sum += float(reconstruction_loss.item()) * batch_size
            factor_loss_sum += float(factor_loss.item()) * batch_size
            total_structures += batch_size

    denominator = max(1, total_structures)
    return {
        "total_loss": total_loss_sum / denominator,
        "reconstruction_loss": reconstruction_loss_sum / denominator,
        "factor_loss": factor_loss_sum / denominator,
    }


def select_device(force_cpu: bool) -> TorchDevice:
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_loss_history_csv(
    history: List[Dict[str, Union[float, int]]], path: Path
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "structures_seen",
                "average_total_loss",
                "average_reconstruction_loss",
                "average_factor_loss",
            ],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def plot_loss_history(
    history: List[Dict[str, Union[float, int]]], path: Path
) -> None:
    if not history:
        return

    x_values = [int(row["structures_seen"]) for row in history]
    y_values = [float(row["average_total_loss"]) for row in history]

    plt.figure(figsize=(8, 4.5))
    plt.plot(x_values, y_values, marker="o", linewidth=1.5)
    plt.xlabel("Structures Seen")
    plt.ylabel("Average Total Loss")
    plt.title("Training Total Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_epoch_history_csv(
    history: List[Dict[str, Union[float, int]]], path: Path
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "average_training_total_loss",
                "average_training_reconstruction_loss",
                "average_training_factor_loss",
                "average_validation_total_loss",
                "average_validation_reconstruction_loss",
                "average_validation_factor_loss",
            ],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def plot_epoch_history(
    history: List[Dict[str, Union[float, int]]], path: Path
) -> None:
    if not history:
        return

    epochs = [int(row["epoch"]) for row in history]
    training_values = [float(row["average_training_total_loss"]) for row in history]
    validation_values = [float(row["average_validation_total_loss"]) for row in history]

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, training_values, marker="o", linewidth=1.5, label="train")
    if any(np.isfinite(value) for value in validation_values):
        plt.plot(
            epochs,
            validation_values,
            marker="s",
            linewidth=1.5,
            label="validation",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Average Total Loss")
    plt.title("Epoch Total Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_phase_slices_txt(phase_labels: NDArray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for z_index in range(int(phase_labels.shape[2])):
            slice_xy = phase_labels[:, :, z_index]
            for x_index in range(int(slice_xy.shape[0])):
                row = " ".join(str(int(value)) for value in slice_xy[x_index, :])
                handle.write(row)
                handle.write("\n")
            if z_index + 1 < int(phase_labels.shape[2]):
                handle.write("\n")


def phase_channels_to_labels(phase_channels: NDArray) -> NDArray:
    if tuple(phase_channels.shape) != EXPECTED_INPUT_SHAPE:
        raise ValueError(
            f"Expected phase-channel shape {EXPECTED_INPUT_SHAPE}, got {tuple(phase_channels.shape)}"
        )
    return np.argmax(phase_channels, axis=3).astype(np.uint8)


def phase_channels_to_display_rgb(phase_channels: NDArray) -> NDArray:
    if tuple(phase_channels.shape) != EXPECTED_INPUT_SHAPE:
        raise ValueError(
            f"Expected phase-channel shape {EXPECTED_INPUT_SHAPE}, got {tuple(phase_channels.shape)}"
        )
    labels = np.argmax(phase_channels, axis=3)
    rgb = PHASE_DISPLAY_COLORS[labels].astype(np.float32, copy=True)
    zero_mask = np.all(phase_channels <= 0.0, axis=3)
    rgb[zero_mask] = MASK_DISPLAY_COLOR
    return rgb


def save_denoising_example_visualizations(
    file_paths: Sequence[Path],
    output_dir: Path,
    mask_ratio: float,
    seed: int,
    sample_count: int = DENOISING_EXAMPLE_COUNT,
) -> Optional[Path]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required to generate denoising examples.")
    if sample_count <= 0 or not file_paths:
        return None

    example_dir = output_dir / "denoising_mask_examples"
    example_dir.mkdir(parents=True, exist_ok=True)

    for sample_index, path in enumerate(file_paths[:sample_count], start=1):
        phase_channels = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        input_tensor = torch.from_numpy(phase_channels).permute(3, 0, 1, 2).unsqueeze(0)
        local_generator = torch.Generator(device="cpu")
        local_generator.manual_seed(int(seed) + sample_index)
        masked_tensor = apply_random_voxel_denoising_mask(
            input_tensor,
            mask_ratio,
            generator=local_generator,
        )
        masked_channels = (
            masked_tensor.squeeze(0).permute(1, 2, 3, 0).cpu().numpy().astype(np.float32)
        )

        original_rgb = phase_channels_to_display_rgb(phase_channels)
        masked_rgb = phase_channels_to_display_rgb(masked_channels)
        mask_rgb = np.ones_like(original_rgb, dtype=np.float32)
        zero_mask = np.all(masked_channels <= 0.0, axis=3)
        mask_rgb[~zero_mask] = 0.0

        mid_x = original_rgb.shape[0] // 2
        mid_y = original_rgb.shape[1] // 2
        mid_z = original_rgb.shape[2] // 2
        slice_specs = (
            (
                f"XY slice (z={mid_z})",
                np.swapaxes(original_rgb[:, :, mid_z, :], 0, 1),
                np.swapaxes(masked_rgb[:, :, mid_z, :], 0, 1),
                np.swapaxes(mask_rgb[:, :, mid_z, :], 0, 1),
            ),
            (
                f"XZ slice (y={mid_y})",
                np.swapaxes(original_rgb[:, mid_y, :, :], 0, 1),
                np.swapaxes(masked_rgb[:, mid_y, :, :], 0, 1),
                np.swapaxes(mask_rgb[:, mid_y, :, :], 0, 1),
            ),
            (
                f"YZ slice (x={mid_x})",
                np.swapaxes(original_rgb[mid_x, :, :, :], 0, 1),
                np.swapaxes(masked_rgb[mid_x, :, :, :], 0, 1),
                np.swapaxes(mask_rgb[mid_x, :, :, :], 0, 1),
            ),
        )

        figure, axes = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
        column_titles = (
            "Original clean structure",
            "Masked input to model",
            "Masked voxel locations",
        )
        for column_index, title in enumerate(column_titles):
            axes[0, column_index].set_title(title, fontsize=11)
        for row_index, (row_title, original_slice, masked_slice, mask_slice) in enumerate(slice_specs):
            axes[row_index, 0].set_ylabel(row_title, fontsize=10)
            for axis, image in zip(
                axes[row_index],
                (original_slice, masked_slice, mask_slice),
            ):
                axis.imshow(image, origin="lower", interpolation="nearest")
                axis.set_xticks([])
                axis.set_yticks([])
        figure.suptitle(
            f"{path.stem}\nDenoising mask ratio = {mask_ratio:.2%}",
            fontsize=12,
        )
        figure.savefig(
            example_dir / f"{sample_index:02d}_{path.stem}_denoising_example.png",
            dpi=180,
        )
        plt.close(figure)

    return example_dir


def infer_verify_run_tag(checkpoint_path: Path) -> str:
    parent_name = checkpoint_path.parent.name
    if parent_name.startswith("CNNtrain_"):
        return parent_name[len("CNNtrain_") :]
    return parent_name


def reconstruct_verify_set(args: argparse.Namespace) -> None:
    if TORCH_IMPORT_ERROR is not None:
        raise SystemExit(
            "This script requires PyTorch, but it is not installed in the current Python environment.\n"
            "Install it first, for example:\n"
            "  pip install torch\n"
            f"Original import error: {TORCH_IMPORT_ERROR}"
        )
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required when --reconstruct-verify is used.")

    device = select_device(args.cpu)
    checkpoint = load_checkpoint(args.checkpoint, device)
    latent_dim = int(checkpoint.get("latent_dim", args.latent_dim))
    checkpoint_factor_names_raw = checkpoint.get("factor_names")
    if not isinstance(checkpoint_factor_names_raw, list) or not checkpoint_factor_names_raw:
        raise ValueError("Checkpoint is missing required 'factor_names'.")
    checkpoint_factor_names = tuple(str(name) for name in checkpoint_factor_names_raw)
    model_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise ValueError("The provided checkpoint does not contain model weights.")

    model = StructureAutoencoder(
        latent_dim=latent_dim,
        factor_output_dim=len(checkpoint_factor_names),
    ).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    verify_files = collect_phase_channel_files(args.verify_root)
    if not verify_files:
        raise FileNotFoundError(
            f"No '*_phase_channels.npy' files were found under verify root: {args.verify_root}"
        )

    output_dir = (
        args.reconstruction_output_dir
        if args.reconstruction_output_dir is not None
        else args.checkpoint.parent / "verify_txt_reconstructions"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = infer_verify_run_tag(args.checkpoint)
    factor_mean_values = checkpoint.get("factor_mean")
    factor_std_values = checkpoint.get("factor_std")
    if (
        not isinstance(factor_mean_values, list)
        or not isinstance(factor_std_values, list)
        or len(factor_mean_values) != len(checkpoint_factor_names)
        or len(factor_std_values) != len(checkpoint_factor_names)
    ):
        raise ValueError(
            "Checkpoint is missing valid 'factor_mean'/'factor_std' statistics."
        )
    factor_mean_array = np.asarray(factor_mean_values, dtype=np.float32)
    factor_std_array = np.asarray(factor_std_values, dtype=np.float32)
    predicted_factor_rows: List[List[object]] = []

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Verify root: {args.verify_root}")
    print(f"Reconstruction TXT output directory: {output_dir}")
    print(f"Verify structures found: {len(verify_files)}")
    print(f"Device: {device}")

    with torch.no_grad():
        for index, path in enumerate(verify_files, start=1):
            phase_channels = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
            input_labels = phase_channels_to_labels(phase_channels)

            input_tensor = torch.from_numpy(phase_channels).permute(3, 0, 1, 2).unsqueeze(0)
            input_tensor = input_tensor.to(device, non_blocking=(device.type == "cuda"))

            reconstruction_logits, factor_predictions, _ = model(input_tensor)
            reconstructed_channels = (
                torch.sigmoid(reconstruction_logits)
                .squeeze(0)
                .permute(1, 2, 3, 0)
                .cpu()
                .numpy()
            )
            reconstructed_labels = phase_channels_to_labels(reconstructed_channels)

            reconstructed_txt_path = (
                output_dir / f"CNNverify_{run_tag}_{path.stem}.txt"
            )
            write_phase_slices_txt(reconstructed_labels, reconstructed_txt_path)

            voxel_accuracy = float(
                np.mean(input_labels == reconstructed_labels)
            )
            normalized_factor_predictions = (
                factor_predictions.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
            )
            predicted_factor_values = (
                normalized_factor_predictions * factor_std_array
                + factor_mean_array
            )
            predicted_factor_rows.append(
                [path.name, *predicted_factor_values.astype(float).tolist()]
            )
            print(
                f"[{index}/{len(verify_files)}] Exported reconstructed TXT for {path.name} -> "
                f"{reconstructed_txt_path.name}, voxel_accuracy={voxel_accuracy:.6f}"
            )

    predicted_factor_csv = (
        output_dir / f"CNNverify_{run_tag}_predicted_structure_factors.csv"
    )
    with predicted_factor_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["structure_name", *checkpoint_factor_names])
        writer.writerows(predicted_factor_rows)
    print(f"Predicted structure factors: {predicted_factor_csv}")


def export_latent_vectors(
    model: StructureAutoencoder,
    dataset: PhaseChannelDataset,
    device: TorchDevice,
    batch_size: int,
    num_workers: int,
    output_dir: Path,
) -> None:
    latent_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    latent_batches: List[NDArray] = []

    model.eval()
    with torch.no_grad():
        for inputs, _ in latent_loader:
            inputs = inputs.to(device, non_blocking=(device.type == "cuda"))
            latent = model.encode(inputs)
            latent_batches.append(latent.cpu().numpy())

    if latent_batches:
        latent_array = np.concatenate(latent_batches, axis=0).astype(np.float32)
    else:
        latent_array = np.empty((0, model.latent_dim), dtype=np.float32)

    np.save(output_dir / "latent_vectors.npy", latent_array)
    with (output_dir / "latent_index.txt").open("w", encoding="utf-8") as handle:
        for path in dataset.file_paths:
            handle.write(f"{path}\n")

def train_model(args: argparse.Namespace) -> None:
    run_start_time = time.perf_counter()
    if TORCH_IMPORT_ERROR is not None:
        raise SystemExit(
            "This script requires PyTorch, but it is not installed in the current Python environment.\n"
            "Install it first, for example:\n"
            "  pip install torch\n"
            f"Original import error: {TORCH_IMPORT_ERROR}"
        )

    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if args.latent_dim <= 0:
        raise ValueError("latent_dim must be positive.")
    if args.report_every <= 0:
        raise ValueError("report_every must be positive.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    denoising_mask_ratio = coerce_float(
        getattr(args, "denoising_mask_ratio", DEFAULT_DENOISING_MASK_RATIO),
        DEFAULT_DENOISING_MASK_RATIO,
    )
    if denoising_mask_ratio < 0.0 or denoising_mask_ratio >= 1.0:
        raise ValueError("denoising_mask_ratio must be in the range [0.0, 1.0).")
    args.denoising_mask_ratio = denoising_mask_ratio

    reconstruction_loss_weight, factor_loss_weight = normalize_loss_weights(
        coerce_float(
            getattr(
                args,
                "reconstruction_loss_weight",
                DEFAULT_RECONSTRUCTION_LOSS_WEIGHT,
            ),
            DEFAULT_RECONSTRUCTION_LOSS_WEIGHT,
        ),
        coerce_float(
            getattr(args, "factor_loss_weight", DEFAULT_FACTOR_LOSS_WEIGHT),
            DEFAULT_FACTOR_LOSS_WEIGHT,
        ),
    )
    args.reconstruction_loss_weight = reconstruction_loss_weight
    args.factor_loss_weight = factor_loss_weight

    set_random_seed(args.seed)

    dataset = PhaseChannelDataset(args.data_root, max_files=args.max_files)
    validation_dataset = PhaseChannelDataset(
        args.validation_root,
        max_files=args.validation_max_files,
    )
    test_dataset = PhaseChannelDataset(
        args.test_root,
        max_files=args.test_max_files,
    )
    factor_names = tuple(dataset.factor_names)
    if tuple(validation_dataset.factor_names) != factor_names:
        raise ValueError("Training and validation datasets use different factor names.")
    if tuple(test_dataset.factor_names) != factor_names:
        raise ValueError("Training and test datasets use different factor names.")

    factor_mean, factor_std = compute_factor_normalization_stats(dataset.factor_values)
    phase_channel_stats = inspect_dataset_phase_channels(dataset.file_paths)
    device = select_device(args.cpu)

    checkpoint: Optional[Dict[str, Any]] = None
    checkpoint_state_dict: Optional[Dict[str, Any]] = None
    checkpoint_latent_dim = args.latent_dim
    completed_epochs_before_resume = 0
    resumed_from_text: Optional[str] = None
    if args.resume_from is None:
        run_dir = make_output_run_dir(args.output_root)
    else:
        run_dir = args.resume_from.parent
        run_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_from is not None:
        checkpoint = load_checkpoint(args.resume_from, device)
        resumed_from_text = str(args.resume_from)
        checkpoint_latent_dim = int(checkpoint["latent_dim"])
        checkpoint_state_dict = checkpoint.get("model_state_dict")
        if not isinstance(checkpoint_state_dict, dict):
            raise ValueError("Checkpoint does not contain a valid model_state_dict.")
        completed_epochs_before_resume = int(checkpoint["completed_epochs"])
        denoising_mask_ratio = coerce_float(
            checkpoint["denoising_mask_ratio"],
            denoising_mask_ratio,
        )
        args.denoising_mask_ratio = denoising_mask_ratio
        reconstruction_loss_weight, factor_loss_weight = normalize_loss_weights(
            coerce_float(
                checkpoint["reconstruction_loss_weight"],
                reconstruction_loss_weight,
            ),
            coerce_float(
                checkpoint["factor_loss_weight"],
                factor_loss_weight,
            ),
        )
        args.reconstruction_loss_weight = reconstruction_loss_weight
        args.factor_loss_weight = factor_loss_weight
        checkpoint_factor_names = tuple(str(name) for name in checkpoint["factor_names"])
        if checkpoint_factor_names != factor_names:
            raise ValueError(
                "Checkpoint factor names do not match the current dataset factor names.\n"
                f"Checkpoint: {checkpoint_factor_names}\n"
                f"Dataset: {factor_names}"
            )
        checkpoint_factor_mean = checkpoint["factor_mean"]
        checkpoint_factor_std = checkpoint["factor_std"]
        if (
            not isinstance(checkpoint_factor_mean, list)
            or not isinstance(checkpoint_factor_std, list)
            or len(checkpoint_factor_mean) != len(factor_names)
            or len(checkpoint_factor_std) != len(factor_names)
        ):
            raise ValueError(
                "Checkpoint factor normalization statistics do not match the model."
            )
        factor_mean = np.asarray(checkpoint_factor_mean, dtype=np.float32)
        factor_std = np.asarray(checkpoint_factor_std, dtype=np.float32)

    if checkpoint_latent_dim != args.latent_dim:
        print(
            "恢复训练时采用 checkpoint 中的隐向量维度。 / Using latent dimension from checkpoint for resumed training: "
            f"{checkpoint_latent_dim}"
        )

    factor_mean_tensor = torch.from_numpy(factor_mean).to(
        device=device,
        dtype=torch.float32,
    )
    factor_std_tensor = torch.from_numpy(factor_std).to(
        device=device,
        dtype=torch.float32,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = StructureAutoencoder(
        latent_dim=checkpoint_latent_dim,
        factor_output_dim=len(factor_names),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    base_learning_rate = float(args.learning_rate)
    channel_counts = cast(List[int], phase_channel_stats["channel_counts"])
    cbd_vector_present = bool(phase_channel_stats["cbd_vector_present"])
    cbd_positive_loss_weight = float(phase_channel_stats["cbd_positive_loss_weight"])
    cbd_pos_weight_tensor: Optional[TorchTensor] = None
    if cbd_vector_present and cbd_positive_loss_weight > 1.0:
        pos_weight_tensor = torch.ones(
            (EXPECTED_INPUT_SHAPE[3], 1, 1, 1),
            dtype=torch.float32,
            device=device,
        )
        pos_weight_tensor[CBD_CHANNEL_INDEX, 0, 0, 0] = cbd_positive_loss_weight
        cbd_pos_weight_tensor = pos_weight_tensor

    report_history: List[Dict[str, Union[float, int]]] = []
    epoch_history: List[Dict[str, Union[float, int]]] = []
    global_structures_seen = 0
    report_total_loss_sum = 0.0
    report_reconstruction_loss_sum = 0.0
    report_factor_loss_sum = 0.0
    report_structure_count = 0

    if checkpoint_state_dict is not None:
        assert checkpoint is not None
        model.load_state_dict(checkpoint_state_dict)

        optimizer_state_dict = checkpoint.get("optimizer_state_dict")
        if not isinstance(optimizer_state_dict, dict):
            raise ValueError("Checkpoint does not contain a valid optimizer_state_dict.")
        optimizer.load_state_dict(optimizer_state_dict)

        loaded_report_history = checkpoint.get("report_history", [])
        if isinstance(loaded_report_history, list):
            report_history = cast(
                List[Dict[str, Union[float, int]]],
                [dict(row) for row in loaded_report_history if isinstance(row, dict)],
            )
        loaded_epoch_history = checkpoint.get("epoch_history", [])
        if isinstance(loaded_epoch_history, list):
            epoch_history = cast(
                List[Dict[str, Union[float, int]]],
                [dict(row) for row in loaded_epoch_history if isinstance(row, dict)],
            )
        try:
            global_structures_seen = int(checkpoint.get("global_structures_seen", 0))
        except (TypeError, ValueError):
            global_structures_seen = 0

    best_epoch = completed_epochs_before_resume
    best_epoch_average_training_total_loss = float("inf")
    best_epoch_average_validation_total_loss = float("inf")
    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
    best_global_structures_seen = global_structures_seen

    if checkpoint is not None:
        try:
            best_epoch = int(checkpoint.get("best_epoch", completed_epochs_before_resume))
        except (TypeError, ValueError):
            best_epoch = completed_epochs_before_resume

        best_epoch_average_validation_total_loss = coerce_float(
            checkpoint["best_epoch_average_validation_total_loss"],
            float("inf"),
        )
        best_epoch_average_training_total_loss = coerce_float(
            checkpoint["best_epoch_average_training_total_loss"],
            float("inf"),
        )

        stored_best_model_state = checkpoint.get("best_model_state_dict")
        if isinstance(stored_best_model_state, dict):
            best_model_state_dict = copy.deepcopy(stored_best_model_state)
        stored_best_optimizer_state = checkpoint.get("best_optimizer_state_dict")
        if isinstance(stored_best_optimizer_state, dict):
            best_optimizer_state_dict = copy.deepcopy(stored_best_optimizer_state)
        try:
            best_global_structures_seen = int(
                checkpoint.get("best_global_structures_seen", global_structures_seen)
            )
        except (TypeError, ValueError):
            best_global_structures_seen = global_structures_seen

    denoising_example_dir = save_denoising_example_visualizations(
        file_paths=dataset.file_paths,
        output_dir=run_dir,
        mask_ratio=denoising_mask_ratio,
        seed=int(args.seed),
    )

    print(f"Data root: {args.data_root}")
    print(f"Validation root: {args.validation_root}")
    print(f"Test root: {args.test_root}")
    print(f"Output run directory: {run_dir}")
    print(f"Found structures: {len(dataset)}")
    print(f"Found validation structures: {len(validation_dataset)}")
    print(f"Found test structures: {len(test_dataset)}")
    print(f"Device: {device}")
    print(
        "Training-set channel voxel counts "
        f"(electrolyte, active, cbd): {channel_counts[0]}, {channel_counts[1]}, {channel_counts[2]}"
    )
    print(
        "CBD one-hot vector (0,0,1) present in training data: "
        f"{'Yes' if cbd_vector_present else 'No'}"
    )
    print(f"CBD positive loss weight: {cbd_positive_loss_weight:.4f}")
    print(f"Denoising mask ratio during training: {denoising_mask_ratio:.4f}")
    print("Structure-factor targets: " + ", ".join(factor_names))
    print(
        "Loss weights (normalized): "
        f"reconstruction={reconstruction_loss_weight:.3f}, "
        f"structure_factors={factor_loss_weight:.3f}"
    )
    if denoising_example_dir is not None:
        print(f"Denoising example figures: {denoising_example_dir}")
    print(
        "Early stopping patience: "
        f"{EARLY_STOPPING_PATIENCE} epochs; stop when either training or validation total loss stops improving"
    )
    print(
        "Learning rate schedule: "
        f"multiply by {LEARNING_RATE_DECAY_FACTOR:.2f} every {LEARNING_RATE_DECAY_EPOCHS} epochs"
    )
    print(f"Base learning rate: {base_learning_rate:.6e}")
    if best_epoch_average_validation_total_loss < float('inf'):
        print(
            "Current best checkpoint validation total loss: "
            f"{best_epoch_average_validation_total_loss:.6f} at epoch {best_epoch}"
        )
    if resumed_from_text is None:
        print(f"Epochs: {args.epochs}")
    else:
        print(f"Resume from: {resumed_from_text}")
        print(f"Completed epochs before resume: {completed_epochs_before_resume}")
        print(f"Additional epochs this run: {args.epochs}")
        print(
            f"Target total completed epochs: {completed_epochs_before_resume + args.epochs}"
        )
    print(f"Batch size: {args.batch_size}")
    print(f"Latent dimension: {checkpoint_latent_dim}")
    print(f"Model architecture: {MODEL_ARCHITECTURE_NAME}")
    print(f"Report every: {args.report_every} structures")

    target_completed_epochs = completed_epochs_before_resume + args.epochs
    last_trained_epoch = completed_epochs_before_resume
    last_observed_epoch_average_training_total_loss = (
        best_epoch_average_training_total_loss
        if best_epoch_average_training_total_loss < float("inf")
        else 0.0
    )
    last_observed_epoch_average_validation_total_loss = (
        best_epoch_average_validation_total_loss
        if best_epoch_average_validation_total_loss < float("inf")
        else 0.0
    )
    epochs_without_training_improvement = 0
    epochs_without_validation_improvement = 0
    early_stopped = False

    for epoch in range(completed_epochs_before_resume + 1, target_completed_epochs + 1):
        current_learning_rate = compute_scheduled_learning_rate(
            base_learning_rate, epoch - 1
        )
        set_optimizer_learning_rate(optimizer, current_learning_rate)
        model.train()
        epoch_total_loss_sum = 0.0
        epoch_reconstruction_loss_sum = 0.0
        epoch_factor_loss_sum = 0.0
        epoch_structure_count = 0

        for batch_index, (inputs, factor_targets) in enumerate(data_loader, start=1):
            inputs = inputs.to(device, non_blocking=(device.type == "cuda"))
            factor_targets = factor_targets.to(
                device,
                non_blocking=(device.type == "cuda"),
            )
            optimizer.zero_grad(set_to_none=True)

            masked_inputs = apply_random_voxel_denoising_mask(
                inputs, denoising_mask_ratio
            )
            reconstruction_logits, factor_predictions, _ = model(masked_inputs)
            reconstruction_loss = compute_reconstruction_loss(
                reconstruction_logits,
                inputs,
                cbd_pos_weight_tensor,
            )
            normalized_factor_targets = normalize_factor_targets(
                factor_targets,
                factor_mean_tensor,
                factor_std_tensor,
            )
            factor_loss = compute_factor_loss(
                factor_predictions,
                normalized_factor_targets,
            )
            total_loss = combine_multitask_losses(
                reconstruction_loss,
                factor_loss,
                reconstruction_loss_weight,
                factor_loss_weight,
            )
            total_loss.backward()
            optimizer.step()

            batch_size = int(inputs.shape[0])
            total_loss_value = float(total_loss.item())
            reconstruction_loss_value = float(reconstruction_loss.item())
            factor_loss_value = float(factor_loss.item())
            epoch_total_loss_sum += total_loss_value * batch_size
            epoch_reconstruction_loss_sum += reconstruction_loss_value * batch_size
            epoch_factor_loss_sum += factor_loss_value * batch_size
            epoch_structure_count += batch_size
            global_structures_seen += batch_size
            report_total_loss_sum += total_loss_value * batch_size
            report_reconstruction_loss_sum += reconstruction_loss_value * batch_size
            report_factor_loss_sum += factor_loss_value * batch_size
            report_structure_count += batch_size

            if report_structure_count >= args.report_every:
                average_report_total_loss = report_total_loss_sum / max(1, report_structure_count)
                average_report_reconstruction_loss = (
                    report_reconstruction_loss_sum / max(1, report_structure_count)
                )
                average_report_factor_loss = (
                    report_factor_loss_sum / max(1, report_structure_count)
                )
                report_history.append(
                    {
                        "epoch": epoch,
                        "structures_seen": global_structures_seen,
                        "average_total_loss": average_report_total_loss,
                        "average_reconstruction_loss": average_report_reconstruction_loss,
                        "average_factor_loss": average_report_factor_loss,
                    }
                )
                print(
                    f"[Epoch {epoch:03d} | Batch {batch_index:04d}] "
                    f"Processed {global_structures_seen} structures, "
                    f"average total/reconstruction/factor loss = "
                    f"{average_report_total_loss:.6f}/"
                    f"{average_report_reconstruction_loss:.6f}/"
                    f"{average_report_factor_loss:.6f}, "
                    f"lr = {current_learning_rate:.6e}"
                )
                report_total_loss_sum = 0.0
                report_reconstruction_loss_sum = 0.0
                report_factor_loss_sum = 0.0
                report_structure_count = 0

        average_epoch_training_total_loss = epoch_total_loss_sum / max(1, epoch_structure_count)
        average_epoch_training_reconstruction_loss = (
            epoch_reconstruction_loss_sum / max(1, epoch_structure_count)
        )
        average_epoch_training_factor_loss = (
            epoch_factor_loss_sum / max(1, epoch_structure_count)
        )
        validation_breakdown = evaluate_average_multitask_losses(
            model,
            validation_loader,
            device,
            cbd_pos_weight_tensor,
            factor_mean_tensor,
            factor_std_tensor,
            reconstruction_loss_weight,
            factor_loss_weight,
        )
        average_epoch_validation_total_loss = float(validation_breakdown["total_loss"])
        average_epoch_validation_reconstruction_loss = float(
            validation_breakdown["reconstruction_loss"]
        )
        average_epoch_validation_factor_loss = float(
            validation_breakdown["factor_loss"]
        )

        last_trained_epoch = epoch
        last_observed_epoch_average_training_total_loss = average_epoch_training_total_loss
        last_observed_epoch_average_validation_total_loss = average_epoch_validation_total_loss
        epoch_history.append(
            {
                "epoch": epoch,
                "average_training_total_loss": average_epoch_training_total_loss,
                "average_training_reconstruction_loss": average_epoch_training_reconstruction_loss,
                "average_training_factor_loss": average_epoch_training_factor_loss,
                "average_validation_total_loss": average_epoch_validation_total_loss,
                "average_validation_reconstruction_loss": average_epoch_validation_reconstruction_loss,
                "average_validation_factor_loss": average_epoch_validation_factor_loss,
            }
        )

        training_improved = (
            average_epoch_training_total_loss
            < (best_epoch_average_training_total_loss - EARLY_STOPPING_MIN_DELTA)
        )
        validation_improved = (
            average_epoch_validation_total_loss
            < (best_epoch_average_validation_total_loss - EARLY_STOPPING_MIN_DELTA)
        )

        if training_improved:
            best_epoch_average_training_total_loss = average_epoch_training_total_loss
            epochs_without_training_improvement = 0
        else:
            epochs_without_training_improvement += 1

        if validation_improved:
            best_epoch_average_validation_total_loss = average_epoch_validation_total_loss
            best_epoch = epoch
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
            best_global_structures_seen = global_structures_seen
            epochs_without_validation_improvement = 0
        else:
            epochs_without_validation_improvement += 1

        print(
            f"Epoch {epoch:03d} finished, "
            f"train total/recon/factor = "
            f"{average_epoch_training_total_loss:.6f}/"
            f"{average_epoch_training_reconstruction_loss:.6f}/"
            f"{average_epoch_training_factor_loss:.6f}, "
            f"validation total/recon/factor = "
            f"{average_epoch_validation_total_loss:.6f}/"
            f"{average_epoch_validation_reconstruction_loss:.6f}/"
            f"{average_epoch_validation_factor_loss:.6f}, "
            f"lr = {current_learning_rate:.6e}, "
            f"train patience = {epochs_without_training_improvement}/{EARLY_STOPPING_PATIENCE}, "
            f"validation patience = {epochs_without_validation_improvement}/{EARLY_STOPPING_PATIENCE}"
        )
        if (
            epochs_without_training_improvement >= EARLY_STOPPING_PATIENCE
            or epochs_without_validation_improvement >= EARLY_STOPPING_PATIENCE
        ):
            early_stopped = True
            print(
                f"Early stopping triggered at epoch {epoch:03d}. "
                f"Restoring validation-selected best parameters from epoch {best_epoch:03d} "
                f"with validation total loss {best_epoch_average_validation_total_loss:.6f}."
            )
            break

    if report_structure_count > 0:
        average_report_total_loss = report_total_loss_sum / max(1, report_structure_count)
        average_report_reconstruction_loss = (
            report_reconstruction_loss_sum / max(1, report_structure_count)
        )
        average_report_factor_loss = report_factor_loss_sum / max(1, report_structure_count)
        report_history.append(
            {
                "epoch": last_trained_epoch,
                "structures_seen": global_structures_seen,
                "average_total_loss": average_report_total_loss,
                "average_reconstruction_loss": average_report_reconstruction_loss,
                "average_factor_loss": average_report_factor_loss,
            }
        )
        print(
            f"[Final] Processed {global_structures_seen} structures, "
            f"average total/reconstruction/factor loss = "
            f"{average_report_total_loss:.6f}/"
            f"{average_report_reconstruction_loss:.6f}/"
            f"{average_report_factor_loss:.6f}"
        )

    model.load_state_dict(best_model_state_dict)
    optimizer.load_state_dict(best_optimizer_state_dict)
    best_learning_rate = compute_scheduled_learning_rate(
        base_learning_rate, max(0, best_epoch - 1)
    )
    set_optimizer_learning_rate(optimizer, best_learning_rate)
    global_structures_seen = best_global_structures_seen
    test_breakdown = evaluate_average_multitask_losses(
        model,
        test_loader,
        device,
        cbd_pos_weight_tensor,
        factor_mean_tensor,
        factor_std_tensor,
        reconstruction_loss_weight,
        factor_loss_weight,
    )
    final_test_total_loss = float(test_breakdown["total_loss"])
    final_test_reconstruction_loss = float(test_breakdown["reconstruction_loss"])
    final_test_factor_loss = float(test_breakdown["factor_loss"])

    checkpoint_report_history = trim_history_to_epoch(report_history, best_epoch)
    checkpoint_epoch_history = trim_history_to_epoch(epoch_history, best_epoch)

    report_tag = f"every_{args.report_every}"
    save_loss_history_csv(report_history, run_dir / f"loss_history_{report_tag}.csv")
    plot_loss_history(report_history, run_dir / f"loss_curve_{report_tag}.png")
    save_epoch_history_csv(epoch_history, run_dir / "epoch_loss_history.csv")
    plot_epoch_history(epoch_history, run_dir / "epoch_loss_curve.png")

    export_latent_vectors(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=run_dir,
    )

    torch.save(model.state_dict(), run_dir / "structure_autoencoder_state_dict.pt")
    completed_epochs_to_save = best_epoch
    epochs_completed_this_run = max(
        0, last_trained_epoch - completed_epochs_before_resume
    )
    torch.save(
        {
            "latent_dim": checkpoint_latent_dim,
            "architecture_name": MODEL_ARCHITECTURE_NAME,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_model_state_dict": best_model_state_dict,
            "best_optimizer_state_dict": best_optimizer_state_dict,
            "input_shape": EXPECTED_INPUT_SHAPE,
            "factor_names": list(factor_names),
            "factor_mean": factor_mean.astype(float).tolist(),
            "factor_std": factor_std.astype(float).tolist(),
            "factor_json_suffix": STRUCTURE_FACTOR_JSON_SUFFIX,
            "completed_epochs": completed_epochs_to_save,
            "epochs_this_run": epochs_completed_this_run,
            "last_trained_epoch": last_trained_epoch,
            "best_epoch": best_epoch,
            "best_epoch_average_total_loss": best_epoch_average_validation_total_loss,
            "best_epoch_average_training_total_loss": best_epoch_average_training_total_loss,
            "best_epoch_average_validation_total_loss": best_epoch_average_validation_total_loss,
            "last_observed_epoch_average_total_loss": last_observed_epoch_average_validation_total_loss,
            "last_observed_epoch_average_training_total_loss": last_observed_epoch_average_training_total_loss,
            "last_observed_epoch_average_validation_total_loss": last_observed_epoch_average_validation_total_loss,
            "early_stopped": early_stopped,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "base_learning_rate": base_learning_rate,
            "learning_rate_decay_every_epochs": LEARNING_RATE_DECAY_EPOCHS,
            "learning_rate_decay_factor": LEARNING_RATE_DECAY_FACTOR,
            "reconstruction_loss_weight": reconstruction_loss_weight,
            "factor_loss_weight": factor_loss_weight,
            "best_global_structures_seen": global_structures_seen,
            "global_structures_seen": global_structures_seen,
            "report_history": checkpoint_report_history,
            "epoch_history": checkpoint_epoch_history,
            "report_every": args.report_every,
            "batch_size": args.batch_size,
            "learning_rate": base_learning_rate,
            "denoising_mask_ratio": denoising_mask_ratio,
            "validation_root": str(args.validation_root),
            "validation_structure_count": len(validation_dataset),
            "test_root": str(args.test_root),
            "test_structure_count": len(test_dataset),
            "final_test_total_loss": final_test_total_loss,
            "final_test_reconstruction_loss": final_test_reconstruction_loss,
            "final_test_factor_loss": final_test_factor_loss,
            "denoising_example_directory": (
                str(denoising_example_dir) if denoising_example_dir is not None else None
            ),
            "data_root": str(args.data_root),
            "resumed_from": resumed_from_text,
            "channel_counts": channel_counts,
            "cbd_vector_present": cbd_vector_present,
            "cbd_positive_loss_weight": cbd_positive_loss_weight,
        },
        run_dir / "structure_autoencoder_checkpoint.pt",
    )

    summary = {
        "data_root": str(args.data_root),
        "validation_root": str(args.validation_root),
        "test_root": str(args.test_root),
        "output_run_directory": str(run_dir),
        "structure_count": len(dataset),
        "validation_structure_count": len(validation_dataset),
        "test_structure_count": len(test_dataset),
        "epochs": args.epochs,
        "epochs_completed_this_run": epochs_completed_this_run,
        "completed_epochs_before_resume": completed_epochs_before_resume,
        "total_completed_epochs": completed_epochs_to_save,
        "last_trained_epoch": last_trained_epoch,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "learning_rate": base_learning_rate,
        "denoising_mask_ratio": denoising_mask_ratio,
        "reconstruction_loss_weight": reconstruction_loss_weight,
        "factor_loss_weight": factor_loss_weight,
        "denoising_example_directory": (
            str(denoising_example_dir) if denoising_example_dir is not None else None
        ),
        "best_learning_rate": best_learning_rate,
        "learning_rate_decay_every_epochs": LEARNING_RATE_DECAY_EPOCHS,
        "learning_rate_decay_factor": LEARNING_RATE_DECAY_FACTOR,
        "latent_dim": checkpoint_latent_dim,
        "architecture_name": MODEL_ARCHITECTURE_NAME,
        "report_every": args.report_every,
        "device": str(device),
        "factor_names": list(factor_names),
        "factor_mean": factor_mean.astype(float).tolist(),
        "factor_std": factor_std.astype(float).tolist(),
        "final_epoch_average_total_loss": best_epoch_average_validation_total_loss,
        "best_epoch_average_total_loss": best_epoch_average_validation_total_loss,
        "best_epoch_average_training_total_loss": best_epoch_average_training_total_loss,
        "best_epoch_average_validation_total_loss": best_epoch_average_validation_total_loss,
        "final_test_total_loss": final_test_total_loss,
        "final_test_reconstruction_loss": final_test_reconstruction_loss,
        "final_test_factor_loss": final_test_factor_loss,
        "last_observed_epoch_average_total_loss": last_observed_epoch_average_validation_total_loss,
        "last_observed_epoch_average_training_total_loss": last_observed_epoch_average_training_total_loss,
        "last_observed_epoch_average_validation_total_loss": last_observed_epoch_average_validation_total_loss,
        "early_stopped": early_stopped,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "resumed_from": resumed_from_text,
        "channel_counts": channel_counts,
        "cbd_vector_present": cbd_vector_present,
        "cbd_positive_loss_weight": cbd_positive_loss_weight,
        "elapsed_seconds_this_run": float(time.perf_counter() - run_start_time),
    }
    with (run_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    elapsed_seconds = float(summary["elapsed_seconds_this_run"])
    print("Training finished.")
    print(
        "Best validation total loss kept in saved model: "
        f"{best_epoch_average_validation_total_loss:.6f} (epoch {best_epoch:03d})"
    )
    print(
        "Held-out test total/reconstruction/factor loss: "
        f"{final_test_total_loss:.6f}/"
        f"{final_test_reconstruction_loss:.6f}/"
        f"{final_test_factor_loss:.6f}"
    )
    if early_stopped:
        print(
            f"Training stopped early after epoch {last_trained_epoch:03d} "
            f"and rolled back to epoch {best_epoch:03d}."
        )
    if resumed_from_text is None:
        print(
            f"本次新训练耗时 / Elapsed time for this training run: "
            f"{format_elapsed_time(elapsed_seconds)} ({elapsed_seconds:.2f} s)"
        )
    else:
        print(
            f"本次续训耗时 / Elapsed time for this resume run: "
            f"{format_elapsed_time(elapsed_seconds)} ({elapsed_seconds:.2f} s)"
        )
    if denoising_example_dir is not None:
        print(f"Denoising examples: {denoising_example_dir}")
    print(f"Model weights: {run_dir / 'structure_autoencoder_state_dict.pt'}")
    print(f"Latents: {run_dir / 'latent_vectors.npy'}")
    print(f"Loss curve: {run_dir / f'loss_curve_{report_tag}.png'}")
    print(f"Epoch loss curve: {run_dir / 'epoch_loss_curve.png'}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args = configure_mode_interactively(args)

    if args.mode == "verify":
        reconstruct_verify_set(args)
        return 0
    if args.mode == "resume" and args.resume_from is None:
        raise ValueError("--resume-from is required in resume mode.")
    train_model(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
