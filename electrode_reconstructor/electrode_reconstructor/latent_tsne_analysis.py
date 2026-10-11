#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast
from xml.etree import ElementTree as ET
import zipfile

import electrode_structure_factors as structure_factors
import train_structure_autoencoder as autoencoder

try:
    TSNE = cast(Any, importlib.import_module("sklearn.manifold").TSNE)
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires scikit-learn. Install it with: pip install scikit-learn"
    ) from exc

try:
    KMeans = cast(Any, importlib.import_module("sklearn.cluster").KMeans)
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires scikit-learn. Install it with: pip install scikit-learn"
    ) from exc


DEFAULT_TEST_ROOT = autoencoder.DEFAULT_DATA_ROOT / "test"
WORKSHEET_NS = {"xlsx": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained checkpoint, scan test-set structures, extract 128-D latent "
            "vectors, and reduce them to 2D with t-SNE."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to structure_autoencoder_checkpoint.pt",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_TEST_ROOT,
        help=f"Root directory containing test structures. Default: {DEFAULT_TEST_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for latent/t-SNE outputs. Default: <checkpoint_dir>\\test_latent_tsne",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size used during latent extraction. Default: 4",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader workers. Default: 0",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Optional limit on the number of test structures.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity. Default: 30",
    )
    parser.add_argument(
        "--kmeans-clusters",
        type=int,
        default=5,
        help="Number of KMeans clusters. Default: 5",
    )
    parser.add_argument(
        "--tsne-iterations",
        type=int,
        default=1000,
        help="Number of t-SNE iterations. Default: 1000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260410,
        help="Random seed for reproducibility. Default: 20260410",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    return parser.parse_args(argv)


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


def prompt_bool(prompt: str, default: bool) -> bool:
    default_text = "是 / yes" if default else "否 / no"
    true_values = {"y", "yes", "1", "true", "是"}
    false_values = {"n", "no", "0", "false", "否"}
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


def configure_interactively(args: argparse.Namespace) -> argparse.Namespace:
    if args.checkpoint is None:
        print("进入隐向量提取与 t-SNE 降维设置。 / Configure latent extraction and t-SNE.")
        checkpoint_text = prompt_text(
            "请输入 checkpoint 路径 / Enter checkpoint path"
        )
        if not checkpoint_text:
            raise ValueError("checkpoint 路径不能为空。 / checkpoint path is required.")
        args.checkpoint = Path(checkpoint_text)
        args.data_root = Path(
            prompt_text("测试集目录 / Test data root", str(args.data_root))
        )
        default_output_dir = args.checkpoint.parent / "test_latent_tsne"
        args.output_dir = Path(
            prompt_text("输出目录 / Output directory", str(default_output_dir))
        )
        args.batch_size = prompt_int("批大小 / Batch size", int(args.batch_size))
        args.num_workers = prompt_int(
            "DataLoader 进程数 / DataLoader workers", int(args.num_workers)
        )
        args.perplexity = prompt_float("t-SNE perplexity", float(args.perplexity))
        args.kmeans_clusters = prompt_int(
            "KMeans 聚类数 / KMeans cluster count", int(args.kmeans_clusters)
        )
        args.tsne_iterations = prompt_int(
            "t-SNE 迭代次数 / t-SNE iterations", int(args.tsne_iterations)
        )
        args.seed = prompt_int("随机种子 / Random seed", int(args.seed))
        args.cpu = prompt_bool("是否强制使用 CPU / Force CPU only", bool(args.cpu))

    if args.output_dir is None:
        args.output_dir = args.checkpoint.parent / "test_latent_tsne"
    return args


def build_tsne(
    perplexity: float,
    tsne_iterations: int,
    seed: int,
) -> Any:
    common_kwargs: Dict[str, Any] = {
        "n_components": 2,
        "perplexity": perplexity,
        "random_state": seed,
        "init": "pca",
        "learning_rate": "auto",
        "verbose": 2,
    }
    try:
        return TSNE(max_iter=tsne_iterations, **common_kwargs)
    except TypeError:
        return TSNE(n_iter=tsne_iterations, **common_kwargs)


def build_kmeans(cluster_count: int, seed: int) -> Any:
    common_kwargs: Dict[str, Any] = {
        "n_clusters": cluster_count,
        "random_state": seed,
    }
    try:
        return KMeans(n_init="auto", **common_kwargs)
    except TypeError:
        return KMeans(n_init=10, **common_kwargs)


def sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return cleaned or "value"


def split_bilingual_label(label_text: object) -> str:
    parts = [part.strip() for part in str(label_text).split(" / ")]
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return str(label_text).strip()


def parse_excel_cell_value(cell: ET.Element) -> object:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        texts = [
            element.text or ""
            for element in cell.findall(".//xlsx:t", WORKSHEET_NS)
        ]
        return "".join(texts)

    value_element = cell.find("xlsx:v", WORKSHEET_NS)
    if value_element is None or value_element.text is None:
        return ""
    raw_text = value_element.text.strip()
    if cell_type == "b":
        return raw_text == "1"
    if raw_text.lower() == "inf":
        return float("inf")
    try:
        if any(character in raw_text.lower() for character in (".", "e")):
            return float(raw_text)
        return int(raw_text)
    except ValueError:
        return raw_text


def find_structure_factor_xlsx(sample_path: Path) -> Optional[Path]:
    parent = sample_path.parent
    expected = parent / f"structure_factor_{parent.name}.xlsx"
    if expected.exists():
        return expected
    candidates = sorted(parent.glob("structure_factor_*.xlsx"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def find_phase_slices_txt(sample_path: Path) -> Optional[Path]:
    expected_name = sample_path.name
    if expected_name.endswith("_phase_channels.npy"):
        expected_name = expected_name.replace("_phase_channels.npy", "_phase_slices.txt")
        expected_path = sample_path.parent / expected_name
        if expected_path.exists():
            return expected_path
    candidates = sorted(sample_path.parent.glob("*_phase_slices.txt"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def read_structure_factor_excel(path: Path) -> Dict[str, object]:
    with zipfile.ZipFile(path, "r") as workbook:
        sheet_xml = workbook.read("xl/worksheets/sheet1.xml")
    root = ET.fromstring(sheet_xml)
    rows: Dict[str, object] = {}
    for row in root.findall(".//xlsx:sheetData/xlsx:row", WORKSHEET_NS):
        cells = row.findall("xlsx:c", WORKSHEET_NS)
        if len(cells) < 2:
            continue
        key = split_bilingual_label(parse_excel_cell_value(cells[0]))
        value = parse_excel_cell_value(cells[1])
        if key:
            rows[key] = value
    return rows


def compute_structure_factor_labels_from_txt(path: Path) -> Dict[str, object]:
    phase_labels = structure_factors.load_phase_slices_txt(path)
    surface = structure_factors.compute_specific_surface_area(
        phase_labels=phase_labels,
        voxel_size_um=0.5,
    )
    agglomeration = structure_factors.compute_particle_agglomeration(
        phase_labels=phase_labels,
    )
    electronic = structure_factors.compute_electronic_connectivity(
        phase_labels=phase_labels,
        particle_value=1,
        conductive_values=(1, 2),
    )
    electrolyte = structure_factors.compute_electrolyte_connectivity(
        phase_labels=phase_labels,
        electrolyte_value=0,
    )
    tortuosity = structure_factors.compute_tortuosity(
        phase_labels=phase_labels,
        voxel_size_um=0.5,
        electrolyte_connectivity=electrolyte,
        electrolyte_value=0,
    )
    equivalent_diameter = structure_factors.compute_equivalent_particle_diameter(
        agglomeration=agglomeration,
        voxel_size_um=0.5,
    )
    return {
        "Input Structure": structure_factors.structure_display_name(path),
        "Solid Voxel Count": surface.solid_voxel_count,
        "Total Interfacial Area (um^2)": surface.total_interface_area_um2,
        "Total Solid Volume (um^3)": surface.solid_volume_um3,
        "Specific Surface Area (1/um)": surface.specific_surface_area_inv_um,
        "Total Cluster Count": agglomeration.particle_cluster_count,
        "Largest Particle Cluster Volume (voxel)": agglomeration.largest_cluster_volume_voxels,
        "Unknown Particle Cluster Count": agglomeration.unknown_cluster_count,
        "Total Unknown Particle Cluster Volume (voxel)": agglomeration.unknown_cluster_volume_voxels,
        "Unknown Particle Volume Fraction": agglomeration.unknown_cluster_volume_fraction,
        "Normalized Average Cluster Volume": (
            agglomeration.normalized_average_cluster_volume
            if agglomeration.normalized_average_cluster_volume is not None
            else "N/A"
        ),
        "Electronic Connectivity": electronic.electronic_connectivity,
        "Electrolyte Connectivity": electrolyte.electrolyte_connectivity,
        "Percolating": electrolyte.percolating,
        "Tortuosity τ": tortuosity.tortuosity_tau,
        "Equivalent Particle Diameter (um)": equivalent_diameter.equivalent_particle_diameter_um,
    }


def load_structure_factor_labels(
    file_paths: Sequence[Path],
) -> Tuple[List[Dict[str, object]], List[str], int]:
    rows: List[Dict[str, object]] = []
    all_factor_names: List[str] = []
    factor_name_set = set()
    matched_count = 0
    total_count = len(file_paths)
    start_time = time.perf_counter()
    last_reported_percent = 0

    if total_count > 0:
        print(
            "Structure factor loading started / "
            "开始加载结构因子: "
            f"{total_count} samples"
        )

    for index, path in enumerate(file_paths, start=1):
        factor_path = find_structure_factor_xlsx(path)
        row: Dict[str, object] = {
            "file_path": str(path),
            "structure_factor_source": "",
        }
        if factor_path is not None:
            matched_count += 1
            factor_data = read_structure_factor_excel(factor_path)
            row["structure_factor_source"] = str(factor_path)
            for key, value in factor_data.items():
                row[key] = value
                if key not in factor_name_set:
                    factor_name_set.add(key)
                    all_factor_names.append(key)
        else:
            txt_path = find_phase_slices_txt(path)
            if txt_path is not None:
                matched_count += 1
                factor_data = compute_structure_factor_labels_from_txt(txt_path)
                row["structure_factor_source"] = str(txt_path)
                for key, value in factor_data.items():
                    row[key] = value
                    if key not in factor_name_set:
                        factor_name_set.add(key)
                        all_factor_names.append(key)
        rows.append(row)

        if total_count > 0:
            elapsed_seconds = float(time.perf_counter() - start_time)
            percent_complete = float(index) / float(total_count) * 100.0
            report_percent = int(percent_complete // 10) * 10
            if index == total_count:
                report_percent = 100
            if report_percent >= 10 and report_percent > last_reported_percent:
                average_seconds_per_sample = elapsed_seconds / float(index)
                remaining_samples = total_count - index
                estimated_remaining_seconds = (
                    average_seconds_per_sample * float(remaining_samples)
                )
                print(
                    "[Structure Factors] "
                    f"{index}/{total_count} "
                    f"({report_percent}%) "
                    f"| matched={matched_count} "
                    f"| elapsed={autoencoder.format_elapsed_time(elapsed_seconds)} "
                    f"({elapsed_seconds:.2f} s) "
                    f"| eta={autoencoder.format_elapsed_time(estimated_remaining_seconds)} "
                    f"({estimated_remaining_seconds:.2f} s)"
                )
                last_reported_percent = report_percent
    return rows, all_factor_names, matched_count


def coerce_plot_numeric(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return float("nan")
        lower_text = text.lower()
        if lower_text == "inf":
            return float("inf")
        if lower_text in {"yes", "true"}:
            return 1.0
        if lower_text in {"no", "false"}:
            return 0.0
        if " / " in text:
            english_text = split_bilingual_label(text).strip().lower()
            if english_text in {"yes", "true"}:
                return 1.0
            if english_text in {"no", "false"}:
                return 0.0
        try:
            return float(text)
        except ValueError:
            return float("nan")
    return float("nan")


def extract_latent_vectors(
    checkpoint_path: Path,
    data_root: Path,
    batch_size: int,
    num_workers: int,
    max_files: Optional[int],
    force_cpu: bool,
) -> Tuple[Any, List[Path], int, str]:
    if autoencoder.TORCH_IMPORT_ERROR is not None:
        raise SystemExit(
            "This script requires PyTorch, but it is not installed in the current Python environment.\n"
            f"Original import error: {autoencoder.TORCH_IMPORT_ERROR}"
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    device = autoencoder.select_device(force_cpu)
    checkpoint = autoencoder.load_checkpoint(checkpoint_path, device)
    latent_dim = int(checkpoint.get("latent_dim", 128))
    checkpoint_factor_names = checkpoint.get("factor_names")
    if isinstance(checkpoint_factor_names, list) and checkpoint_factor_names:
        factor_output_dim = len(checkpoint_factor_names)
    else:
        factor_output_dim = len(autoencoder.STRUCTURE_FACTOR_NAMES)
    model_state_dict = cast(
        Optional[Dict[str, Any]],
        checkpoint.get("model_state_dict", checkpoint),
    )
    if model_state_dict is None:
        raise ValueError("The checkpoint does not contain model weights.")

    dataset = autoencoder.PhaseChannelDataset(data_root, max_files=max_files)
    data_loader = autoencoder.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = autoencoder.StructureAutoencoder(
        latent_dim=latent_dim,
        factor_output_dim=factor_output_dim,
    ).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    latent_batches: List[Any] = []
    with autoencoder.torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (tuple, list)):
                if not batch:
                    raise ValueError("Received an empty batch from DataLoader.")
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.to(device, non_blocking=(device.type == "cuda"))
            latent = model.encode(inputs)
            latent_batches.append(latent.cpu())

    if not latent_batches:
        raise ValueError("No latent vectors were generated from the test set.")

    latent_tensor = autoencoder.torch.cat(latent_batches, dim=0)
    latent_array = latent_tensor.numpy()
    return latent_array, list(dataset.file_paths), latent_dim, str(device)


def save_latent_csv(
    file_paths: Sequence[Path], latent_vectors: Any, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latent_dim = int(latent_vectors.shape[1])
    fieldnames = ["sample_index", "file_name", "file_path"] + [
        f"latent_{index:03d}" for index in range(latent_dim)
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (path, row) in enumerate(zip(file_paths, latent_vectors), start=1):
            record: Dict[str, Any] = {
                "sample_index": index,
                "file_name": path.name,
                "file_path": str(path),
            }
            for latent_index, value in enumerate(row):
                record[f"latent_{latent_index:03d}"] = float(value)
            writer.writerow(record)


def save_tsne_csv(
    file_paths: Sequence[Path], tsne_2d: Any, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_index", "file_name", "file_path", "tsne_x", "tsne_y"],
        )
        writer.writeheader()
        for index, (path, row) in enumerate(zip(file_paths, tsne_2d), start=1):
            writer.writerow(
                {
                    "sample_index": index,
                    "file_name": path.name,
                    "file_path": str(path),
                    "tsne_x": float(row[0]),
                    "tsne_y": float(row[1]),
                }
            )


def save_combined_analysis_csv(
    file_paths: Sequence[Path],
    latent_vectors: Any,
    tsne_2d: Any,
    kmeans_labels: Any,
    factor_rows: Sequence[Dict[str, object]],
    factor_names: Sequence[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latent_dim = int(latent_vectors.shape[1])
    fieldnames = [
        "sample_index",
        "file_name",
        "file_path",
        "structure_factor_source",
        "kmeans_cluster",
        "tsne_x",
        "tsne_y",
    ]
    fieldnames.extend(f"latent_{index:03d}" for index in range(latent_dim))
    fieldnames.extend(factor_names)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, path in enumerate(file_paths, start=1):
            factor_row = factor_rows[index - 1] if index - 1 < len(factor_rows) else {}
            record: Dict[str, object] = {
                "sample_index": index,
                "file_name": path.name,
                "file_path": str(path),
                "structure_factor_source": factor_row.get("structure_factor_source", ""),
                "kmeans_cluster": int(kmeans_labels[index - 1]),
                "tsne_x": float(tsne_2d[index - 1, 0]),
                "tsne_y": float(tsne_2d[index - 1, 1]),
            }
            for latent_index, value in enumerate(latent_vectors[index - 1]):
                record[f"latent_{latent_index:03d}"] = float(value)
            for factor_name in factor_names:
                record[factor_name] = factor_row.get(factor_name, "")
            writer.writerow(record)


def plot_tsne(tsne_2d: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    autoencoder.plt.figure(figsize=(8, 6))
    autoencoder.plt.scatter(tsne_2d[:, 0], tsne_2d[:, 1], s=18, alpha=0.8)
    autoencoder.plt.xlabel("t-SNE 1")
    autoencoder.plt.ylabel("t-SNE 2")
    autoencoder.plt.title("t-SNE of Structure Latent Vectors")
    autoencoder.plt.grid(True, alpha=0.25)
    autoencoder.plt.tight_layout()
    autoencoder.plt.savefig(output_path, dpi=180)
    autoencoder.plt.close()


def plot_tsne_by_cluster(
    tsne_2d: Any,
    kmeans_labels: Any,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unique_labels = sorted(int(value) for value in autoencoder.np.unique(kmeans_labels))
    color_map = autoencoder.plt.cm.get_cmap("tab20", max(1, len(unique_labels)))
    autoencoder.plt.figure(figsize=(8, 6))
    for color_index, cluster_label in enumerate(unique_labels):
        mask = kmeans_labels == cluster_label
        autoencoder.plt.scatter(
            tsne_2d[mask, 0],
            tsne_2d[mask, 1],
            s=20,
            alpha=0.85,
            color=color_map(color_index),
            label=f"Cluster {cluster_label}",
        )
    autoencoder.plt.xlabel("t-SNE 1")
    autoencoder.plt.ylabel("t-SNE 2")
    autoencoder.plt.title("t-SNE Colored by KMeans Cluster")
    autoencoder.plt.legend(loc="best", fontsize=8)
    autoencoder.plt.grid(True, alpha=0.25)
    autoencoder.plt.tight_layout()
    autoencoder.plt.savefig(output_path, dpi=180)
    autoencoder.plt.close()


def plot_tsne_by_factor(
    tsne_2d: Any,
    values: Any,
    factor_name: str,
    output_path: Path,
) -> bool:
    finite_mask = autoencoder.np.isfinite(values)
    if not bool(autoencoder.np.any(finite_mask)):
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    autoencoder.plt.figure(figsize=(8, 6))
    invalid_mask = autoencoder.np.logical_not(finite_mask)
    if bool(autoencoder.np.any(invalid_mask)):
        autoencoder.plt.scatter(
            tsne_2d[invalid_mask, 0],
            tsne_2d[invalid_mask, 1],
            s=18,
            alpha=0.45,
            color="lightgray",
            label="Missing / N.A.",
        )
    scatter = autoencoder.plt.scatter(
        tsne_2d[finite_mask, 0],
        tsne_2d[finite_mask, 1],
        c=values[finite_mask],
        cmap="viridis",
        s=22,
        alpha=0.9,
    )
    autoencoder.plt.xlabel("t-SNE 1")
    autoencoder.plt.ylabel("t-SNE 2")
    autoencoder.plt.title(f"t-SNE Colored by {factor_name}")
    autoencoder.plt.grid(True, alpha=0.25)
    if bool(autoencoder.np.any(invalid_mask)):
        autoencoder.plt.legend(loc="best", fontsize=8)
    colorbar = autoencoder.plt.colorbar(scatter)
    colorbar.set_label(factor_name)
    autoencoder.plt.tight_layout()
    autoencoder.plt.savefig(output_path, dpi=180)
    autoencoder.plt.close()
    return True


def run_tsne_analysis(args: argparse.Namespace) -> None:
    total_start_time = time.perf_counter()
    if args.checkpoint is None:
        raise ValueError("checkpoint path is required.")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    if args.perplexity <= 0.0:
        raise ValueError("perplexity must be positive.")
    if args.kmeans_clusters <= 0:
        raise ValueError("kmeans_clusters must be positive.")
    if args.tsne_iterations <= 0:
        raise ValueError("tsne_iterations must be positive.")

    output_dir = cast(Path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extraction_start_time = time.perf_counter()
    latent_vectors, file_paths, latent_dim, device_text = extract_latent_vectors(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_files=args.max_files,
        force_cpu=args.cpu,
    )
    extraction_elapsed_seconds = float(time.perf_counter() - extraction_start_time)

    sample_count = int(latent_vectors.shape[0])
    if sample_count < 2:
        raise ValueError("At least 2 samples are required for t-SNE.")

    effective_perplexity = min(float(args.perplexity), float(sample_count - 1))
    if effective_perplexity != float(args.perplexity):
        print(
            f"Perplexity adjusted from {args.perplexity} to {effective_perplexity} "
            f"because only {sample_count} samples were available."
        )

    effective_kmeans_clusters = min(int(args.kmeans_clusters), sample_count)
    if effective_kmeans_clusters != int(args.kmeans_clusters):
        print(
            f"KMeans cluster count adjusted from {args.kmeans_clusters} to "
            f"{effective_kmeans_clusters} because only {sample_count} samples were available."
        )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data root: {args.data_root}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device_text}")
    print(f"Sample count: {sample_count}")
    print(f"Latent dimension: {latent_dim}")
    print(f"t-SNE perplexity: {effective_perplexity}")
    print(f"KMeans clusters: {effective_kmeans_clusters}")
    print(f"t-SNE iterations: {args.tsne_iterations}")
    print(
        "t-SNE progress logging: every 50 iterations in terminal / "
        "t-SNE progress logging: every 50 iterations in terminal"
    )
    print(
        f"Latent extraction elapsed time: "
        f"{autoencoder.format_elapsed_time(extraction_elapsed_seconds)} "
        f"({extraction_elapsed_seconds:.2f} s)"
    )

    tsne_start_time = time.perf_counter()
    tsne = build_tsne(effective_perplexity, args.tsne_iterations, args.seed)
    tsne_2d = tsne.fit_transform(latent_vectors)
    tsne_elapsed_seconds = float(time.perf_counter() - tsne_start_time)
    print(
        f"t-SNE elapsed time: {autoencoder.format_elapsed_time(tsne_elapsed_seconds)} "
        f"({tsne_elapsed_seconds:.2f} s)"
    )

    kmeans_start_time = time.perf_counter()
    kmeans = build_kmeans(effective_kmeans_clusters, args.seed)
    kmeans_labels = kmeans.fit_predict(latent_vectors)
    kmeans_elapsed_seconds = float(time.perf_counter() - kmeans_start_time)
    print(
        f"KMeans elapsed time: {autoencoder.format_elapsed_time(kmeans_elapsed_seconds)} "
        f"({kmeans_elapsed_seconds:.2f} s)"
    )

    structure_factor_start_time = time.perf_counter()
    factor_rows, factor_names, matched_factor_count = load_structure_factor_labels(file_paths)
    structure_factor_elapsed_seconds = float(
        time.perf_counter() - structure_factor_start_time
    )
    print(
        f"Structure factor label loading elapsed time: "
        f"{autoencoder.format_elapsed_time(structure_factor_elapsed_seconds)} "
        f"({structure_factor_elapsed_seconds:.2f} s)"
    )

    latent_npy_path = output_dir / "test_latent_vectors.npy"
    latent_csv_path = output_dir / "test_latent_vectors.csv"
    tsne_npy_path = output_dir / "test_tsne_2d.npy"
    tsne_csv_path = output_dir / "test_tsne_2d.csv"
    tsne_png_path = output_dir / "test_tsne_scatter.png"
    kmeans_npy_path = output_dir / "test_kmeans_labels.npy"
    kmeans_centers_path = output_dir / "test_kmeans_cluster_centers.npy"
    kmeans_png_path = output_dir / "test_tsne_kmeans_clusters.png"
    combined_csv_path = output_dir / "test_latent_tsne_kmeans_structure_factors.csv"
    factor_plot_dir = output_dir / "structure_factor_colored_plots"
    summary_path = output_dir / "test_tsne_summary.json"

    output_start_time = time.perf_counter()
    autoencoder.np.save(latent_npy_path, latent_vectors)
    save_latent_csv(file_paths, latent_vectors, latent_csv_path)
    autoencoder.np.save(tsne_npy_path, tsne_2d)
    save_tsne_csv(file_paths, tsne_2d, tsne_csv_path)
    autoencoder.np.save(kmeans_npy_path, kmeans_labels)
    autoencoder.np.save(kmeans_centers_path, kmeans.cluster_centers_)
    plot_tsne(tsne_2d, tsne_png_path)
    plot_tsne_by_cluster(tsne_2d, kmeans_labels, kmeans_png_path)
    save_combined_analysis_csv(
        file_paths=file_paths,
        latent_vectors=latent_vectors,
        tsne_2d=tsne_2d,
        kmeans_labels=kmeans_labels,
        factor_rows=factor_rows,
        factor_names=factor_names,
        output_path=combined_csv_path,
    )

    generated_factor_plots: List[str] = []
    for factor_name in factor_names:
        if factor_name in {"Input Structure", "Name"}:
            continue
        factor_values = autoencoder.np.asarray(
            [coerce_plot_numeric(row.get(factor_name, "")) for row in factor_rows],
            dtype=autoencoder.np.float64,
        )
        factor_plot_path = factor_plot_dir / (
            f"tsne_color_by_{sanitize_filename(factor_name)}.png"
        )
        if plot_tsne_by_factor(tsne_2d, factor_values, factor_name, factor_plot_path):
            generated_factor_plots.append(str(factor_plot_path))
            print(
                f"[Factor Plot {len(generated_factor_plots):02d}] Generated plot for: {factor_name}"
            )
    output_elapsed_seconds = float(time.perf_counter() - output_start_time)
    total_elapsed_seconds = float(time.perf_counter() - total_start_time)

    summary = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "output_dir": str(output_dir),
        "device": device_text,
        "sample_count": sample_count,
        "latent_dim": latent_dim,
        "perplexity": effective_perplexity,
        "kmeans_clusters": effective_kmeans_clusters,
        "tsne_iterations": args.tsne_iterations,
        "seed": args.seed,
        "matched_structure_factor_files": matched_factor_count,
        "unmatched_structure_factor_files": sample_count - matched_factor_count,
        "extraction_elapsed_seconds": extraction_elapsed_seconds,
        "tsne_elapsed_seconds": tsne_elapsed_seconds,
        "kmeans_elapsed_seconds": kmeans_elapsed_seconds,
        "structure_factor_elapsed_seconds": structure_factor_elapsed_seconds,
        "output_elapsed_seconds": output_elapsed_seconds,
        "total_elapsed_seconds": total_elapsed_seconds,
        "latent_npy": str(latent_npy_path),
        "latent_csv": str(latent_csv_path),
        "tsne_npy": str(tsne_npy_path),
        "tsne_csv": str(tsne_csv_path),
        "tsne_png": str(tsne_png_path),
        "kmeans_npy": str(kmeans_npy_path),
        "kmeans_centers_npy": str(kmeans_centers_path),
        "kmeans_png": str(kmeans_png_path),
        "combined_csv": str(combined_csv_path),
        "structure_factor_plot_count": len(generated_factor_plots),
        "structure_factor_plots": generated_factor_plots,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print("Latent extraction, t-SNE, and KMeans finished.")
    print(f"Latent vectors (NPY): {latent_npy_path}")
    print(f"Latent vectors (CSV): {latent_csv_path}")
    print(f"t-SNE coordinates (NPY): {tsne_npy_path}")
    print(f"t-SNE coordinates (CSV): {tsne_csv_path}")
    print(f"t-SNE scatter plot: {tsne_png_path}")
    print(f"KMeans labels (NPY): {kmeans_npy_path}")
    print(f"KMeans cluster plot: {kmeans_png_path}")
    print(f"Combined analysis CSV: {combined_csv_path}")
    print(
        f"Structure-factor-colored plots generated: {len(generated_factor_plots)}"
    )
    print(
        f"Output writing elapsed time: {autoencoder.format_elapsed_time(output_elapsed_seconds)} "
        f"({output_elapsed_seconds:.2f} s)"
    )
    print(
        f"Total elapsed time: {autoencoder.format_elapsed_time(total_elapsed_seconds)} "
        f"({total_elapsed_seconds:.2f} s)"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args = configure_interactively(args)
    run_tsne_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
