#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict, cast

import latent_tsne_analysis as latent_analysis
import train_structure_autoencoder as autoencoder

try:
    PCA = cast(Any, importlib.import_module("sklearn.decomposition").PCA)
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires scikit-learn. Install it with: pip install scikit-learn"
    ) from exc


DEFAULT_TEST_ROOT = autoencoder.DEFAULT_DATA_ROOT / "test"


class RegressionResultRow(TypedDict):
    factor_index: int
    factor_name: str
    valid_sample_count: int
    slope: float
    intercept: float
    pearson_r: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract latent vectors, perform PCA on the latent space, and fit "
            "linear regressions between PC1 and one or more structure factors."
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
        help=f"Root directory containing structures. Default: {DEFAULT_TEST_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for PCA/regression outputs. Default: <checkpoint_dir>\\test_latent_pca",
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
        help="Optional limit on the number of structures.",
    )
    parser.add_argument(
        "--factor-index",
        type=int,
        help="Optional 1-based index of the single structure factor for regression.",
    )
    parser.add_argument(
        "--regress-all-factors",
        action="store_true",
        help="Run PC1 regression against all selectable structure factors.",
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
            print("请输入有效整数 / Please enter a valid integer.")


def prompt_bool(prompt: str, default: bool) -> bool:
    default_text = "yes" if default else "no"
    true_values = {"y", "yes", "1", "true", "是"}
    false_values = {"n", "no", "0", "false", "否"}
    while True:
        raw = input(
            f"{prompt} [yes(y) | no(n), default={default_text}]: "
        ).strip().lower()
        if not raw:
            return default
        if raw in true_values:
            return True
        if raw in false_values:
            return False
        print("请输入 yes/no 或 y/n / Please enter yes/no or y/n.")


def prompt_choice(prompt: str, default: str, valid_choices: Sequence[str]) -> str:
    valid_choice_set = {choice.lower() for choice in valid_choices}
    while True:
        raw = input(f"{prompt} [{default}]: ").strip().lower()
        value = raw if raw else default.lower()
        if value in valid_choice_set:
            return value
        print(f"请输入以下之一 / Please enter one of: {', '.join(valid_choices)}")


def parse_optional_int_text(raw_text: str) -> Optional[int]:
    text = raw_text.strip()
    if not text:
        return None
    return int(text)


def default_output_dir_for_args(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return cast(Path, args.output_dir)
    if args.checkpoint is None:
        return Path("test_latent_pca")
    return args.checkpoint.parent / "test_latent_pca"


def configure_interactively(args: argparse.Namespace) -> argparse.Namespace:
    if args.checkpoint is None:
        print("进入隐向量 PCA 与 PC1 回归设置 / Configure latent PCA and PC1 regression.")
        checkpoint_text = prompt_text("请输入 checkpoint 路径 / Enter checkpoint path")
        if not checkpoint_text:
            raise ValueError("checkpoint 路径不能为空 / checkpoint path is required.")
        args.checkpoint = Path(checkpoint_text)
        args.data_root = Path(
            prompt_text("结构数据目录 / Structure data root", str(args.data_root))
        )
        args.output_dir = Path(
            prompt_text(
                "输出目录 / Output directory",
                str(default_output_dir_for_args(args)),
            )
        )
        args.batch_size = prompt_int("批大小 / Batch size", int(args.batch_size))
        args.num_workers = prompt_int(
            "DataLoader 进程数 / DataLoader workers", int(args.num_workers)
        )
        max_files_default = "" if args.max_files is None else str(args.max_files)
        max_files_text = prompt_text(
            "最大结构数量，回车表示全部 / Maximum number of structures, blank for all",
            max_files_default,
        )
        args.max_files = (
            parse_optional_int_text(max_files_text)
            if max_files_text.strip()
            else None
        )
        args.seed = prompt_int("随机种子 / Random seed", int(args.seed))
        args.cpu = prompt_bool("是否强制使用 CPU / Force CPU only", bool(args.cpu))

    if args.output_dir is None:
        args.output_dir = default_output_dir_for_args(args)
    return args


def coerce_numeric_series(
    factor_rows: Sequence[Dict[str, object]],
    factor_name: str,
) -> Any:
    values = [
        latent_analysis.coerce_plot_numeric(row.get(factor_name, ""))
        for row in factor_rows
    ]
    return autoencoder.np.asarray(values, dtype=autoencoder.np.float64)


def list_selectable_factors(
    factor_rows: Sequence[Dict[str, object]],
    factor_names: Sequence[str],
) -> List[Tuple[str, int]]:
    selectable: List[Tuple[str, int]] = []
    excluded_names = {"Input Structure"}
    for factor_name in factor_names:
        if factor_name in excluded_names:
            continue
        numeric_values = coerce_numeric_series(factor_rows, factor_name)
        finite_mask = autoencoder.np.isfinite(numeric_values)
        finite_count = int(finite_mask.sum())
        if finite_count < 2:
            continue
        finite_values = numeric_values[finite_mask]
        if float(autoencoder.np.std(finite_values)) <= 0.0:
            continue
        selectable.append((factor_name, finite_count))
    return selectable


def prompt_regression_mode() -> str:
    print("回归分析模式 / Regression mode:")
    print("1. 单个结构因子 / Single structure factor")
    print("2. 所有结构因子 / All structure factors")
    choice = prompt_choice(
        "请选择回归模式 / Choose regression mode",
        "1",
        ("1", "2", "single", "all"),
    )
    return "all" if choice in {"2", "all"} else "single"


def prompt_regression_factor_index(
    selectable_factors: Sequence[Tuple[str, int]],
) -> int:
    if not selectable_factors:
        raise ValueError("No selectable numeric structure factors were found.")
    print("可回归的结构因子 / Selectable structure factors:")
    for index, (factor_name, finite_count) in enumerate(selectable_factors, start=1):
        print(f"{index}. {factor_name} (valid samples: {finite_count})")
    while True:
        raw = input("请输入结构因子编号 / Enter factor number: ").strip()
        if not raw:
            print("编号不能为空 / Factor number is required.")
            continue
        try:
            selected_index = int(raw)
        except ValueError:
            print("请输入有效整数 / Please enter a valid integer.")
            continue
        if 1 <= selected_index <= len(selectable_factors):
            return selected_index
        print("编号超出范围 / Factor number is out of range.")


def build_pca(seed: int, n_components: int) -> Any:
    common_kwargs: Dict[str, Any] = {
        "n_components": n_components,
        "random_state": seed,
    }
    try:
        return PCA(svd_solver="auto", **common_kwargs)
    except TypeError:
        return PCA(**common_kwargs)


def fit_pca_scores(latent_vectors: Any, seed: int) -> Tuple[Any, Any, Any, int]:
    sample_count = int(latent_vectors.shape[0])
    latent_dim = int(latent_vectors.shape[1])
    if sample_count < 2:
        raise ValueError("At least 2 samples are required for PCA.")
    if latent_dim <= 0:
        raise ValueError("Latent vectors must contain at least one feature.")

    component_count = min(2, sample_count, latent_dim)
    pca = build_pca(seed, component_count)
    scores = pca.fit_transform(latent_vectors)

    full_scores = autoencoder.np.full(
        (sample_count, 2),
        autoencoder.np.nan,
        dtype=autoencoder.np.float64,
    )
    full_scores[:, :component_count] = scores

    explained_variance_ratio = autoencoder.np.full(
        2,
        autoencoder.np.nan,
        dtype=autoencoder.np.float64,
    )
    explained_variance_ratio[:component_count] = pca.explained_variance_ratio_
    return pca, full_scores, explained_variance_ratio, component_count


def fit_linear_regression(x_values: Any, y_values: Any) -> Tuple[float, float]:
    x_matrix = autoencoder.np.column_stack(
        [x_values, autoencoder.np.ones_like(x_values)]
    )
    solution, _, _, _ = autoencoder.np.linalg.lstsq(x_matrix, y_values, rcond=None)
    slope = float(solution[0])
    intercept = float(solution[1])
    return slope, intercept


def compute_pearson_r(x_values: Any, y_values: Any) -> float:
    if int(x_values.size) < 2:
        return float("nan")
    x_std = float(autoencoder.np.std(x_values))
    y_std = float(autoencoder.np.std(y_values))
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    correlation_matrix = autoencoder.np.corrcoef(x_values, y_values)
    return float(correlation_matrix[0, 1])


def save_pca_scores_csv(
    file_paths: Sequence[Path],
    pca_scores: Any,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_index",
                "file_name",
                "file_path",
                "pc1",
                "pc2",
            ],
        )
        writer.writeheader()
        for index, path in enumerate(file_paths, start=1):
            pc1_value = pca_scores[index - 1, 0]
            pc2_value = pca_scores[index - 1, 1]
            writer.writerow(
                {
                    "sample_index": index,
                    "file_name": path.name,
                    "file_path": str(path),
                    "pc1": float(pc1_value)
                    if bool(autoencoder.np.isfinite(pc1_value))
                    else "",
                    "pc2": float(pc2_value)
                    if bool(autoencoder.np.isfinite(pc2_value))
                    else "",
                }
            )


def save_regression_summary_csv(
    regression_rows: Sequence[RegressionResultRow],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "factor_index",
        "factor_name",
        "valid_sample_count",
        "slope",
        "intercept",
        "pearson_r",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in regression_rows:
            writer.writerow(row)


def plot_pc1_regression(
    x_values: Any,
    y_values: Any,
    factor_name: str,
    slope: float,
    intercept: float,
    pearson_r: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sort_indices = autoencoder.np.argsort(x_values)
    x_sorted = x_values[sort_indices]
    regression_y = slope * x_sorted + intercept

    autoencoder.plt.figure(figsize=(8, 6))
    autoencoder.plt.scatter(x_values, y_values, s=24, alpha=0.85, label="Samples")
    autoencoder.plt.plot(
        x_sorted,
        regression_y,
        color="crimson",
        linewidth=2.0,
        label=f"Linear fit: y = {slope:.4f}x + {intercept:.4f}",
    )
    autoencoder.plt.xlabel(factor_name)
    autoencoder.plt.ylabel("PC1")
    autoencoder.plt.title(f"PC1 vs {factor_name}\nPearson r = {pearson_r:.4f}")
    autoencoder.plt.grid(True, alpha=0.25)
    autoencoder.plt.legend(loc="best", fontsize=8)
    autoencoder.plt.tight_layout()
    autoencoder.plt.savefig(output_path, dpi=180)
    autoencoder.plt.close()


def run_analysis(args: argparse.Namespace) -> None:
    total_start_time = time.perf_counter()
    if args.checkpoint is None:
        raise ValueError("checkpoint path is required.")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    output_dir = cast(Path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extraction_start_time = time.perf_counter()
    latent_vectors, file_paths, latent_dim, device_text = latent_analysis.extract_latent_vectors(
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
        raise ValueError("At least 2 samples are required for PCA regression.")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Structure data root: {args.data_root}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device_text}")
    print(f"Sample count: {sample_count}")
    print(f"Latent dimension: {latent_dim}")
    print(
        f"Latent extraction elapsed time: "
        f"{autoencoder.format_elapsed_time(extraction_elapsed_seconds)} "
        f"({extraction_elapsed_seconds:.2f} s)"
    )

    factor_start_time = time.perf_counter()
    factor_rows, factor_names, matched_factor_count = latent_analysis.load_structure_factor_labels(
        file_paths
    )
    factor_elapsed_seconds = float(time.perf_counter() - factor_start_time)
    print(
        f"Structure factor loading elapsed time: "
        f"{autoencoder.format_elapsed_time(factor_elapsed_seconds)} "
        f"({factor_elapsed_seconds:.2f} s)"
    )

    selectable_factors = list_selectable_factors(factor_rows, factor_names)
    if not selectable_factors:
        raise ValueError("No numeric structure factors are available for regression.")

    if args.regress_all_factors and args.factor_index is not None:
        raise ValueError(
            "Use either --regress-all-factors or --factor-index, not both."
        )

    if args.regress_all_factors:
        regression_factor_indices = list(range(1, len(selectable_factors) + 1))
    elif args.factor_index is not None:
        regression_factor_index = int(args.factor_index)
        if regression_factor_index < 1 or regression_factor_index > len(selectable_factors):
            raise ValueError(
                f"factor_index must be between 1 and {len(selectable_factors)}."
            )
        regression_factor_indices = [regression_factor_index]
    else:
        regression_mode = prompt_regression_mode()
        if regression_mode == "all":
            regression_factor_indices = list(range(1, len(selectable_factors) + 1))
        else:
            regression_factor_indices = [
                prompt_regression_factor_index(selectable_factors)
            ]

    selected_factor_names = [
        selectable_factors[index - 1][0] for index in regression_factor_indices
    ]

    pca_start_time = time.perf_counter()
    _, pca_scores, explained_variance_ratio, component_count = fit_pca_scores(
        latent_vectors=latent_vectors,
        seed=args.seed,
    )
    pca_elapsed_seconds = float(time.perf_counter() - pca_start_time)
    pc1_values = pca_scores[:, 0]

    print(
        f"PCA elapsed time: {autoencoder.format_elapsed_time(pca_elapsed_seconds)} "
        f"({pca_elapsed_seconds:.2f} s)"
    )
    print(f"PC1 explained variance ratio: {float(explained_variance_ratio[0]):.6f}")
    if component_count >= 2:
        print(f"PC2 explained variance ratio: {float(explained_variance_ratio[1]):.6f}")
    else:
        print("PC2 explained variance ratio: N/A (only one principal component)")

    regression_start_time = time.perf_counter()
    regression_rows: List[RegressionResultRow] = []
    regression_plot_dir = output_dir / "pc1_structure_factor_regression_plots"

    for factor_index in regression_factor_indices:
        factor_name = selectable_factors[factor_index - 1][0]
        factor_values = coerce_numeric_series(factor_rows, factor_name)
        valid_mask = autoencoder.np.isfinite(pc1_values) & autoencoder.np.isfinite(factor_values)
        valid_sample_count = int(valid_mask.sum())
        if valid_sample_count < 2:
            continue
        x_values = factor_values[valid_mask]
        y_values = pc1_values[valid_mask]
        slope, intercept = fit_linear_regression(x_values, y_values)
        pearson_r = compute_pearson_r(x_values, y_values)
        regression_rows.append(
            {
                "factor_index": factor_index,
                "factor_name": factor_name,
                "valid_sample_count": valid_sample_count,
                "slope": slope,
                "intercept": intercept,
                "pearson_r": pearson_r,
            }
        )

        if len(regression_factor_indices) == 1:
            plot_path = output_dir / "pc1_structure_factor_regression.png"
        else:
            plot_path = (
                regression_plot_dir
                / f"{latent_analysis.sanitize_filename(factor_name)}.png"
            )
        plot_pc1_regression(
            x_values=x_values,
            y_values=y_values,
            factor_name=factor_name,
            slope=slope,
            intercept=intercept,
            pearson_r=pearson_r,
            output_path=plot_path,
        )

    regression_elapsed_seconds = float(time.perf_counter() - regression_start_time)
    if not regression_rows:
        raise ValueError("No valid regression results could be computed.")

    print(
        f"Linear regression elapsed time: "
        f"{autoencoder.format_elapsed_time(regression_elapsed_seconds)} "
        f"({regression_elapsed_seconds:.4f} s)"
    )
    if len(regression_rows) == 1:
        row = regression_rows[0]
        print(
            f"Selected regression factor: {row['factor_name']} "
            f"(valid samples: {row['valid_sample_count']}/{sample_count})"
        )
        print(
            f"Linear fit: PC1 = {float(row['slope']):.6f} * x + "
            f"{float(row['intercept']):.6f}"
        )
        print(f"Pearson correlation coefficient: {float(row['pearson_r']):.6f}")
    else:
        print(f"Regression factor count: {len(regression_rows)}")

    pca_npy_path = output_dir / "test_pca_2d.npy"
    pca_csv_path = output_dir / "test_pca_2d.csv"
    regression_csv_path = output_dir / "pc1_structure_factor_regressions.csv"
    summary_path = output_dir / "pca_regression_summary.json"

    output_start_time = time.perf_counter()
    autoencoder.np.save(pca_npy_path, pca_scores)
    save_pca_scores_csv(
        file_paths=file_paths,
        pca_scores=pca_scores,
        output_path=pca_csv_path,
    )
    save_regression_summary_csv(
        regression_rows=regression_rows,
        output_path=regression_csv_path,
    )
    output_elapsed_seconds = float(time.perf_counter() - output_start_time)

    total_elapsed_seconds = float(time.perf_counter() - total_start_time)
    summary = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "output_directory": str(output_dir),
        "sample_count": sample_count,
        "matched_structure_factor_count": matched_factor_count,
        "latent_dimension": latent_dim,
        "regression_factor_count": len(regression_rows),
        "regression_factor_names": [row["factor_name"] for row in regression_rows],
        "pc1_explained_variance_ratio": float(explained_variance_ratio[0]),
        "pc2_explained_variance_ratio": (
            float(explained_variance_ratio[1])
            if bool(autoencoder.np.isfinite(explained_variance_ratio[1]))
            else None
        ),
        "regression_results": regression_rows,
        "latent_extraction_elapsed_seconds": extraction_elapsed_seconds,
        "structure_factor_elapsed_seconds": factor_elapsed_seconds,
        "pca_elapsed_seconds": pca_elapsed_seconds,
        "regression_elapsed_seconds": regression_elapsed_seconds,
        "output_elapsed_seconds": output_elapsed_seconds,
        "total_elapsed_seconds": total_elapsed_seconds,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(
        f"Output writing elapsed time: "
        f"{autoencoder.format_elapsed_time(output_elapsed_seconds)} "
        f"({output_elapsed_seconds:.2f} s)"
    )
    print(
        f"Total elapsed time: {autoencoder.format_elapsed_time(total_elapsed_seconds)} "
        f"({total_elapsed_seconds:.2f} s)"
    )
    print(f"PCA coordinates: {pca_npy_path}")
    print(f"PCA table: {pca_csv_path}")
    print(f"Regression summary CSV: {regression_csv_path}")
    if len(regression_rows) == 1:
        print(f"Regression plot: {output_dir / 'pc1_structure_factor_regression.png'}")
    else:
        print(f"Regression plot directory: {regression_plot_dir}")
    print(f"Summary JSON: {summary_path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args = configure_interactively(args)
    run_analysis(args)


if __name__ == "__main__":
    main()
