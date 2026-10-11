#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def configure_cache_env() -> None:
    tmp_root = Path(tempfile.gettempdir()) / "electrode_kit_eval_cache"
    mpl_dir = tmp_root / "mpl"
    xdg_dir = tmp_root / "xdg-cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_dir))


configure_cache_env()

import numpy as np

import train_structure_autoencoder as autoencoder


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multitask factor regression quality on test set using "
            "a trained structure autoencoder checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to structure_autoencoder_checkpoint.pt",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        required=True,
        help="Test split directory containing *_phase_channels.npy + *_structure_factors.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for evaluation. Default: 2",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Default: 0",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Optional cap on number of test files to evaluate.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU evaluation.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output JSON path. Default: <checkpoint_dir>/test_factor_metrics.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Output CSV path. Default: <checkpoint_dir>/test_factor_metrics.csv",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable regression plot export.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Plot output directory. Default: <checkpoint_dir>/test_factor_plots",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=180,
        help="DPI for plot export. Default: 180",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.test_root.exists():
        raise FileNotFoundError(f"Test root not found: {args.test_root}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative.")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("max-files must be positive when provided.")
    if args.plot_dpi <= 0:
        raise ValueError("plot-dpi must be positive.")


def load_model_and_stats(
    checkpoint_path: Path,
    device: Any,
) -> Tuple[Any, Tuple[str, ...], Any, Any]:
    checkpoint = autoencoder.load_checkpoint(checkpoint_path, device)
    latent_dim = int(checkpoint.get("latent_dim", 128))
    factor_names_raw = checkpoint.get("factor_names")
    if not isinstance(factor_names_raw, list) or not factor_names_raw:
        raise ValueError("Checkpoint is missing valid 'factor_names'.")
    factor_names = tuple(str(name) for name in factor_names_raw)

    factor_mean_raw = checkpoint.get("factor_mean")
    factor_std_raw = checkpoint.get("factor_std")
    if (
        not isinstance(factor_mean_raw, list)
        or not isinstance(factor_std_raw, list)
        or len(factor_mean_raw) != len(factor_names)
        or len(factor_std_raw) != len(factor_names)
    ):
        raise ValueError("Checkpoint is missing valid 'factor_mean'/'factor_std'.")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing 'model_state_dict'.")

    model = autoencoder.StructureAutoencoder(
        latent_dim=latent_dim,
        factor_output_dim=len(factor_names),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    factor_mean_tensor = autoencoder.torch.from_numpy(
        np.asarray(factor_mean_raw, dtype=np.float32)
    ).to(device)
    factor_std_tensor = autoencoder.torch.from_numpy(
        np.asarray(factor_std_raw, dtype=np.float32)
    ).to(device)
    return model, factor_names, factor_mean_tensor, factor_std_tensor


def evaluate_metrics(
    model: Any,
    data_loader: Any,
    device: Any,
    factor_mean_tensor: Any,
    factor_std_tensor: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true_batches: List[np.ndarray] = []
    y_pred_batches: List[np.ndarray] = []

    with autoencoder.torch.no_grad():
        for inputs, factor_targets in data_loader:
            inputs = inputs.to(device, non_blocking=(device.type == "cuda"))
            factor_targets = factor_targets.to(
                device,
                non_blocking=(device.type == "cuda"),
            )
            _, factor_predictions_norm, _ = model(inputs)
            factor_predictions = (
                factor_predictions_norm * factor_std_tensor + factor_mean_tensor
            )
            y_true_batches.append(
                factor_targets.detach().cpu().numpy().astype(np.float64, copy=False)
            )
            y_pred_batches.append(
                factor_predictions.detach().cpu().numpy().astype(np.float64, copy=False)
            )

    if not y_true_batches:
        raise RuntimeError("No batches were evaluated. Check test-root contents.")

    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)
    return y_true, y_pred


def build_metric_payload(
    factor_names: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    if y_true.shape != y_pred.shape:
        raise ValueError("Prediction and target arrays have mismatched shapes.")
    if y_true.ndim != 2:
        raise ValueError("Prediction and target arrays must be 2D.")

    error = y_pred - y_true
    mae_each = np.mean(np.abs(error), axis=0)
    rmse_each = np.sqrt(np.mean(np.square(error), axis=0))

    ss_res = np.sum(np.square(error), axis=0)
    y_mean = np.mean(y_true, axis=0, keepdims=True)
    ss_tot = np.sum(np.square(y_true - y_mean), axis=0)
    r2_each = np.where(ss_tot > 1.0e-12, 1.0 - (ss_res / ss_tot), np.nan)

    per_factor: List[Dict[str, Any]] = []
    for index, factor_name in enumerate(factor_names):
        per_factor.append(
            {
                "name": str(factor_name),
                "mae": float(mae_each[index]),
                "rmse": float(rmse_each[index]),
                "r2": None if np.isnan(r2_each[index]) else float(r2_each[index]),
                "target_mean": float(np.mean(y_true[:, index])),
                "target_std": float(np.std(y_true[:, index], ddof=0)),
            }
        )

    valid_r2_mask = np.isfinite(r2_each)
    overall_r2 = (
        float(np.mean(r2_each[valid_r2_mask]))
        if np.any(valid_r2_mask)
        else None
    )

    return {
        "sample_count": int(y_true.shape[0]),
        "factor_count": int(y_true.shape[1]),
        "overall_mae": float(np.mean(mae_each)),
        "overall_rmse": float(np.mean(rmse_each)),
        "overall_r2_mean": overall_r2,
        "per_factor": per_factor,
    }


def write_outputs(
    payload: Dict[str, Any],
    json_path: Path,
    csv_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["factor_name", "mae", "rmse", "r2", "target_mean", "target_std"])
        for row in payload["per_factor"]:
            writer.writerow(
                [
                    row["name"],
                    f"{row['mae']:.12g}",
                    f"{row['rmse']:.12g}",
                    "" if row["r2"] is None else f"{row['r2']:.12g}",
                    f"{row['target_mean']:.12g}",
                    f"{row['target_std']:.12g}",
                ]
            )


def sanitize_plot_stem(text: str) -> str:
    safe = []
    for char in str(text):
        if char.isalnum() or char in {"_", "-", "."}:
            safe.append(char)
        else:
            safe.append("_")
    stem = "".join(safe).strip("._")
    return stem or "factor"


def save_regression_plots(
    factor_names: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics_payload: Dict[str, Any],
    plot_dir: Path,
    dpi: int,
) -> Dict[str, Any]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    per_factor_metrics = metrics_payload["per_factor"]

    rows = int(np.ceil(len(factor_names) / 2.0))
    cols = 2
    figure, axes = autoencoder.plt.subplots(
        rows,
        cols,
        figsize=(7.5 * cols, 4.5 * rows),
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes_array = np.asarray([axes], dtype=object)
    else:
        axes_array = axes.reshape(-1)

    single_plot_paths: List[str] = []

    for index, factor_name in enumerate(factor_names):
        axis = axes_array[index]
        x = y_true[:, index].astype(np.float64, copy=False)
        y = y_pred[:, index].astype(np.float64, copy=False)
        metric_row = per_factor_metrics[index]
        r2_value = metric_row["r2"]
        r2_text = "nan" if r2_value is None else f"{float(r2_value):.4f}"

        axis.scatter(x, y, s=14, alpha=0.60, color="#2a6fbb", edgecolors="none")

        low = float(min(np.min(x), np.min(y)))
        high = float(max(np.max(x), np.max(y)))
        span = max(1.0e-9, high - low)
        low -= 0.05 * span
        high += 0.05 * span
        ref_line = np.linspace(low, high, 200)

        fit_label = "fit unavailable"
        has_fit = False
        if x.size >= 2 and np.std(x) > 1.0e-12:
            slope, intercept = np.polyfit(x, y, 1)
            fit_line = slope * ref_line + intercept
            axis.plot(ref_line, fit_line, linewidth=1.5, color="#d94841", label="fit")
            fit_label = f"y={slope:.3f}x+{intercept:.3f}"
            has_fit = True

        axis.set_title(f"{factor_name}\nR2={r2_text} | MAE={metric_row['mae']:.4g}")
        axis.set_xlabel("True")
        axis.set_ylabel("Predicted")
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.grid(alpha=0.25, linestyle=":")
        axis.text(
            0.02,
            0.98,
            fit_label,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        if has_fit:
            axis.legend(loc="lower right", fontsize=8)

        single_fig, single_axis = autoencoder.plt.subplots(figsize=(6.0, 5.0))
        single_axis.scatter(
            x, y, s=18, alpha=0.60, color="#2a6fbb", edgecolors="none"
        )
        single_has_fit = False
        if x.size >= 2 and np.std(x) > 1.0e-12:
            slope, intercept = np.polyfit(x, y, 1)
            single_axis.plot(
                ref_line,
                slope * ref_line + intercept,
                linewidth=1.6,
                color="#d94841",
                label=f"fit: y={slope:.3f}x+{intercept:.3f}",
            )
            single_has_fit = True
        single_axis.set_title(f"{factor_name} | R2={r2_text} | MAE={metric_row['mae']:.4g}")
        single_axis.set_xlabel("True")
        single_axis.set_ylabel("Predicted")
        single_axis.set_xlim(low, high)
        single_axis.set_ylim(low, high)
        single_axis.grid(alpha=0.25, linestyle=":")
        if single_has_fit:
            single_axis.legend(loc="lower right", fontsize=8)
        single_path = plot_dir / f"{index + 1:02d}_{sanitize_plot_stem(str(factor_name))}_regression.png"
        single_fig.savefig(single_path, dpi=dpi)
        autoencoder.plt.close(single_fig)
        single_plot_paths.append(str(single_path))

    for extra_index in range(len(factor_names), len(axes_array)):
        axes_array[extra_index].axis("off")

    summary_path = plot_dir / "all_factors_regression.png"
    figure.savefig(summary_path, dpi=dpi)
    autoencoder.plt.close(figure)

    return {
        "plot_dir": str(plot_dir),
        "summary_plot": str(summary_path),
        "single_plots": single_plot_paths,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    if autoencoder.TORCH_IMPORT_ERROR is not None:
        raise SystemExit(
            "PyTorch is required but unavailable in current environment.\n"
            "Install first: pip install torch\n"
            f"Original error: {autoencoder.TORCH_IMPORT_ERROR}"
        )

    args = parse_args(argv)
    validate_args(args)
    device = autoencoder.select_device(args.cpu)

    test_dataset = autoencoder.PhaseChannelDataset(
        args.test_root,
        max_files=args.max_files,
    )
    test_loader = autoencoder.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model, factor_names, factor_mean_tensor, factor_std_tensor = load_model_and_stats(
        args.checkpoint,
        device,
    )

    dataset_factor_names = tuple(test_dataset.factor_names)
    if dataset_factor_names != factor_names:
        raise ValueError(
            "Checkpoint factor names do not match test dataset factor names.\n"
            f"Checkpoint: {factor_names}\n"
            f"Dataset: {dataset_factor_names}"
        )

    y_true, y_pred = evaluate_metrics(
        model,
        test_loader,
        device,
        factor_mean_tensor,
        factor_std_tensor,
    )
    payload = build_metric_payload(factor_names, y_true, y_pred)
    payload["checkpoint"] = str(args.checkpoint)
    payload["test_root"] = str(args.test_root)
    payload["device"] = str(device)

    plot_info: Optional[Dict[str, Any]] = None
    if not args.no_plot:
        default_plot_dir = args.checkpoint.parent / "test_factor_plots"
        plot_dir = args.plot_dir if args.plot_dir is not None else default_plot_dir
        plot_info = save_regression_plots(
            factor_names=factor_names,
            y_true=y_true,
            y_pred=y_pred,
            metrics_payload=payload,
            plot_dir=plot_dir,
            dpi=args.plot_dpi,
        )
        payload["regression_plots"] = plot_info

    default_json = args.checkpoint.parent / "test_factor_metrics.json"
    default_csv = args.checkpoint.parent / "test_factor_metrics.csv"
    output_json = args.output_json if args.output_json is not None else default_json
    output_csv = args.output_csv if args.output_csv is not None else default_csv
    write_outputs(payload, output_json, output_csv)

    overall_r2_text = (
        "nan"
        if payload["overall_r2_mean"] is None
        else f"{float(payload['overall_r2_mean']):.6f}"
    )
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test root: {args.test_root}")
    print(f"Samples: {payload['sample_count']}")
    print(
        "Overall MAE/RMSE/R2(mean): "
        f"{payload['overall_mae']:.6f}/"
        f"{payload['overall_rmse']:.6f}/"
        f"{overall_r2_text}"
    )
    print(f"Metrics JSON: {output_json}")
    print(f"Metrics CSV:  {output_csv}")
    if plot_info is not None:
        print(f"Regression plots: {plot_info['plot_dir']}")
        print(f"Regression summary: {plot_info['summary_plot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
