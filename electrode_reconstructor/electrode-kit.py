
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

import electrode_structure_factors as structure_factors
import latent_pca_regression_analysis as latent_pca
import latent_tsne_analysis as latent_tsne
import train_structure_autoencoder as autoencoder
from electrode_reconstruction_pkg.reconstructor import run_random_batch_reconstruction
from electrode_reconstruction_pkg.shared import (
    DEFAULT_OUTPUT_BASE_DIR,
    RandomBatchConfig,
    default_random_seed,
)


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


def prompt_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("请输入有效数值 / Please enter a valid number.")


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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full integrated workflow: generate train/verify/test sets, "
            "extract split-wise structure factors, train the multitask autoencoder, and "
            "analyze correlations with t-SNE and PCA."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help=(
            "Shared dataset root directory. The script will place generated "
            "structures into the train, verify, and test subdirectories here."
        ),
    )
    parser.add_argument("--train-count", type=int, default=1600)
    parser.add_argument("--verify-count", type=int, default=200)
    parser.add_argument("--test-count", type=int, default=200)
    parser.add_argument("--batch-seed", type=int, default=default_random_seed())
    parser.add_argument(
        "--training-output-root",
        type=Path,
        default=autoencoder.DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260410)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-kmeans-clusters", type=int, default=5)
    parser.add_argument("--tsne-iterations", type=int, default=1000)
    parser.add_argument(
        "--pipeline-stage",
        choices=("full", "cpu", "gpu"),
        default="full",
        help=(
            "Pipeline stage selector: 'full' runs generation+factors+training+analysis; "
            "'cpu' runs generation+factor extraction only; "
            "'gpu' runs training+t-SNE+PCA only on existing datasets."
        ),
    )
    parser.add_argument(
        "--recompute-existing-factors",
        action="store_true",
        help=(
            "Recompute structure factors even when target output files already exist. "
            "By default, existing factor files are skipped."
        ),
    )
    parser.add_argument(
        "--analysis-max-files",
        type=int,
        help="Optional limit on the number of test structures used for t-SNE/PCA analysis.",
    )
    structure_factors.add_constrictivity_arguments(parser)
    structure_factors.add_distribution_arguments(parser)
    return parser.parse_args(argv)


def _legacy_configure_interactively_unused(args: argparse.Namespace) -> argparse.Namespace:
    if args.dataset_root is None:
        default_dataset_root = DEFAULT_OUTPUT_BASE_DIR
        print("开始整合流程配置 / Configure integrated structure-ML pipeline.")
        args.dataset_root = Path(
            prompt_text(
                "数据集输出根目录 / Dataset output root directory",
                str(default_dataset_root),
            )
        )
        args.train_count = prompt_int(
            "训练集新增结构数量（0 表示跳过本次生成） / New train structures to generate (0 means skip this split)",
            int(args.train_count),
        )
        args.verify_count = prompt_int(
            "验证集新增结构数量（0 表示跳过本次生成） / New verify structures to generate (0 means skip this split)",
            int(args.verify_count),
        )
        args.test_count = prompt_int(
            "测试集新增结构数量（0 表示跳过本次生成） / New test structures to generate (0 means skip this split)",
            int(args.test_count),
        )
        args.batch_seed = prompt_int(
            "批量随机种子 / Batch random seed",
            int(args.batch_seed),
        )
        args.training_output_root = Path(
            prompt_text(
                "模型训练输出根目录 / Training output root",
                str(args.training_output_root),
            )
        )
        args.epochs = prompt_int("训练轮数 / Epochs", int(args.epochs))
        args.batch_size = prompt_int("批大小 / Batch size", int(args.batch_size))
        args.learning_rate = prompt_float(
            "学习率 / Learning rate", float(args.learning_rate)
        )
        args.latent_dim = prompt_int(
            "隐向量维度 / Latent dimension", int(args.latent_dim)
        )
        args.report_every = prompt_int(
            "误差输出间隔（结构数） / Report every N structures",
            int(args.report_every),
        )
        args.num_workers = prompt_int(
            "DataLoader 进程数 / DataLoader workers", int(args.num_workers)
        )
        args.tsne_perplexity = prompt_float(
            "t-SNE perplexity", float(args.tsne_perplexity)
        )
        args.tsne_kmeans_clusters = prompt_int(
            "t-SNE KMeans 聚类数 / t-SNE KMeans clusters",
            int(args.tsne_kmeans_clusters),
        )
        args.tsne_iterations = prompt_int(
            "t-SNE 迭代次数 / t-SNE iterations", int(args.tsne_iterations)
        )
        analysis_max_files_default = (
            "" if args.analysis_max_files is None else str(args.analysis_max_files)
        )
        analysis_max_files_text = prompt_text(
            "分析时测试结构数量上限，回车表示全部 / Analysis max test files, blank for all",
            analysis_max_files_default,
        )
        args.analysis_max_files = (
            int(analysis_max_files_text)
            if analysis_max_files_text.strip()
            else None
        )
        args.seed = prompt_int("训练与分析随机种子 / Training-analysis seed", int(args.seed))
        args.cpu = prompt_bool("是否强制使用 CPU / Force CPU only", bool(args.cpu))
    return args


def configure_interactively(args: argparse.Namespace) -> argparse.Namespace:
    if args.dataset_root is None:
        default_dataset_root = DEFAULT_OUTPUT_BASE_DIR
        print("开始整合流程配置 / Configure integrated structure-ML pipeline.")
        args.dataset_root = Path(
            prompt_text(
                "数据集共享根目录（结构将累积到 train/verify/test 子目录） / Shared dataset root directory (structures accumulate in train/verify/test)",
                str(default_dataset_root),
            )
        )
        args.train_count = prompt_int(
            "训练集结构数量 / Train structure count",
            int(args.train_count),
        )
        args.verify_count = prompt_int(
            "验证集结构数量 / Verify structure count",
            int(args.verify_count),
        )
        args.test_count = prompt_int(
            "测试集结构数量 / Test structure count",
            int(args.test_count),
        )
        args.batch_seed = prompt_int(
            "批量随机种子 / Batch random seed",
            int(args.batch_seed),
        )
        args.training_output_root = Path(
            prompt_text(
                "模型训练输出根目录 / Training output root",
                str(args.training_output_root),
            )
        )
        args.epochs = prompt_int("训练轮数 / Epochs", int(args.epochs))
        args.batch_size = prompt_int("批大小 / Batch size", int(args.batch_size))
        args.learning_rate = prompt_float(
            "学习率 / Learning rate", float(args.learning_rate)
        )
        args.latent_dim = prompt_int(
            "隐向量维度 / Latent dimension", int(args.latent_dim)
        )
        args.report_every = prompt_int(
            "误差输出间隔（结构数） / Report every N structures",
            int(args.report_every),
        )
        args.num_workers = prompt_int(
            "DataLoader 进程数 / DataLoader workers", int(args.num_workers)
        )
        args.tsne_perplexity = prompt_float(
            "t-SNE perplexity", float(args.tsne_perplexity)
        )
        args.tsne_kmeans_clusters = prompt_int(
            "t-SNE KMeans 聚类数 / t-SNE KMeans clusters",
            int(args.tsne_kmeans_clusters),
        )
        args.tsne_iterations = prompt_int(
            "t-SNE 迭代次数 / t-SNE iterations", int(args.tsne_iterations)
        )
        analysis_max_files_default = (
            "" if args.analysis_max_files is None else str(args.analysis_max_files)
        )
        analysis_max_files_text = prompt_text(
            "分析时测试结构数量上限，回车表示全部 / Analysis max test files, blank for all",
            analysis_max_files_default,
        )
        args.analysis_max_files = (
            int(analysis_max_files_text)
            if analysis_max_files_text.strip()
            else None
        )
        args.seed = prompt_int(
            "训练与分析随机种子 / Training-analysis seed",
            int(args.seed),
        )
        args.cpu = prompt_bool("是否强制使用 CPU / Force CPU only", bool(args.cpu))
    return args


def build_split_batch_config(
    structure_count: int,
    split_seed: int,
    output_prefix_base: str,
) -> RandomBatchConfig:
    return RandomBatchConfig(
        structure_count=structure_count,
        grid_shape_vox=(64, 64, 64),
        voxel_size_um=0.5,
        max_major_axis_um=10.0,
        per_structure_timeout_s=30.0,
        electrode_mode="mixed",
        gradient_slice_count=5,
        gradient_slice1_major_diameter_um=4.0,
        gradient_sliceN_major_diameter_um=2.0,
        output_prefix_base=output_prefix_base,
        batch_seed=split_seed,
    )


def count_phase_channel_structures(data_root: Path) -> int:
    if not data_root.exists():
        return 0
    return len(autoencoder.collect_phase_channel_files(data_root))


def run_generation_stage(
    split_name: str,
    target_dir: Path,
    structure_count: int,
    split_seed: int,
    output_prefix_base: str,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    print("")
    print(
        f"=== 生成 {split_name} 集 / Generate {split_name} split ==="
    )
    existing_count_before = count_phase_channel_structures(target_dir)
    print(f"Existing structures before generation: {existing_count_before}")
    if structure_count <= 0:
        print(
            "No new structures requested for this split; skipping generation and keeping existing data."
        )
        return
    batch_config = build_split_batch_config(
        structure_count=structure_count,
        split_seed=split_seed,
        output_prefix_base=output_prefix_base,
    )
    run_random_batch_reconstruction(batch_config, base_dir=target_dir)
    existing_count_after = count_phase_channel_structures(target_dir)
    print(f"Existing structures after generation: {existing_count_after}")


def extract_split_structure_factors(
    split_root: Path,
    split_name: str,
    voxel_size_um: float,
    solid_value: int,
    liquid_values: Sequence[int],
    unknown_cluster_threshold: float,
    write_json: bool,
    write_xlsx: bool,
    skip_existing_outputs: bool = True,
    constrictivity_axis: int = 0,
    constrictivity_connectivity: int = 26,
    constrictivity_inlet: str = "low",
    chord_axis: int = 0,
) -> Dict[str, int]:
    txt_files = structure_factors.collect_phase_slice_txt_files(split_root)
    if not txt_files:
        raise ValueError(f"No *_phase_slices.txt files found under: {split_root}")
    if not write_json and not write_xlsx:
        raise ValueError("At least one of write_json/write_xlsx must be True.")

    print("")
    print(f"=== 提取 {split_name} 结构因子 / Extract {split_name} structure factors ===")
    print(f"{split_name} TXT files found: {len(txt_files)}")
    if write_json and write_xlsx:
        print("Output format: json + xlsx")
    elif write_json:
        print("Output format: json")
    else:
        print("Output format: xlsx")
    start_time = time.perf_counter()
    last_reported_percent = 0
    json_written_count = 0
    xlsx_written_count = 0
    skipped_existing_count = 0
    computed_count = 0
    if skip_existing_outputs:
        print("Existing-output policy: skip existing files")
    else:
        print("Existing-output policy: recompute even if existing files are present")

    for index, input_txt in enumerate(txt_files, start=1):
        output_json = (
            structure_factors.default_output_json_path(input_txt) if write_json else None
        )
        output_xlsx = (
            structure_factors.default_output_xlsx_path(input_txt) if write_xlsx else None
        )
        json_exists = bool(output_json is not None and output_json.exists())
        xlsx_exists = bool(output_xlsx is not None and output_xlsx.exists())
        all_requested_outputs_exist = (
            (not write_json or json_exists) and (not write_xlsx or xlsx_exists)
        )
        needs_compute = not (skip_existing_outputs and all_requested_outputs_exist)
        if not needs_compute:
            skipped_existing_count += 1
        else:
            (
                result,
                phase_volume_fractions,
                agglomeration,
                electronic_connectivity,
                electrolyte_connectivity,
                equivalent_particle_diameter,
                tortuosity,
                constrictivity,
                chord_length,
                spherical_contact,
                two_point,
            ) = structure_factors.compute_all_structure_factors(
                input_txt=input_txt,
                voxel_size_um=voxel_size_um,
                solid_value=solid_value,
                liquid_values=liquid_values,
                unknown_cluster_threshold=unknown_cluster_threshold,
                constrictivity_axis=constrictivity_axis,
                constrictivity_connectivity=constrictivity_connectivity,
                constrictivity_inlet=constrictivity_inlet,
                chord_axis=chord_axis,
            )
            computed_count += 1
            if write_json:
                if output_json is None:
                    raise RuntimeError("Internal error: output_json should not be None.")
                if (not skip_existing_outputs) or (not json_exists):
                    payload = structure_factors.build_multitask_structure_factor_payload(
                        phase_volume_fractions=phase_volume_fractions,
                        result=result,
                        electronic_connectivity=electronic_connectivity,
                        electrolyte_connectivity=electrolyte_connectivity,
                        equivalent_particle_diameter=equivalent_particle_diameter,
                        constrictivity=constrictivity,
                        chord_length=chord_length,
                        spherical_contact=spherical_contact,
                        two_point=two_point,
                    )
                    structure_factors.write_multitask_structure_factor_json(
                        output_json, payload
                    )
                    json_written_count += 1
            if write_xlsx:
                if output_xlsx is None:
                    raise RuntimeError("Internal error: output_xlsx should not be None.")
                if (not skip_existing_outputs) or (not xlsx_exists):
                    structure_factors.write_specific_surface_area_excel(
                        input_txt=input_txt,
                        output_xlsx=output_xlsx,
                        result=result,
                        phase_volume_fractions=phase_volume_fractions,
                        agglomeration=agglomeration,
                        electronic_connectivity=electronic_connectivity,
                        electrolyte_connectivity=electrolyte_connectivity,
                        equivalent_particle_diameter=equivalent_particle_diameter,
                        constrictivity=constrictivity,
                        chord_length=chord_length,
                        spherical_contact=spherical_contact,
                        two_point=two_point,
                        tortuosity=tortuosity,
                        solid_value=solid_value,
                        liquid_values=liquid_values,
                    )
                    xlsx_written_count += 1
        percent_complete = float(index) / float(len(txt_files)) * 100.0
        report_percent = int(percent_complete // 10) * 10
        if index == len(txt_files):
            report_percent = 100
        if report_percent >= 10 and report_percent > last_reported_percent:
            elapsed_seconds = float(time.perf_counter() - start_time)
            remaining_files = len(txt_files) - index
            average_seconds_per_file = elapsed_seconds / float(index)
            remaining_seconds = average_seconds_per_file * float(remaining_files)
            print(
                "[Structure Factors] "
                f"{index}/{len(txt_files)} ({report_percent}%) "
                f"| elapsed={autoencoder.format_elapsed_time(elapsed_seconds)} "
                f"({elapsed_seconds:.2f} s) "
                f"| eta={autoencoder.format_elapsed_time(remaining_seconds)} "
                f"({remaining_seconds:.2f} s)"
            )
            last_reported_percent = report_percent
    return {
        "txt_count": len(txt_files),
        "computed_count": computed_count,
        "json_written_count": json_written_count,
        "xlsx_written_count": xlsx_written_count,
        "skipped_existing_count": skipped_existing_count,
    }


def list_training_run_dirs(output_root: Path) -> List[Path]:
    if not output_root.exists():
        return []
    return sorted(
        (
            path
            for path in output_root.iterdir()
            if path.is_dir() and path.name.startswith("CNNtrain_")
        ),
        key=lambda path: path.name,
    )


def resolve_new_training_run_dir(
    output_root: Path,
    previous_run_dirs: Sequence[Path],
) -> Path:
    previous_set = {path.resolve() for path in previous_run_dirs}
    current_run_dirs = list_training_run_dirs(output_root)
    new_dirs = [
        path for path in current_run_dirs if path.resolve() not in previous_set
    ]
    if len(new_dirs) == 1:
        return new_dirs[0]
    if new_dirs:
        return sorted(new_dirs, key=lambda path: path.stat().st_mtime)[-1]
    if current_run_dirs:
        return sorted(current_run_dirs, key=lambda path: path.stat().st_mtime)[-1]
    raise RuntimeError("Unable to resolve the training output directory.")


def train_autoencoder_model(
    train_root: Path,
    validation_root: Path,
    test_root: Path,
    output_root: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    latent_dim: int,
    report_every: int,
    num_workers: int,
    seed: int,
    force_cpu: bool,
) -> Path:
    print("")
    print("=== 训练自编码器模型 / Train autoencoder model ===")
    before_run_dirs = list_training_run_dirs(output_root)
    train_args = argparse.Namespace(
        mode="train",
        data_root=train_root,
        validation_root=validation_root,
        test_root=test_root,
        output_root=output_root,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        latent_dim=latent_dim,
        report_every=report_every,
        num_workers=num_workers,
        max_files=None,
        validation_max_files=None,
        test_max_files=None,
        seed=seed,
        cpu=force_cpu,
        denoising_mask_ratio=autoencoder.DEFAULT_DENOISING_MASK_RATIO,
        reconstruction_loss_weight=autoencoder.DEFAULT_RECONSTRUCTION_LOSS_WEIGHT,
        factor_loss_weight=autoencoder.DEFAULT_FACTOR_LOSS_WEIGHT,
        resume_from=None,
        checkpoint=None,
        verify_root=validation_root,
        reconstruction_output_dir=None,
    )
    autoencoder.train_model(train_args)
    run_dir = resolve_new_training_run_dir(output_root, before_run_dirs)
    checkpoint_path = run_dir / "structure_autoencoder_checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found after training: {checkpoint_path}")
    return checkpoint_path


def run_tsne_stage(
    checkpoint_path: Path,
    test_root: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    max_files: Optional[int],
    perplexity: float,
    kmeans_clusters: int,
    tsne_iterations: int,
    seed: int,
    force_cpu: bool,
) -> None:
    print("")
    print("=== 运行 t-SNE 分析 / Run t-SNE analysis ===")
    tsne_args = argparse.Namespace(
        checkpoint=checkpoint_path,
        data_root=test_root,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_files=max_files,
        perplexity=perplexity,
        kmeans_clusters=kmeans_clusters,
        tsne_iterations=tsne_iterations,
        seed=seed,
        cpu=force_cpu,
    )
    latent_tsne.run_tsne_analysis(tsne_args)


def run_pca_stage(
    checkpoint_path: Path,
    test_root: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    max_files: Optional[int],
    seed: int,
    force_cpu: bool,
) -> None:
    print("")
    print("=== 运行 PCA 与全结构因子回归分析 / Run PCA and all-factor PC1 regression ===")
    pca_args = argparse.Namespace(
        checkpoint=checkpoint_path,
        data_root=test_root,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_files=max_files,
        factor_index=None,
        regress_all_factors=True,
        seed=seed,
        cpu=force_cpu,
    )
    latent_pca.run_analysis(pca_args)


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_root = cast(Path, args.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    train_root = dataset_root / "train"
    verify_root = dataset_root / "verify"
    test_root = dataset_root / "test"
    pipeline_stage = str(args.pipeline_stage).lower()
    if pipeline_stage not in {"full", "cpu", "gpu"}:
        raise ValueError("pipeline_stage must be one of: full, cpu, gpu")

    train_seed = int(args.batch_seed)
    verify_seed = int(args.batch_seed) + 1000
    test_seed = int(args.batch_seed) + 2000

    summary: Dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "train_root": str(train_root),
        "verify_root": str(verify_root),
        "test_root": str(test_root),
        "train_count": int(args.train_count),
        "verify_count": int(args.verify_count),
        "test_count": int(args.test_count),
        "train_seed": train_seed,
        "verify_seed": verify_seed,
        "test_seed": test_seed,
        "training_output_root": str(args.training_output_root),
        "pipeline_stage": pipeline_stage,
        "note": (
            "The verify split is used as validation data in every epoch for "
            "early stopping and best-model selection. The test split is held out "
            "and evaluated after best-model rollback."
        ),
    }

    total_start_time = time.perf_counter()
    epochs_requested = int(args.epochs)
    should_run_cpu_stage = pipeline_stage in {"full", "cpu"}
    should_run_gpu_stage = pipeline_stage in {"full", "gpu"}
    if epochs_requested == 0:
        should_run_gpu_stage = False
        summary["training_skipped_reason"] = "epochs_is_zero"
        print("")
        print("epochs=0 detected: training + t-SNE + PCA will be skipped.")
        if pipeline_stage == "gpu":
            summary["pipeline_noop"] = True
            print("pipeline-stage=gpu with epochs=0: no training-stage work will run.")

    def update_current_structure_counts() -> None:
        current_train_structure_count = count_phase_channel_structures(train_root)
        current_verify_structure_count = count_phase_channel_structures(verify_root)
        current_test_structure_count = count_phase_channel_structures(test_root)
        summary["current_train_structure_count"] = current_train_structure_count
        summary["current_verify_structure_count"] = current_verify_structure_count
        summary["current_test_structure_count"] = current_test_structure_count
        print("")
        print(
            "Current accumulated structures available for downstream stages: "
            f"train={current_train_structure_count}, "
            f"verify={current_verify_structure_count}, "
            f"test={current_test_structure_count}"
        )
        print(
            "Training and test analysis will scan all structures already present in these directories, "
            "not only the structures generated in this run."
        )

    if should_run_cpu_stage:
        generation_start_time = time.perf_counter()
        run_generation_stage(
            split_name="train",
            target_dir=train_root,
            structure_count=int(args.train_count),
            split_seed=train_seed,
            output_prefix_base="electrode_train",
        )
        run_generation_stage(
            split_name="verify",
            target_dir=verify_root,
            structure_count=int(args.verify_count),
            split_seed=verify_seed,
            output_prefix_base="electrode_verify",
        )
        run_generation_stage(
            split_name="test",
            target_dir=test_root,
            structure_count=int(args.test_count),
            split_seed=test_seed,
            output_prefix_base="electrode_test",
        )
        generation_elapsed_seconds = float(time.perf_counter() - generation_start_time)
        summary["generation_elapsed_seconds"] = generation_elapsed_seconds
        update_current_structure_counts()

        factor_start_time = time.perf_counter()
        skip_existing_factors = not bool(args.recompute_existing_factors)
        train_factor_stats = extract_split_structure_factors(
            split_root=train_root,
            split_name="train",
            voxel_size_um=0.5,
            solid_value=structure_factors.DEFAULT_SOLID_VALUE,
            liquid_values=structure_factors.DEFAULT_LIQUID_VALUES,
            unknown_cluster_threshold=structure_factors.DEFAULT_UNKNOWN_CLUSTER_THRESHOLD,
            write_json=True,
            write_xlsx=False,
            skip_existing_outputs=skip_existing_factors,
            constrictivity_axis="xyz".index(args.constrictivity_axis),
            constrictivity_connectivity=args.constrictivity_connectivity,
            constrictivity_inlet=args.constrictivity_inlet,
            chord_axis="xyz".index(args.chord_axis),
        )
        verify_factor_stats = extract_split_structure_factors(
            split_root=verify_root,
            split_name="verify",
            voxel_size_um=0.5,
            solid_value=structure_factors.DEFAULT_SOLID_VALUE,
            liquid_values=structure_factors.DEFAULT_LIQUID_VALUES,
            unknown_cluster_threshold=structure_factors.DEFAULT_UNKNOWN_CLUSTER_THRESHOLD,
            write_json=True,
            write_xlsx=False,
            skip_existing_outputs=skip_existing_factors,
            constrictivity_axis="xyz".index(args.constrictivity_axis),
            constrictivity_connectivity=args.constrictivity_connectivity,
            constrictivity_inlet=args.constrictivity_inlet,
            chord_axis="xyz".index(args.chord_axis),
        )
        test_factor_stats = extract_split_structure_factors(
            split_root=test_root,
            split_name="test",
            voxel_size_um=0.5,
            solid_value=structure_factors.DEFAULT_SOLID_VALUE,
            liquid_values=structure_factors.DEFAULT_LIQUID_VALUES,
            unknown_cluster_threshold=structure_factors.DEFAULT_UNKNOWN_CLUSTER_THRESHOLD,
            write_json=True,
            write_xlsx=True,
            skip_existing_outputs=skip_existing_factors,
            constrictivity_axis="xyz".index(args.constrictivity_axis),
            constrictivity_connectivity=args.constrictivity_connectivity,
            constrictivity_inlet=args.constrictivity_inlet,
            chord_axis="xyz".index(args.chord_axis),
        )
        factor_elapsed_seconds = float(time.perf_counter() - factor_start_time)
        summary["train_structure_factor_stats"] = train_factor_stats
        summary["verify_structure_factor_stats"] = verify_factor_stats
        summary["test_structure_factor_stats"] = test_factor_stats
        summary["structure_factor_elapsed_seconds"] = factor_elapsed_seconds
    else:
        print("")
        print("=== 跳过 CPU 阶段 / Skip CPU stage ===")
        print(
            "Generation and structure-factor extraction were skipped because "
            f"--pipeline-stage={pipeline_stage}."
        )
        summary["generation_elapsed_seconds"] = 0.0
        summary["structure_factor_elapsed_seconds"] = 0.0
        update_current_structure_counts()

    if should_run_gpu_stage:
        training_start_time = time.perf_counter()
        checkpoint_path = train_autoencoder_model(
            train_root=train_root,
            validation_root=verify_root,
            test_root=test_root,
            output_root=cast(Path, args.training_output_root),
            epochs=epochs_requested,
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            latent_dim=int(args.latent_dim),
            report_every=int(args.report_every),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
            force_cpu=bool(args.cpu),
        )
        training_elapsed_seconds = float(time.perf_counter() - training_start_time)
        training_run_dir = checkpoint_path.parent
        summary["training_elapsed_seconds"] = training_elapsed_seconds
        summary["training_run_dir"] = str(training_run_dir)
        summary["checkpoint_path"] = str(checkpoint_path)
        print(
            "Downstream test-set analysis will continue with the saved checkpoint above, "
            "including the rolled-back best model state if early stopping was triggered."
        )

        tsne_output_dir = training_run_dir / "pipeline_test_tsne"
        tsne_start_time = time.perf_counter()
        run_tsne_stage(
            checkpoint_path=checkpoint_path,
            test_root=test_root,
            output_dir=tsne_output_dir,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            max_files=args.analysis_max_files,
            perplexity=float(args.tsne_perplexity),
            kmeans_clusters=int(args.tsne_kmeans_clusters),
            tsne_iterations=int(args.tsne_iterations),
            seed=int(args.seed),
            force_cpu=bool(args.cpu),
        )
        tsne_elapsed_seconds = float(time.perf_counter() - tsne_start_time)
        summary["tsne_elapsed_seconds"] = tsne_elapsed_seconds
        summary["tsne_output_dir"] = str(tsne_output_dir)

        pca_output_dir = training_run_dir / "pipeline_test_pca"
        pca_start_time = time.perf_counter()
        run_pca_stage(
            checkpoint_path=checkpoint_path,
            test_root=test_root,
            output_dir=pca_output_dir,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            max_files=args.analysis_max_files,
            seed=int(args.seed),
            force_cpu=bool(args.cpu),
        )
        pca_elapsed_seconds = float(time.perf_counter() - pca_start_time)
        summary["pca_elapsed_seconds"] = pca_elapsed_seconds
        summary["pca_output_dir"] = str(pca_output_dir)
    else:
        print("")
        print("=== 跳过 GPU 阶段 / Skip GPU stage ===")
        print(
            "Training, t-SNE, and PCA were skipped because "
            f"--pipeline-stage={pipeline_stage}."
        )
        summary["training_elapsed_seconds"] = 0.0
        summary["tsne_elapsed_seconds"] = 0.0
        summary["pca_elapsed_seconds"] = 0.0

    total_elapsed_seconds = float(time.perf_counter() - total_start_time)
    summary["total_elapsed_seconds"] = total_elapsed_seconds
    summary["dataset_root_mode"] = "shared_accumulating_train_verify_test"
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = configure_interactively(parse_args(argv))

    if args.train_count < 0 or args.verify_count < 0 or args.test_count < 0:
        raise ValueError("train/verify/test counts must all be non-negative.")
    if args.epochs < 0:
        raise ValueError("epochs must be non-negative.")
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
    if args.tsne_perplexity <= 0.0:
        raise ValueError("tsne_perplexity must be positive.")
    if args.tsne_kmeans_clusters <= 0:
        raise ValueError("tsne_kmeans_clusters must be positive.")
    if args.tsne_iterations <= 0:
        raise ValueError("tsne_iterations must be positive.")

    summary = run_pipeline(args)
    if "training_run_dir" in summary:
        summary_path = (
            Path(str(summary["training_run_dir"])) / "integrated_pipeline_summary.json"
        )
    else:
        summary_path = Path(str(summary["dataset_root"])) / (
            f"integrated_pipeline_{summary.get('pipeline_stage', 'stage')}_summary.json"
        )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("")
    print("=== 全流程完成 / Integrated pipeline finished ===")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Pipeline stage: {summary.get('pipeline_stage', 'full')}")
    if "training_run_dir" in summary:
        print(f"Training run dir: {summary['training_run_dir']}")
    if "checkpoint_path" in summary:
        print(f"Checkpoint: {summary['checkpoint_path']}")
    if "tsne_output_dir" in summary:
        print(f"t-SNE output dir: {summary['tsne_output_dir']}")
    if "pca_output_dir" in summary:
        print(f"PCA output dir: {summary['pca_output_dir']}")
    print(
        f"Total elapsed time: {autoencoder.format_elapsed_time(summary['total_elapsed_seconds'])} "
        f"({summary['total_elapsed_seconds']:.2f} s)"
    )
    print(f"Pipeline summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
