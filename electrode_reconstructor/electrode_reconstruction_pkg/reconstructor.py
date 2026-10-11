from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .batch import (
    generate_random_batch_simulation_config,
    random_batch_domain_size_um,
    random_batch_max_major_axis_limit_um,
)
from .cbd import build_phase_labels, cleanup_and_balance_cbd_phase, grow_cbd_phase
from .exporters import (
    encode_phase_labels_three_channel,
    write_ascii_stl_from_voxels,
    write_faceted_step_from_voxels,
    write_particle_records_excel,
    write_particle_size_histogram_svg,
    write_phase_slices_txt,
)
from .particles import (
    base_semi_axes_um,
    build_ellipsoid_template,
    build_gradient_slice_specs,
    config_wrap_axes,
    default_center_bounds,
    generate_particle_pool,
    place_particle,
)
from .shared import (
    CBDGrowthStats,
    CenterBounds3,
    Float3,
    GradientSliceSpec,
    Int3,
    ParticleRecord,
    PHASE_ACTIVE,
    PHASE_CBD,
    PHASE_ELECTROLYTE,
    RANDOM_BATCH_CBD_VOLUME_FRACTION_RANGE,
    RANDOM_BATCH_ELECTROLYTE_VOLUME_FRACTION_RANGE,
    RANDOM_BATCH_MAX_MAJOR_AXIS_VOX,
    RandomBatchConfig,
    SimulationConfig,
    create_unique_run_output_dir,
    format_float3,
    sanitize_output_stem,
)

def _reconfigure_stream_errors(stream: Any) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(errors="replace")
        except Exception:
            pass


_reconfigure_stream_errors(sys.stdout)
_reconfigure_stream_errors(sys.stderr)

def count_filled_voxels_in_z_range(
    occupied: np.ndarray, z_start: int, z_end: int
) -> int:
    lower = max(0, int(z_start))
    upper = min(int(occupied.shape[2]), int(z_end))
    if upper <= lower:
        return 0
    return int(np.count_nonzero(occupied[:, :, lower:upper]))

def run_random_batch_reconstruction(
    batch_config: RandomBatchConfig, base_dir: Path
) -> None:
    batch_config.validate()
    rng = random.Random(batch_config.batch_seed)
    total_start = time.time()
    successful_structures = 0
    failed_attempts = 0
    timed_out_attempts = 0
    total_attempts = 0
    per_structure_timeout_s = float(batch_config.per_structure_timeout_s)
    gradient_timeout_floor_s = 60.0

    print("")
    print("开始随机批量生成... / Starting random batch generation...")
    print(
        f"目标结构数量 / Target structure count:            {batch_config.structure_count}"
    )
    print(
        f"目标网格尺寸 / Target grid shape (vox):           {batch_config.grid_shape_vox}"
    )
    print(
        f"体素尺寸（um） / Voxel size (um):               {batch_config.voxel_size_um:.3f}"
    )
    print(
        "对应物理尺寸（um） / Represented domain size (um): "
        f"{format_float3(random_batch_domain_size_um(batch_config))}"
    )
    print("粒径分布 / Size distribution:                    lognormal")
    print("取向分布 / Orientation distribution:            normal")
    print(
        "电极结构模式 / Electrode structure mode:             "
        f"{batch_config.electrode_mode}"
    )
    if batch_config.electrode_mode == "gradient":
        print(
            "梯度切片设置 / Gradient slicing:                   "
            f"slices={batch_config.gradient_slice_count}, "
            f"slice1_major={batch_config.gradient_slice1_major_diameter_um:.3f} um, "
            f"sliceN_major={batch_config.gradient_sliceN_major_diameter_um:.3f} um"
        )
    elif batch_config.electrode_mode == "mixed":
        print(
            "Electrode structure mode: mixed "
            "(uniform/gradient sampled per structure, p=0.5 each)"
        )
    print(
        "随机体积分数范围 / Random volume-fraction ranges: "
        f"CBD={RANDOM_BATCH_CBD_VOLUME_FRACTION_RANGE[0]:.2f}-"
        f"{RANDOM_BATCH_CBD_VOLUME_FRACTION_RANGE[1]:.2f}, "
        f"electrolyte={RANDOM_BATCH_ELECTROLYTE_VOLUME_FRACTION_RANGE[0]:.2f}-"
        f"{RANDOM_BATCH_ELECTROLYTE_VOLUME_FRACTION_RANGE[1]:.2f} "
        "(biased toward lower porosity), particle=remaining volume"
    )
    print(
        "标准颗粒长轴上限（um） / Max base major diameter: "
        f"{random_batch_max_major_axis_limit_um(batch_config):.3f}"
    )
    print(
        "标准颗粒长轴上限（vox） / Max base major diameter (vox): "
        f"{RANDOM_BATCH_MAX_MAJOR_AXIS_VOX:.1f}"
    )
    print(
        "说明 / Note: 该上限仅作用于标准颗粒；实际颗粒会再乘以对数正态尺度因子，"
        "因此长轴仍可能大于 8 个体素。 / "
        "This limit applies only to the base particle; actual particles are further "
        "scaled by the lognormal factor and may still exceed 8 voxels."
    )
    print(
        f"批量随机种子 / Batch random seed:                {batch_config.batch_seed}"
    )
    print(
        f"批量输出根目录 / Batch output root:              {base_dir}"
    )
    print("输出文件类型 / Output file types:               TXT + NPY only")
    print(
        "单个结构超时限制（s） / Per-structure timeout (s):         "
        f"{per_structure_timeout_s:.1f}"
    )
    print(
        "Gradient timeout policy: gradient structures use "
        f"max(base_timeout, {gradient_timeout_floor_s:.1f}s)."
    )
    print("")

    while successful_structures < batch_config.structure_count:
        total_attempts += 1
        structure_index = successful_structures + 1
        config = generate_random_batch_simulation_config(
            batch_config, rng, structure_index
        )
        print(
            f"[{structure_index}/{batch_config.structure_count}] "
            "随机结构生成 / Random structure generation"
        )
        print(
            "参数摘要 / Parameter summary: "
            f"vf=({config.particle_volume_fraction:.3f}, "
            f"{config.cbd_volume_fraction:.3f}, "
            f"{config.electrolyte_volume_fraction:.3f}), "
            f"mode={config.electrode_mode}, "
            f"base_diam_um={format_float3(config.base_axes_um)}, "
            f"lognormal(mean={config.size_distribution.mean:.3f}, "
            f"sigma={config.size_distribution.sigma:.3f}), "
            f"normal_std_deg={format_float3(config.orientation_distribution.std_deg)}, "
            f"omega={config.cbd_growth.morphology_parameter:.3f}, "
            f"seed={config.random_seed}"
        )
        try:
            runtime_limit_s = per_structure_timeout_s
            if config.electrode_mode == "gradient":
                runtime_limit_s = max(runtime_limit_s, gradient_timeout_floor_s)
            run_reconstruction(
                config,
                base_dir=base_dir,
                max_runtime_s=runtime_limit_s,
            )
            successful_structures += 1
            if (
                successful_structures % 25 == 0
                or successful_structures == batch_config.structure_count
            ):
                elapsed = time.time() - total_start
                print(
                    "批量进度 / Batch progress: "
                    f"{successful_structures}/{batch_config.structure_count}, "
                    f"attempts={total_attempts}, failed_attempts={failed_attempts}, "
                    f"timed_out_attempts={timed_out_attempts}, elapsed={elapsed:.1f}s"
                )
        except TimeoutError as exc:
            failed_attempts += 1
            timed_out_attempts += 1
            print(
                "Single structure timed out and was skipped; continuing with compensation. "
                f"failed_attempts={failed_attempts}, timed_out_attempts={timed_out_attempts}, "
                f"error={exc}"
            )
        except Exception as exc:
            failed_attempts += 1
            print(
                "随机参数下本次生成失败，正在重试。 / This randomized case failed; retrying. "
                f"failed_attempts={failed_attempts}, error={exc}"
            )
    total_elapsed = time.time() - total_start
    print("")
    print("随机批量生成完成。 / Random batch generation finished.")
    print(
        f"成功结构数 / Successful structures:               {successful_structures}"
    )
    print(
        f"失败重试次数 / Failed retries:                   {failed_attempts}"
    )
    print(
        f"超时跳过次数 / Timed-out skips:                  {timed_out_attempts}"
    )
    print(
        f"总尝试次数 / Total attempts:                     {total_attempts}"
    )
    print(
        f"总耗时（s） / Total elapsed time (s):            {total_elapsed:.3f}"
    )

def run_reconstruction(
    config: SimulationConfig,
    base_dir: Path,
    max_runtime_s: float | None = None,
) -> Dict[str, object]:
    rng = random.Random(config.random_seed)
    reconstruction_start_s = time.time()
    reconstruction_deadline_s: float | None = None
    if max_runtime_s is not None:
        if max_runtime_s <= 0.0:
            raise ValueError("max_runtime_s must be positive when provided.")
        reconstruction_deadline_s = reconstruction_start_s + float(max_runtime_s)

    def ensure_within_runtime(stage: str) -> None:
        if reconstruction_deadline_s is None:
            return
        if time.time() > reconstruction_deadline_s:
            elapsed_s = time.time() - reconstruction_start_s
            raise TimeoutError(
                f"Reconstruction exceeded {max_runtime_s:.1f}s at stage '{stage}' "
                f"(elapsed={elapsed_s:.1f}s)."
            )

    base_axes_vox_array = base_semi_axes_um(config) / config.voxel_size_um
    base_axes_vox: Float3 = (
        float(base_axes_vox_array[0]),
        float(base_axes_vox_array[1]),
        float(base_axes_vox_array[2]),
    )
    standard_template = build_ellipsoid_template(
        base_axes_vox,
        (0.0, 0.0, 0.0),
    )

    shape_values = [
        max(1, int(round(length / config.voxel_size_um)))
        for length in config.domain_size_um
    ]
    shape: Int3 = (shape_values[0], shape_values[1], shape_values[2])
    total_voxels = int(np.prod(shape))
    target_particle_voxels = int(round(total_voxels * config.particle_volume_fraction))
    target_particle_volume_um3 = target_particle_voxels * (config.voxel_size_um ** 3)
    represented_domain_size_um = tuple(
        dimension * config.voxel_size_um for dimension in shape
    )

    print("开始重构... / Starting reconstruction...")
    print(f"体素网格尺寸 / Voxel grid shape: {shape}")
    print(f"重构域实际尺寸（um） / Represented domain size (um): {represented_domain_size_um}")
    print(f"目标颗粒体素数 / Target particle voxels: {target_particle_voxels}")
    if config.cbd_volume_fraction > 0.0:
        print(
            "目标 CBD 体素数 / Target CBD voxels: "
            f"{int(round(total_voxels * config.cbd_volume_fraction))}"
        )
    ensure_within_runtime("initialization")

    template_cache: Dict[Tuple[float, ...], np.ndarray] = {}
    wrap_axes = config_wrap_axes(config)
    particle_records: List[ParticleRecord] = []
    placement_queue: List[Tuple[ParticleRecord, CenterBounds3]] = []
    gradient_slice_specs: List[GradientSliceSpec] = []
    gradient_slice_results: List[Dict[str, object]] = []

    if config.electrode_mode == "gradient":
        gradient_slice_specs = build_gradient_slice_specs(
            config,
            shape,
        )
        target_particle_voxels = sum(
            int(slice_spec["target_particle_voxels"])
            for slice_spec in gradient_slice_specs
        )
        target_particle_volume_um3 = target_particle_voxels * (config.voxel_size_um ** 3)
        print(
            "电极结构模式 / Electrode structure mode: gradient"
        )
        print(
            "Gradient slice total target particle voxels: "
            f"{target_particle_voxels}"
        )
        for slice_spec in gradient_slice_specs:
            print(
                f"  slice{slice_spec['slice_index']}: "
                f"z=[{slice_spec['z_start']}, {slice_spec['z_end']}) vox, "
                f"major_diameter={float(slice_spec['major_diameter_um']):.3f} um, "
                f"target_particle_voxels={int(slice_spec['target_particle_voxels'])}"
            )
            pass
            _ = """Obsolete pre-sampling log removed.
                f"  slice{slice_spec['slice_index']} sampled particle pool size "
                ""
            )
            """
    else:
        print("电极结构模式 / Electrode structure mode: uniform")
        particle_records = generate_particle_pool(
            config=config,
            target_particle_volume_um3=target_particle_volume_um3,
            template_cache=template_cache,
            rng=rng,
        )
        default_bounds = default_center_bounds(shape)
        placement_queue = [(record, default_bounds) for record in particle_records]

    print(f"采样颗粒池数量 / Sampled particle pool size: {len(particle_records)}")
    if config.electrode_mode == "gradient":
        print(
            "Gradient mode note: particle pools are generated slice-by-slice during placement."
        )
    active_phase = np.zeros(shape, dtype=bool)
    occupied_count = 0
    placed_particles = 0
    skipped_particles = 0
    processed_particles = 0
    start_time = time.time()
    if config.electrode_mode == "gradient":
        max_slice_pool_generations = 6
        max_stalled_slice_pools = 2
        for slice_spec in gradient_slice_specs:
            slice_index = int(slice_spec["slice_index"])
            z_start = int(slice_spec["z_start"])
            z_end = int(slice_spec["z_end"])
            slice_target_voxels = int(slice_spec["target_particle_voxels"])
            slice_bounds = slice_spec["center_bounds"]
            slice_config = replace(
                config,
                base_axes_um=slice_spec["base_axes_um"],
            )
            slice_volume_voxels = int(shape[0]) * int(shape[1]) * max(1, z_end - z_start)
            slice_filled_voxels = count_filled_voxels_in_z_range(
                active_phase, z_start, z_end
            )
            slice_prefilled_voxels = slice_filled_voxels
            slice_processed_particles = 0
            slice_placed_particles = 0
            slice_skipped_particles = 0
            slice_pool_generations = 0
            stalled_slice_pools = 0

            print(
                f"slice{slice_index} / Locking slice particle volume fraction: "
                f"target={slice_target_voxels}, prefilled={slice_prefilled_voxels}"
            )
            if slice_filled_voxels > slice_target_voxels:
                print(
                    f"Warning: slice{slice_index} was already overfilled before local placement. "
                    f"filled={slice_filled_voxels}, target={slice_target_voxels}"
                )

            while (
                slice_filled_voxels < slice_target_voxels
                and slice_pool_generations < max_slice_pool_generations
            ):
                ensure_within_runtime(f"gradient_slice_{slice_index}_pool_sampling")
                remaining_slice_voxels = max(
                    0, slice_target_voxels - slice_filled_voxels
                )
                if slice_pool_generations == 0:
                    requested_slice_voxels = max(
                        slice_target_voxels,
                        remaining_slice_voxels,
                    )
                else:
                    requested_slice_voxels = max(
                        remaining_slice_voxels * 2,
                        max(1, slice_target_voxels // 2),
                    )
                slice_pool_generations += 1
                slice_records = generate_particle_pool(
                    config=slice_config,
                    target_particle_volume_um3=float(requested_slice_voxels)
                    * (config.voxel_size_um ** 3),
                    template_cache=template_cache,
                    rng=rng,
                )
                particle_records.extend(slice_records)
                print(
                    f"  slice{slice_index} pool {slice_pool_generations}: "
                    f"remaining={remaining_slice_voxels}, sampled={len(slice_records)}"
                )

                slice_pool_added_voxels = 0
                slice_report_interval = (
                    max(1, len(slice_records) // 10) if slice_records else 1
                )
                for pool_index, record in enumerate(slice_records, start=1):
                    ensure_within_runtime(f"gradient_slice_{slice_index}_placement")
                    processed_particles += 1
                    slice_processed_particles += 1
                    new_cells, new_slice_cells = place_particle(
                        active_phase,
                        record,
                        shape,
                        slice_config,
                        rng,
                        center_bounds=slice_bounds,
                        min_z_inclusive=int(slice_spec["min_z_inclusive"]),
                        wrap_axes=wrap_axes,
                        count_z_range=(z_start, z_end),
                        timeout_deadline_s=reconstruction_deadline_s,
                    )
                    if record.placed:
                        occupied_count += new_cells
                        slice_filled_voxels += new_slice_cells
                        slice_pool_added_voxels += new_slice_cells
                        placed_particles += 1
                        slice_placed_particles += 1
                    else:
                        skipped_particles += 1
                        slice_skipped_particles += 1

                    if (
                        pool_index % slice_report_interval == 0
                        or slice_filled_voxels >= slice_target_voxels
                    ):
                        current_slice_vf = slice_filled_voxels / max(
                            1, slice_volume_voxels
                        )
                        print(
                            f"    slice{slice_index} [{pool_index}/{len(slice_records)}] "
                            f"placed={slice_placed_particles}, skipped={slice_skipped_particles}, "
                            f"slice_particle_vf={current_slice_vf:.4f}, "
                            f"filled={slice_filled_voxels}/{slice_target_voxels}"
                        )

                    if slice_filled_voxels >= slice_target_voxels:
                        break

                if slice_filled_voxels >= slice_target_voxels:
                    break

                if slice_pool_added_voxels <= 0:
                    stalled_slice_pools += 1
                    if stalled_slice_pools >= max_stalled_slice_pools:
                        print(
                            f"Warning: slice{slice_index} stalled before reaching target. "
                            f"filled={slice_filled_voxels}, target={slice_target_voxels}"
                        )
                        break
                else:
                    stalled_slice_pools = 0

            final_slice_filled_voxels = count_filled_voxels_in_z_range(
                active_phase, z_start, z_end
            )
            gradient_slice_results.append(
                {
                    "slice_index": slice_index,
                    "z_start": z_start,
                    "z_end": z_end,
                    "target_particle_voxels": slice_target_voxels,
                    "prefilled_particle_voxels": slice_prefilled_voxels,
                    "final_particle_voxels": final_slice_filled_voxels,
                    "added_particle_voxels": max(
                        0, final_slice_filled_voxels - slice_prefilled_voxels
                    ),
                    "processed_particles": slice_processed_particles,
                    "placed_particles": slice_placed_particles,
                    "skipped_particles": slice_skipped_particles,
                    "pool_generations": slice_pool_generations,
                    "target_met": final_slice_filled_voxels >= slice_target_voxels,
                }
            )
    report_interval = max(1, len(placement_queue) // 20) if placement_queue else 1

    for record, center_bounds in placement_queue:
        ensure_within_runtime("uniform_particle_placement")
        processed_particles += 1
        new_cells, _ = place_particle(
            active_phase,
            record,
            shape,
            config,
            rng,
            center_bounds=center_bounds,
            wrap_axes=wrap_axes,
            timeout_deadline_s=reconstruction_deadline_s,
        )
        if record.placed:
            occupied_count += new_cells
            placed_particles += 1
        else:
            skipped_particles += 1

        if (
            processed_particles % report_interval == 0
            or occupied_count >= target_particle_voxels
        ):
            current_vf = occupied_count / max(1, total_voxels)
            print(
                f"[{processed_particles}/{len(particle_records)}] "
                f"已放置/placed={placed_particles}, 已跳过/skipped={skipped_particles}, "
                f"颗粒体积分数/particle_vf={current_vf:.4f}"
            )

        if occupied_count >= target_particle_voxels:
            break

    particle_elapsed = time.time() - start_time
    for particle_id, record in enumerate(particle_records, start=1):
        record.particle_id = particle_id
    cbd_elapsed = 0.0
    cbd_phase = np.zeros(shape, dtype=bool)
    cbd_stats: CBDGrowthStats = {
        "target_cbd_voxels": 0,
        "seeded_cbd_voxels": 0,
        "grown_cbd_voxels": 0,
        "growth_iterations": 0,
        "candidate_voxels_final": 0,
        "growth_terminated_early": False,
        "growth_stop_reason": "disabled",
        "cleanup_removed_components": 0,
        "cleanup_removed_voxels": 0,
        "cleanup_regrown_voxels": 0,
        "cleanup_eroded_voxels": 0,
        "cleanup_final_voxels": 0,
    }
    if config.cbd_volume_fraction > 0.0:
        print("开始生长 CBD 相... / Starting CBD growth...")
        ensure_within_runtime("cbd_growth_start")
        cbd_start = time.time()
        cbd_phase, cbd_stats = grow_cbd_phase(active_phase, config, rng)
        ensure_within_runtime("cbd_growth_post_growth")
        cbd_phase, cbd_cleanup_stats = cleanup_and_balance_cbd_phase(
            cbd_phase,
            active_phase,
            config,
            rng,
        )
        ensure_within_runtime("cbd_growth_post_cleanup")
        cbd_stats["cleanup_removed_components"] = cbd_cleanup_stats[
            "removed_components"
        ]
        cbd_stats["cleanup_removed_voxels"] = cbd_cleanup_stats["removed_voxels"]
        cbd_stats["cleanup_regrown_voxels"] = cbd_cleanup_stats["regrown_voxels"]
        cbd_stats["cleanup_eroded_voxels"] = cbd_cleanup_stats["eroded_voxels"]
        cbd_stats["cleanup_final_voxels"] = cbd_cleanup_stats["final_voxels"]
        print(
            "CBD 清理和平衡 / CBD cleanup and balance: "
            f"min_component={config.cbd_growth.cleanup_min_component_voxels}, "
            f"removed_components={cbd_cleanup_stats['removed_components']}, "
            f"removed_voxels={cbd_cleanup_stats['removed_voxels']}, "
            f"regrown={cbd_cleanup_stats['regrown_voxels']}, "
            f"eroded={cbd_cleanup_stats['eroded_voxels']}, "
            f"final={cbd_cleanup_stats['final_voxels']}"
        )
        cbd_elapsed = time.time() - cbd_start

    ensure_within_runtime("output_preparation")
    phase_labels = build_phase_labels(active_phase, cbd_phase)
    actual_particle_vf = float(np.count_nonzero(phase_labels == PHASE_ACTIVE)) / max(
        1, total_voxels
    )
    actual_cbd_vf = float(np.count_nonzero(phase_labels == PHASE_CBD)) / max(
        1, total_voxels
    )
    actual_electrolyte_vf = float(
        np.count_nonzero(phase_labels == PHASE_ELECTROLYTE)
    ) / max(1, total_voxels)
    elapsed = particle_elapsed + cbd_elapsed

    run_output_dir = create_unique_run_output_dir(base_dir)
    output_stem = sanitize_output_stem(config.output_prefix)
    output_prefix = run_output_dir / output_stem
    active_step_vertex_count = 0
    active_step_face_count = 0
    cbd_step_vertex_count = 0
    cbd_step_face_count = 0
    particle_histogram_path = ""
    particle_histogram_time_s = 0.0
    particle_histogram_warning = ""
    output_times_s: Dict[str, float] = {}

    if config.export_structure_txt:
        output_start = time.time()
        write_phase_slices_txt(
            phase_labels,
            Path(str(output_prefix) + "_phase_slices.txt"),
        )
        output_times_s["structure_txt"] = time.time() - output_start
        print(
            "结构 TXT 输出完成 / Structure TXT written: "
            f"elapsed={output_times_s['structure_txt']:.3f}s"
        )
    if config.export_voxel_npy:
        output_start = time.time()
        np.save(
            Path(str(output_prefix) + "_phase_channels.npy"),
            encode_phase_labels_three_channel(phase_labels),
        )
        output_times_s["phase_channels_npy"] = time.time() - output_start
        print(
            "三通道 NPY 输出完成 / Three-channel NPY written: "
            f"elapsed={output_times_s['phase_channels_npy']:.3f}s"
        )
    if config.export_stl:
        output_start = time.time()
        if np.any(active_phase):
            write_ascii_stl_from_voxels(
                active_phase,
                config.voxel_size_um,
                Path(str(output_prefix) + "_active_phase.stl"),
                solid_name="active_phase",
            )
        if np.any(cbd_phase):
            write_ascii_stl_from_voxels(
                cbd_phase,
                config.voxel_size_um,
                Path(str(output_prefix) + "_cbd_phase.stl"),
                solid_name="cbd_phase",
            )
        output_times_s["stl"] = time.time() - output_start
        print(
            "STL 输出完成 / STL written: "
            f"elapsed={output_times_s['stl']:.3f}s"
        )
    if config.export_step and np.any(active_phase):
        output_start = time.time()
        active_step_vertex_count, active_step_face_count = write_faceted_step_from_voxels(
            active_phase,
            config.voxel_size_um,
            Path(str(output_prefix) + "_active_phase.step"),
        )
        output_times_s["active_step"] = time.time() - output_start
        print(
            "活性颗粒 STEP 输出完成 / Active STEP written: "
            f"elapsed={output_times_s['active_step']:.3f}s"
        )
    if config.export_step and np.any(cbd_phase):
        output_start = time.time()
        cbd_step_vertex_count, cbd_step_face_count = write_faceted_step_from_voxels(
            cbd_phase,
            config.voxel_size_um,
            Path(str(output_prefix) + "_cbd_phase.step"),
        )
        output_times_s["cbd_step"] = time.time() - output_start
        print(
            "CBD STEP 输出完成 / CBD STEP written: "
            f"elapsed={output_times_s['cbd_step']:.3f}s"
        )
    if config.export_particle_excel:
        output_start = time.time()
        write_particle_records_excel(
            particle_records, Path(str(output_prefix) + "_particles.xlsx")
        )
        output_times_s["particle_excel"] = time.time() - output_start
        print(
            "颗粒 Excel 输出完成 / Particle Excel written: "
            f"records={len(particle_records)}, "
            f"elapsed={output_times_s['particle_excel']:.3f}s"
        )
    if config.export_particle_histogram:
        placed_record_count = sum(1 for record in particle_records if record.placed)
        if placed_record_count > 0:
            histogram_path = Path(str(output_prefix) + "_particle_size_histogram.svg")
            print(
                "开始生成粒径直方图 / Generating particle size histogram: "
                f"placed_particles={placed_record_count}"
            )
            histogram_start = time.time()
            try:
                write_particle_size_histogram_svg(particle_records, histogram_path)
                particle_histogram_time_s = time.time() - histogram_start
                output_times_s["particle_histogram"] = particle_histogram_time_s
                if histogram_path.exists():
                    particle_histogram_path = str(histogram_path)
                print(
                    "粒径直方图生成完成 / Particle size histogram finished: "
                    f"elapsed={particle_histogram_time_s:.3f}s"
                )
            except Exception as exc:
                particle_histogram_time_s = time.time() - histogram_start
                output_times_s["particle_histogram"] = particle_histogram_time_s
                particle_histogram_warning = str(exc)
                print(
                    "警告：粒径直方图生成失败，核心结构输出已保留。 / "
                    "Warning: particle size histogram failed; core structure outputs were kept. "
                    f"elapsed={particle_histogram_time_s:.3f}s, error={exc}"
                )

    summary = {
        "config": asdict(config),
        "voxel_grid_shape": list(shape),
        "represented_domain_size_um": list(represented_domain_size_um),
        "target_particle_voxels": target_particle_voxels,
        "target_cbd_voxels": int(cbd_stats["target_cbd_voxels"]),
        "filled_particle_voxels": int(np.count_nonzero(active_phase)),
        "filled_cbd_voxels": int(np.count_nonzero(cbd_phase)),
        "filled_electrolyte_voxels": int(
            np.count_nonzero(phase_labels == PHASE_ELECTROLYTE)
        ),
        "standard_particle_template_voxels": int(standard_template.shape[0]),
        "sampled_particles": len(particle_records),
        "processed_particles": processed_particles,
        "placed_particles": placed_particles,
        "skipped_particles": skipped_particles,
        "step_surface_vertices": active_step_vertex_count,
        "step_surface_faces": active_step_face_count,
        "active_phase_step_surface_vertices": active_step_vertex_count,
        "active_phase_step_surface_faces": active_step_face_count,
        "cbd_phase_step_surface_vertices": cbd_step_vertex_count,
        "cbd_phase_step_surface_faces": cbd_step_face_count,
        "actual_particle_volume_fraction": actual_particle_vf,
        "actual_cbd_volume_fraction": actual_cbd_vf,
        "actual_electrolyte_volume_fraction": actual_electrolyte_vf,
        "particle_reconstruction_time_s": particle_elapsed,
        "cbd_growth_time_s": cbd_elapsed,
        "reconstruction_time_s": elapsed,
        "cbd_growth": cbd_stats,
        "gradient_slice_specs": gradient_slice_specs,
        "gradient_slice_results": gradient_slice_results,
        "particle_size_histogram": particle_histogram_path,
        "particle_size_histogram_time_s": particle_histogram_time_s,
        "particle_size_histogram_warning": particle_histogram_warning,
        "output_times_s": output_times_s,
        "output_directory": str(run_output_dir),
        "output_prefix": str(output_prefix),
    }

    if config.export_summary_json:
        with Path(str(output_prefix) + "_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")

    print("")
    print("重构完成。 / Reconstruction finished.")
    print(f"实际颗粒体积分数 / Actual particle volume fraction:   {actual_particle_vf:.6f}")
    print(f"实际 CBD 体积分数 / Actual CBD volume fraction:        {actual_cbd_vf:.6f}")
    print(f"实际电解液体积分数 / Actual electrolyte volume fraction:{actual_electrolyte_vf:.6f}")
    print(f"已放置颗粒数 / Placed particles:                  {placed_particles}")
    print(f"已跳过颗粒数 / Skipped particles:                 {skipped_particles}")
    if config.electrode_mode == "gradient":
        print(
            f"梯度切片数量 / Gradient slice count:             {config.gradient_slice_count}"
        )
    if particle_histogram_path:
        print(f"粒径直方图 / Particle size histogram:           {particle_histogram_path}")
    if config.cbd_volume_fraction > 0.0:
        print(
            "CBD 生长信息 / CBD growth summary:                "
            f"seeded={cbd_stats['seeded_cbd_voxels']}, "
            f"grown={cbd_stats['grown_cbd_voxels']}, "
            f"stop={cbd_stats['growth_stop_reason']}"
        )
    print(f"耗时（s） / Elapsed time (s):                  {elapsed:.3f}")
    print(f"本次输出目录 / Run output directory:            {run_output_dir}")
    print(f"输出文件前缀路径 / Outputs written with prefix:      {output_prefix}")

    if actual_particle_vf + 1.0e-6 < config.particle_volume_fraction:
        print(
            "Warning: target particle volume fraction was not fully reached. "
            "Try increasing particle_pool_factor or max_placement_attempts."
        )
    if actual_cbd_vf + 1.0e-6 < config.cbd_volume_fraction:
        print(
            "Warning: target CBD volume fraction was not fully reached. "
            "Try increasing cbd_growth.max_growth_steps or adjusting "
            "cbd_growth.morphology_parameter / seed_fraction."
        )

    return summary

