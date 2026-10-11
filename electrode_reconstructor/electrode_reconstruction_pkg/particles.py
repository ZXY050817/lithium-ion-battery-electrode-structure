from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .shared import (
    CenterBounds3,
    EPS,
    Float3,
    GradientSliceSpec,
    Int3,
    ParticleRecord,
    SimulationConfig,
    SizeDistributionConfig,
    WrapAxes,
    normalize_angles_deg,
)

def rotation_matrix_xyz_deg(angles_deg: Sequence[float]) -> np.ndarray:
    thetax, thetay, thetaz = (math.radians(float(value)) for value in angles_deg)

    r_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(thetax), -math.sin(thetax)],
            [0.0, math.sin(thetax), math.cos(thetax)],
        ],
        dtype=float,
    )
    r_y = np.array(
        [
            [math.cos(thetay), 0.0, math.sin(thetay)],
            [0.0, 1.0, 0.0],
            [-math.sin(thetay), 0.0, math.cos(thetay)],
        ],
        dtype=float,
    )
    r_z = np.array(
        [
            [math.cos(thetaz), -math.sin(thetaz), 0.0],
            [math.sin(thetaz), math.cos(thetaz), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return r_z @ r_y @ r_x

def ellipsoid_volume_um3(semi_axes_um: Sequence[float]) -> float:
    a, b, c = (float(value) for value in semi_axes_um)
    return 4.0 / 3.0 * math.pi * a * b * c

def build_ellipsoid_template(
    semi_axes_vox: Sequence[float], angles_deg: Sequence[float]
) -> np.ndarray:
    semi_axes = np.asarray(semi_axes_vox, dtype=float)
    if np.any(semi_axes <= 0.0):
        raise ValueError("semi_axes_vox must all be positive.")

    rotation = rotation_matrix_xyz_deg(angles_deg)
    extents = np.ceil(np.abs(rotation) @ semi_axes).astype(int) + 1

    xs = np.arange(-extents[0], extents[0] + 1, dtype=int)
    ys = np.arange(-extents[1], extents[1] + 1, dtype=int)
    zs = np.arange(-extents[2], extents[2] + 1, dtype=int)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    local_coordinates = grid @ rotation
    normalized_distance = np.sum((local_coordinates / semi_axes[None, :]) ** 2, axis=1)
    inside = normalized_distance <= (1.0 + EPS)
    template = grid[inside]

    if template.size == 0:
        template = np.zeros((1, 3), dtype=np.int32)

    return np.unique(template.astype(np.int32), axis=0)

def template_key(
    semi_axes_vox: Sequence[float], angles_deg: Sequence[float]
) -> Tuple[float, ...]:
    normalized_angles = normalize_angles_deg(angles_deg)
    rounded_axes = tuple(round(float(value), 6) for value in semi_axes_vox)
    rounded_angles = tuple(round(float(value), 6) for value in normalized_angles)
    return rounded_axes + rounded_angles

def base_semi_axes_um(config: SimulationConfig) -> np.ndarray:
    axes = np.asarray(config.base_axes_um, dtype=float)
    if config.axis_definition == "diameter":
        axes = axes / 2.0
    return axes

def config_wrap_axes(config: SimulationConfig) -> WrapAxes:
    if config.electrode_mode == "gradient":
        return (True, True, False)
    return (True, True, True)

def current_base_major_diameter_um(config: SimulationConfig) -> float:
    return 2.0 * float(np.max(base_semi_axes_um(config)))

def scaled_base_axes_um_for_major_diameter(
    config: SimulationConfig, major_diameter_um: float
) -> Float3:
    reference_major = current_base_major_diameter_um(config)
    scale = float(major_diameter_um) / max(EPS, reference_major)
    scaled_axes = np.asarray(config.base_axes_um, dtype=float) * scale
    return (
        float(scaled_axes[0]),
        float(scaled_axes[1]),
        float(scaled_axes[2]),
    )

def gradient_slice_major_diameters_um(config: SimulationConfig) -> List[float]:
    if config.gradient_slice_count <= 1:
        return [float(config.gradient_slice1_major_diameter_um)]
    values = np.linspace(
        float(config.gradient_slice1_major_diameter_um),
        float(config.gradient_sliceN_major_diameter_um),
        int(config.gradient_slice_count),
        dtype=np.float64,
    )
    return [float(value) for value in values.tolist()]

def gradient_slice_particle_volume_fractions(
    config: SimulationConfig,
) -> List[float]:
    if config.gradient_slice_count <= 1:
        return [float(config.gradient_slice1_particle_volume_fraction)]
    values = np.linspace(
        float(config.gradient_slice1_particle_volume_fraction),
        float(config.gradient_sliceN_particle_volume_fraction),
        int(config.gradient_slice_count),
        dtype=np.float64,
    )
    return [float(value) for value in values.tolist()]

def gradient_slice_ranges(depth_voxels: int, slice_count: int) -> List[Tuple[int, int]]:
    if slice_count <= 0:
        raise ValueError("slice_count must be positive.")
    if depth_voxels <= 0:
        raise ValueError("depth_voxels must be positive.")
    base_thickness = depth_voxels // slice_count
    remainder = depth_voxels % slice_count
    start = 0
    ranges: List[Tuple[int, int]] = []
    for slice_index in range(slice_count):
        thickness = base_thickness + (1 if slice_index < remainder else 0)
        end = start + thickness
        ranges.append((start, end))
        start = end
    return ranges

def distribute_integer_targets(total: int, weights: Sequence[int]) -> List[int]:
    if total < 0:
        raise ValueError("total must be non-negative.")
    if not weights:
        return []
    weight_values = [max(0, int(weight)) for weight in weights]
    weight_sum = sum(weight_values)
    if weight_sum <= 0:
        base = total // len(weight_values)
        remainder = total - base * len(weight_values)
        targets = [base for _ in weight_values]
        for index in range(remainder):
            targets[index] += 1
        return targets
    raw_targets = [float(total) * float(weight) / float(weight_sum) for weight in weight_values]
    floor_targets = [int(math.floor(value)) for value in raw_targets]
    remainder = total - sum(floor_targets)
    ranked_indices = sorted(
        range(len(raw_targets)),
        key=lambda index: (raw_targets[index] - floor_targets[index], -index),
        reverse=True,
    )
    for index in ranked_indices[:remainder]:
        floor_targets[index] += 1
    return floor_targets

def sample_lognormal_factor(
    cfg: SizeDistributionConfig, rng: random.Random
) -> float:
    if cfg.parameterization == "log_space":
        mu_ln = cfg.mean
        sigma_ln = cfg.sigma
    else:
        if cfg.sigma == 0.0:
            return float(cfg.mean)
        variance = cfg.sigma ** 2
        sigma_ln_sq = math.log(1.0 + variance / (cfg.mean ** 2))
        sigma_ln = math.sqrt(max(sigma_ln_sq, 0.0))
        mu_ln = math.log(cfg.mean) - 0.5 * sigma_ln_sq

    factor = rng.lognormvariate(mu_ln, sigma_ln)
    return max(factor, EPS)

def sample_inverse_histogram_factor(
    cfg: SizeDistributionConfig,
    base_major_diameter_um: float,
    rng: random.Random,
) -> float:
    weights = [max(0.0, float(value)) for value in cfg.inverse_probabilities]
    total_weight = sum(weights)
    if total_weight <= EPS:
        return 1.0

    random_value = rng.random() * total_weight
    cumulative = 0.0
    selected_bin = len(weights) - 1
    for index, weight in enumerate(weights):
        cumulative += weight
        if random_value <= cumulative:
            selected_bin = index
            break

    bin_width = float(cfg.inverse_bin_width_um)
    lower = float(cfg.inverse_bin_start_um) + float(selected_bin) * bin_width
    upper = lower + bin_width
    sampled_diameter_um = rng.uniform(max(EPS, lower), max(EPS, upper))
    return max(sampled_diameter_um / max(EPS, float(base_major_diameter_um)), EPS)

def sample_size_factor(
    cfg: SizeDistributionConfig,
    base_major_diameter_um: float,
    rng: random.Random,
) -> float:
    if cfg.mode == "homogeneous":
        return 1.0
    if cfg.mode == "inverse_histogram":
        return sample_inverse_histogram_factor(cfg, base_major_diameter_um, rng)
    return sample_lognormal_factor(cfg, rng)

def particle_size_rank_value(record: ParticleRecord) -> float:
    return 2.0 * max(float(value) for value in record.semi_axes_um)

def particle_pool_volume(records: Sequence[ParticleRecord]) -> float:
    return float(sum(record.analytic_volume_um3 for record in records))

def trim_lognormal_particle_pool_by_size_rank(
    records: Sequence[ParticleRecord],
    size_distribution: SizeDistributionConfig,
) -> List[ParticleRecord]:
    if (
        size_distribution.mode != "lognormal"
        or not bool(size_distribution.rank_trim_enabled)
        or len(records) < 20
    ):
        return list(records)

    sorted_by_size = sorted(
        records,
        key=lambda item: (particle_size_rank_value(item), item.analytic_volume_um3),
        reverse=True,
    )
    count = len(sorted_by_size)
    start_index = int(math.floor(count * float(size_distribution.rank_trim_lower)))
    stop_index = int(math.ceil(count * float(size_distribution.rank_trim_upper)))
    start_index = max(0, min(start_index, count - 1))
    stop_index = max(start_index + 1, min(stop_index, count))
    return sorted_by_size[start_index:stop_index]

def order_particle_pool_for_placement(
    records: Sequence[ParticleRecord],
    size_distribution: SizeDistributionConfig,
    rng: random.Random,
) -> List[ParticleRecord]:
    ordered_by_size = sorted(
        records,
        key=lambda item: (particle_size_rank_value(item), item.analytic_volume_um3),
        reverse=True,
    )
    if size_distribution.placement_order == "descending":
        return list(ordered_by_size)
    if size_distribution.placement_order == "random":
        shuffled = list(ordered_by_size)
        rng.shuffle(shuffled)
        return shuffled

    bin_count = max(1, min(int(size_distribution.placement_size_bins), len(ordered_by_size)))
    bins: List[List[ParticleRecord]] = [[] for _ in range(bin_count)]
    for index, record in enumerate(ordered_by_size):
        bin_index = min(bin_count - 1, int(index * bin_count / len(ordered_by_size)))
        bins[bin_index].append(record)

    for group in bins:
        rng.shuffle(group)

    # Start around the middle-small sizes, then alternate upward/downward.
    start_bin = min(bin_count - 1, max(0, int(round(bin_count * 0.55))))
    bin_visit_order = [start_bin]
    for distance in range(1, bin_count):
        lower = start_bin - distance
        upper = start_bin + distance
        if lower >= 0:
            bin_visit_order.append(lower)
        if upper < bin_count:
            bin_visit_order.append(upper)
    bin_visit_order = list(dict.fromkeys(bin_visit_order))

    interleaved: List[ParticleRecord] = []
    remaining = sum(len(group) for group in bins)
    while remaining > 0:
        for bin_index in bin_visit_order:
            if bins[bin_index]:
                interleaved.append(bins[bin_index].pop())
                remaining -= 1
    return interleaved

def sample_orientation_deg(
    config: SimulationConfig, rng: random.Random
) -> Float3:
    if config.orientation_distribution.mode == "homogeneous":
        return normalize_angles_deg(config.fixed_angles_deg)

    means = config.orientation_distribution.mean_deg
    stds = config.orientation_distribution.std_deg
    sampled = [
        rng.gauss(means[0], stds[0]),
        rng.gauss(means[1], stds[1]),
        rng.gauss(means[2], stds[2]),
    ]
    return normalize_angles_deg(sampled)

def generate_particle_pool(
    config: SimulationConfig,
    target_particle_volume_um3: float,
    template_cache: Dict[Tuple[float, ...], np.ndarray],
    rng: random.Random,
) -> List[ParticleRecord]:
    base_axes = base_semi_axes_um(config)
    base_major_diameter_um = 2.0 * float(np.max(base_axes))
    overlap_inflation = 2.0 / max(EPS, 2.0 - config.allowed_overlap)
    required_pool_volume = (
        target_particle_volume_um3
        * overlap_inflation
        * config.particle_pool_factor
    )

    if target_particle_volume_um3 <= 0.0:
        return []

    records: List[ParticleRecord] = []
    accumulated_volume = 0.0
    effective_records: List[ParticleRecord] = []
    effective_volume = 0.0
    max_particles = 200000

    while effective_volume < required_pool_volume:
        if len(records) >= max_particles:
            raise RuntimeError(
                "Particle pool exceeded the safety limit. "
                "Consider increasing voxel size or particle size."
            )

        scale_factor = sample_size_factor(
            config.size_distribution,
            base_major_diameter_um,
            rng,
        )
        semi_axes_um = tuple((base_axes * scale_factor).tolist())
        semi_axes_vox = tuple(
            (np.asarray(semi_axes_um, dtype=float) / config.voxel_size_um).tolist()
        )
        angles_deg = sample_orientation_deg(config, rng)

        key = template_key(semi_axes_vox, angles_deg)
        template = template_cache.get(key)
        if template is None:
            template = build_ellipsoid_template(semi_axes_vox, angles_deg)
            template_cache[key] = template

        analytic_volume = ellipsoid_volume_um3(semi_axes_um)
        records.append(
            ParticleRecord(
                particle_id=0,
                scale_factor=scale_factor,
                semi_axes_um=(
                    float(semi_axes_um[0]),
                    float(semi_axes_um[1]),
                    float(semi_axes_um[2]),
                ),
                semi_axes_vox=(
                    float(semi_axes_vox[0]),
                    float(semi_axes_vox[1]),
                    float(semi_axes_vox[2]),
                ),
                angles_deg=angles_deg,
                analytic_volume_um3=analytic_volume,
                template_voxels=int(template.shape[0]),
                template=template,
            )
        )
        accumulated_volume += analytic_volume
        if accumulated_volume >= required_pool_volume and (
            len(records) % 50 == 0 or len(records) >= max_particles
        ):
            effective_records = trim_lognormal_particle_pool_by_size_rank(
                records,
                config.size_distribution,
            )
            effective_volume = particle_pool_volume(effective_records)

    effective_records = trim_lognormal_particle_pool_by_size_rank(
        records,
        config.size_distribution,
    )
    effective_volume = particle_pool_volume(effective_records)
    if effective_volume < required_pool_volume:
        raise RuntimeError(
            "Trimmed particle pool volume is still below the required volume. "
            "Increase particle_pool_factor or relax size_distribution rank trimming."
        )

    if len(effective_records) != len(records):
        print(
            "颗粒池尺寸裁剪 / Particle size-rank trimming: "
            f"sampled={len(records)}, kept={len(effective_records)}, "
            f"rank_window={config.size_distribution.rank_trim_lower:.2f}-"
            f"{config.size_distribution.rank_trim_upper:.2f}, "
            f"candidate_volume={accumulated_volume:.3g} um^3, "
            f"kept_volume={effective_volume:.3g} um^3"
        )

    records = order_particle_pool_for_placement(
        effective_records,
        config.size_distribution,
        rng,
    )
    print(
        "颗粒放置顺序 / Particle placement order: "
        f"{config.size_distribution.placement_order}, "
        f"size_bins={config.size_distribution.placement_size_bins}"
    )
    for particle_id, record in enumerate(records, start=1):
        record.particle_id = particle_id
    return records

def bounded_template_coordinates(
    template: np.ndarray,
    center_vox: Sequence[int],
    shape: Int3,
    wrap_axes: WrapAxes,
) -> np.ndarray:
    coords = template.astype(np.int64, copy=False) + np.asarray(center_vox, dtype=np.int64)
    valid = np.ones(coords.shape[0], dtype=bool)
    for axis, wrap in enumerate(wrap_axes):
        if wrap:
            coords[:, axis] %= shape[axis]
        else:
            valid &= (coords[:, axis] >= 0) & (coords[:, axis] < shape[axis])
    if not np.all(valid):
        coords = coords[valid]
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.unique(coords, axis=0)

def wrapped_template_coordinates(
    template: np.ndarray, center_vox: Sequence[int], shape: Int3
) -> np.ndarray:
    return bounded_template_coordinates(template, center_vox, shape, (True, True, True))

def default_center_bounds(shape: Int3) -> CenterBounds3:
    return ((0, int(shape[0])), (0, int(shape[1])), (0, int(shape[2])))

def normalize_center_bounds(center_bounds: Sequence[Tuple[int, int]]) -> CenterBounds3:
    if len(center_bounds) != 3:
        raise ValueError("center_bounds must contain exactly 3 axis bounds.")
    return (
        (int(center_bounds[0][0]), int(center_bounds[0][1])),
        (int(center_bounds[1][0]), int(center_bounds[1][1])),
        (int(center_bounds[2][0]), int(center_bounds[2][1])),
    )

def random_center_in_bounds(
    center_bounds: CenterBounds3, rng: random.Random
) -> Int3:
    values: List[int] = []
    for axis, bounds in enumerate(center_bounds):
        lower = int(bounds[0])
        upper = int(bounds[1])
        if upper <= lower:
            raise ValueError(f"center_bounds axis {axis} is empty.")
        values.append(rng.randrange(lower, upper))
    return (values[0], values[1], values[2])

def perturb_center_in_bounds(
    center: Int3,
    step: int,
    center_bounds: CenterBounds3,
    shape: Int3,
    wrap_axes: WrapAxes,
    rng: random.Random,
) -> Int3:
    updated: List[int] = []
    for axis in range(3):
        lower = int(center_bounds[axis][0])
        upper = int(center_bounds[axis][1])
        candidate = int(center[axis]) + rng.randint(-step, step)
        if wrap_axes[axis]:
            width = max(1, upper - lower)
            candidate = lower + ((candidate - lower) % width)
        else:
            candidate = max(lower, min(candidate, upper - 1))
        candidate = max(0, min(candidate, int(shape[axis]) - 1))
        updated.append(candidate)
    return (updated[0], updated[1], updated[2])

def enforce_template_min_z(
    center: Int3,
    template: np.ndarray,
    shape: Int3,
    min_z_inclusive: Optional[int],
) -> Optional[Int3]:
    if min_z_inclusive is None or template.size == 0:
        return center
    template_min_z = int(np.min(template[:, 2]))
    required_center_z = int(min_z_inclusive) - template_min_z
    if required_center_z > int(shape[2]) - 1:
        return None
    adjusted_z = max(int(center[2]), required_center_z)
    adjusted_z = min(adjusted_z, int(shape[2]) - 1)
    return (int(center[0]), int(center[1]), adjusted_z)

def place_particle(
    occupied: np.ndarray,
    record: ParticleRecord,
    shape: Int3,
    config: SimulationConfig,
    rng: random.Random,
    center_bounds: Optional[CenterBounds3] = None,
    min_z_inclusive: Optional[int] = None,
    wrap_axes: WrapAxes = (True, True, True),
    count_z_range: Optional[Tuple[int, int]] = None,
    timeout_deadline_s: Optional[float] = None,
) -> Tuple[int, int]:
    active_center_bounds = (
        normalize_center_bounds(center_bounds)
        if center_bounds is not None
        else default_center_bounds(shape)
    )
    center = enforce_template_min_z(
        random_center_in_bounds(active_center_bounds, rng),
        record.template,
        shape,
        min_z_inclusive,
    )
    if center is None:
        record.placed = False
        record.center_vox = None
        record.attempts = 0
        record.overlap_ratio = 1.0
        return 0, 0
    step = max(
        1,
        int(math.ceil(max(record.semi_axes_vox) * config.perturbation_fraction)),
    )
    best_overlap = 1.0

    for attempt in range(1, config.max_placement_attempts + 1):
        if timeout_deadline_s is not None and time.time() > timeout_deadline_s:
            raise TimeoutError(
                "Per-structure timeout reached during particle placement."
            )
        coords = bounded_template_coordinates(record.template, center, shape, wrap_axes)
        if coords.size == 0:
            updated_center = enforce_template_min_z(
                perturb_center_in_bounds(
                    center,
                    step,
                    active_center_bounds,
                    shape,
                    wrap_axes,
                    rng,
                ),
                record.template,
                shape,
                min_z_inclusive,
            )
            if updated_center is None:
                break
            center = updated_center
            continue
        overlap_mask = occupied[coords[:, 0], coords[:, 1], coords[:, 2]]
        overlap_count = int(np.count_nonzero(overlap_mask))
        overlap_ratio = overlap_count / max(1, int(coords.shape[0]))

        if overlap_ratio <= config.allowed_overlap:
            new_coords = coords[~overlap_mask]
            new_cells = int(new_coords.shape[0])
            if count_z_range is None:
                new_cells_in_range = new_cells
            else:
                z_start = int(count_z_range[0])
                z_end = int(count_z_range[1])
                if z_end <= z_start or new_coords.size == 0:
                    new_cells_in_range = 0
                else:
                    new_cells_in_range = int(
                        np.count_nonzero(
                            (new_coords[:, 2] >= z_start)
                            & (new_coords[:, 2] < z_end)
                        )
                    )
            occupied[coords[:, 0], coords[:, 1], coords[:, 2]] = True
            record.placed = True
            record.center_vox = center
            record.attempts = attempt
            record.overlap_ratio = overlap_ratio
            return new_cells, new_cells_in_range

        best_overlap = min(best_overlap, overlap_ratio)
        updated_center = enforce_template_min_z(
            perturb_center_in_bounds(
                center,
                step,
                active_center_bounds,
                shape,
                wrap_axes,
                rng,
            ),
            record.template,
            shape,
            min_z_inclusive,
        )
        if updated_center is None:
            break
        center = updated_center

    record.placed = False
    record.center_vox = None
    record.attempts = config.max_placement_attempts
    record.overlap_ratio = best_overlap
    return 0, 0

def build_gradient_slice_specs(
    config: SimulationConfig,
    shape: Int3,
) -> List[GradientSliceSpec]:
    z_ranges = gradient_slice_ranges(int(shape[2]), int(config.gradient_slice_count))
    major_diameters = gradient_slice_major_diameters_um(config)
    particle_volume_fractions = gradient_slice_particle_volume_fractions(config)
    slice_specs: List[GradientSliceSpec] = []
    voxel_volume_um3 = float(config.voxel_size_um ** 3)
    for slice_index, (
        (z_start, z_end),
        major_diameter_um,
        slice_particle_volume_fraction,
    ) in enumerate(
        zip(z_ranges, major_diameters, particle_volume_fractions),
        start=1,
    ):
        if z_end <= z_start:
            raise ValueError("Gradient slice generation produced an empty z-range.")
        slice_volume_voxels = (
            int(shape[0]) * int(shape[1]) * max(0, int(z_end) - int(z_start))
        )
        slice_target_voxels = int(
            round(slice_volume_voxels * float(slice_particle_volume_fraction))
        )
        center_bounds = (
            (0, int(shape[0])),
            (0, int(shape[1])),
            (int(z_start), int(z_end)),
        )
        slice_specs.append(
            {
                "slice_index": slice_index,
                "z_start": int(z_start),
                "z_end": int(z_end),
                "thickness_vox": int(z_end - z_start),
                "particle_volume_fraction": float(slice_particle_volume_fraction),
                "major_diameter_um": float(major_diameter_um),
                "base_axes_um": scaled_base_axes_um_for_major_diameter(
                    config, float(major_diameter_um)
                ),
                "target_particle_voxels": int(slice_target_voxels),
                "target_particle_volume_um3": float(slice_target_voxels) * voxel_volume_um3,
                "center_bounds": center_bounds,
                "min_z_inclusive": int(z_start),
            }
        )
    return slice_specs

def template_to_local_grid(template: np.ndarray) -> np.ndarray:
    mins = template.min(axis=0)
    shifted = template - mins
    shape = tuple((shifted.max(axis=0) + 1).tolist())
    grid = np.zeros(shape, dtype=np.uint8)
    grid[shifted[:, 0], shifted[:, 1], shifted[:, 2]] = 1
    return grid
