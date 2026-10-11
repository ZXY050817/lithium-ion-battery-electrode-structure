#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


Float2 = Tuple[float, float]
Int2 = Tuple[int, int]

EPS = 1.0e-12
PHASE_ELECTROLYTE = 0
PHASE_ACTIVE = 1
PHASE_CBD = 2

NEIGHBOR_4: Tuple[Int2, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBOR_8: Tuple[Int2, ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass
class ReconstructionParams2D:
    domain_size_um: Float2
    voxel_size_um: float
    particle_area_fraction: float
    cbd_area_fraction: float
    allowed_overlap: float
    base_axes_um: Float2
    particle_size_mean_factor: float
    particle_size_sigma_factor: float
    orientation_mean_deg: float
    orientation_std_deg: float
    max_placement_attempts: int
    particle_pool_factor: float
    cbd_morphology_parameter: float
    cbd_seed_fraction: float
    cbd_max_growth_steps: int
    cbd_cleanup_min_component_pixels: int
    preserve_active_fraction_after_connectivity: bool
    random_seed: Optional[int]


@dataclass
class ParticleRecord2D:
    particle_id: int
    analytic_area_um2: float
    template_offsets: np.ndarray
    placed: bool = False


def to_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc


def to_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def to_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"{name} must be boolean.")


def as_float2(value: Any, name: str) -> Float2:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain two numeric values.")
    if len(value) != 2:
        raise ValueError(f"{name} must contain two numeric values.")
    return (
        to_float(value[0], f"{name}[0]"),
        to_float(value[1], f"{name}[1]"),
    )


def required(data: Dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required config key: {key}")
    return data[key]


def params_from_dict(data: Dict[str, Any]) -> ReconstructionParams2D:
    params = ReconstructionParams2D(
        domain_size_um=as_float2(required(data, "domain_size_um"), "domain_size_um"),
        voxel_size_um=to_float(required(data, "voxel_size_um"), "voxel_size_um"),
        particle_area_fraction=to_float(
            required(data, "particle_area_fraction"), "particle_area_fraction"
        ),
        cbd_area_fraction=to_float(
            required(data, "cbd_area_fraction"), "cbd_area_fraction"
        ),
        allowed_overlap=to_float(required(data, "allowed_overlap"), "allowed_overlap"),
        base_axes_um=as_float2(required(data, "base_axes_um"), "base_axes_um"),
        particle_size_mean_factor=to_float(
            required(data, "particle_size_mean_factor"), "particle_size_mean_factor"
        ),
        particle_size_sigma_factor=to_float(
            required(data, "particle_size_sigma_factor"), "particle_size_sigma_factor"
        ),
        orientation_mean_deg=to_float(
            required(data, "orientation_mean_deg"), "orientation_mean_deg"
        ),
        orientation_std_deg=to_float(
            required(data, "orientation_std_deg"), "orientation_std_deg"
        ),
        max_placement_attempts=to_int(
            required(data, "max_placement_attempts"), "max_placement_attempts"
        ),
        particle_pool_factor=to_float(
            required(data, "particle_pool_factor"), "particle_pool_factor"
        ),
        cbd_morphology_parameter=to_float(
            required(data, "cbd_morphology_parameter"), "cbd_morphology_parameter"
        ),
        cbd_seed_fraction=to_float(
            required(data, "cbd_seed_fraction"), "cbd_seed_fraction"
        ),
        cbd_max_growth_steps=to_int(
            required(data, "cbd_max_growth_steps"), "cbd_max_growth_steps"
        ),
        cbd_cleanup_min_component_pixels=to_int(
            required(data, "cbd_cleanup_min_component_pixels"),
            "cbd_cleanup_min_component_pixels",
        ),
        preserve_active_fraction_after_connectivity=to_bool(
            required(data, "preserve_active_fraction_after_connectivity"),
            "preserve_active_fraction_after_connectivity",
        ),
        random_seed=(
            None
            if required(data, "random_seed") is None
            else to_int(required(data, "random_seed"), "random_seed")
        ),
    )
    validate_params(params)
    return params


def validate_params(params: ReconstructionParams2D) -> None:
    if params.voxel_size_um <= 0.0:
        raise ValueError("voxel_size_um must be positive.")
    if min(params.domain_size_um) <= 0.0:
        raise ValueError("domain_size_um values must be positive.")
    if not (0.0 <= params.particle_area_fraction < 1.0):
        raise ValueError("particle_area_fraction must be in [0, 1).")
    if not (0.0 <= params.cbd_area_fraction < 1.0):
        raise ValueError("cbd_area_fraction must be in [0, 1).")
    if params.particle_area_fraction + params.cbd_area_fraction >= 1.0:
        raise ValueError(
            "particle_area_fraction + cbd_area_fraction must be less than 1."
        )
    if not (0.0 <= params.allowed_overlap < 1.0):
        raise ValueError("allowed_overlap must be in [0, 1).")
    if min(params.base_axes_um) <= 0.0:
        raise ValueError("base_axes_um values must be positive.")
    if params.particle_size_mean_factor <= 0.0:
        raise ValueError("particle_size_mean_factor must be positive.")
    if params.particle_size_sigma_factor < 0.0:
        raise ValueError("particle_size_sigma_factor must be non-negative.")
    if params.orientation_std_deg < 0.0:
        raise ValueError("orientation_std_deg must be non-negative.")
    if params.max_placement_attempts <= 0:
        raise ValueError("max_placement_attempts must be positive.")
    if params.particle_pool_factor < 1.0:
        raise ValueError("particle_pool_factor must be >= 1.")
    if params.cbd_morphology_parameter < 0.0:
        raise ValueError("cbd_morphology_parameter must be non-negative.")
    if not (0.0 < params.cbd_seed_fraction <= 1.0):
        raise ValueError("cbd_seed_fraction must be in (0, 1].")
    if params.cbd_max_growth_steps <= 0:
        raise ValueError("cbd_max_growth_steps must be positive.")
    if params.cbd_cleanup_min_component_pixels < 1:
        raise ValueError("cbd_cleanup_min_component_pixels must be >= 1.")


def grid_shape(params: ReconstructionParams2D) -> Int2:
    nx = max(1, int(round(params.domain_size_um[0] / params.voxel_size_um)))
    ny = max(1, int(round(params.domain_size_um[1] / params.voxel_size_um)))
    return (ny, nx)


def normalize_angle_deg(angle: float) -> float:
    normalized = math.fmod(float(angle), 360.0)
    if normalized < 0.0:
        normalized += 360.0
    return normalized


def sample_lognormal_factor(
    mean_factor: float,
    sigma_factor: float,
    rng: random.Random,
) -> float:
    if sigma_factor == 0.0:
        return max(mean_factor, EPS)
    variance = sigma_factor ** 2
    sigma_ln_sq = math.log(1.0 + variance / (mean_factor ** 2))
    sigma_ln = math.sqrt(max(sigma_ln_sq, 0.0))
    mu_ln = math.log(mean_factor) - 0.5 * sigma_ln_sq
    return max(rng.lognormvariate(mu_ln, sigma_ln), EPS)


def sample_orientation_deg(
    mean_deg: float,
    std_deg: float,
    rng: random.Random,
) -> float:
    if std_deg == 0.0:
        return normalize_angle_deg(mean_deg)
    return normalize_angle_deg(rng.gauss(mean_deg, std_deg))


def build_ellipse_template(semi_axes_vox: Float2, angle_deg: float) -> np.ndarray:
    a = max(float(semi_axes_vox[0]), EPS)
    b = max(float(semi_axes_vox[1]), EPS)
    radius = int(math.ceil(max(a, b))) + 2
    values = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(values, values, indexing="ij")
    theta = math.radians(float(angle_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    local_x = cos_t * xx + sin_t * yy
    local_y = -sin_t * xx + cos_t * yy
    inside = (local_x / a) ** 2 + (local_y / b) ** 2 <= 1.0
    offsets = np.column_stack((yy[inside], xx[inside])).astype(np.int16)
    if offsets.size == 0:
        return np.asarray([[0, 0]], dtype=np.int16)
    return offsets


def template_key(semi_axes_vox: Float2, angle_deg: float) -> Tuple[float, float, float]:
    return (
        round(float(semi_axes_vox[0]), 4),
        round(float(semi_axes_vox[1]), 4),
        round(float(angle_deg), 4),
    )


def wrapped_template_indices(
    shape: Int2,
    center_yx: Int2,
    offsets: np.ndarray,
) -> np.ndarray:
    ny, nx = shape
    yy = (int(center_yx[0]) + offsets[:, 0].astype(np.int64)) % ny
    xx = (int(center_yx[1]) + offsets[:, 1].astype(np.int64)) % nx
    linear = yy * nx + xx
    return np.unique(linear)


def place_particle(
    active_mask: np.ndarray,
    record: ParticleRecord2D,
    rng: random.Random,
    allowed_overlap: float,
    max_attempts: int,
) -> bool:
    ny, nx = int(active_mask.shape[0]), int(active_mask.shape[1])
    flat = active_mask.reshape(-1)
    best_overlap = 1.0
    best_indices: Optional[np.ndarray] = None

    for _ in range(max_attempts):
        center = (rng.randrange(ny), rng.randrange(nx))
        indices = wrapped_template_indices((ny, nx), center, record.template_offsets)
        overlap_count = int(np.count_nonzero(flat[indices]))
        overlap_ratio = overlap_count / max(1, int(indices.size))
        if overlap_ratio < best_overlap:
            best_overlap = overlap_ratio
            best_indices = indices
        if overlap_ratio <= allowed_overlap:
            flat[indices] = True
            record.placed = True
            return True

    if best_indices is not None:
        flat[best_indices] = True
        record.placed = True
        return True
    return False


def neighbor_count(mask: np.ndarray, offsets: Sequence[Int2]) -> np.ndarray:
    count = np.zeros(mask.shape, dtype=np.int16)
    ny, nx = mask.shape
    for y in range(ny):
        for x in range(nx):
            if not mask[y, x]:
                continue
            for dy, dx in offsets:
                yy = y + dy
                xx = x + dx
                if 0 <= yy < ny and 0 <= xx < nx:
                    count[yy, xx] += 1
    return count


def active_boundary_indices_from_mask(active_mask: np.ndarray) -> List[int]:
    if not np.any(active_mask):
        return []
    boundary_mask = active_mask & (neighbor_count(active_mask, NEIGHBOR_4) < 4)
    indices = np.flatnonzero(boundary_mask)
    if indices.size == 0:
        indices = np.flatnonzero(active_mask)
    return indices.tolist()


def attached_active_growth_candidates(
    active_mask: np.ndarray,
    available_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if not np.any(active_mask):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    active_neighbors = neighbor_count(active_mask, NEIGHBOR_8)
    candidate_mask = available_mask & (active_neighbors > 0)
    candidate_indices = np.flatnonzero(candidate_mask).astype(np.int64, copy=False)
    if candidate_indices.size == 0:
        return candidate_indices, np.empty(0, dtype=np.float64)
    neighbor_weights = active_neighbors.reshape(-1)[candidate_indices].astype(
        np.float64, copy=False
    )
    return candidate_indices, neighbor_weights


def choose_weighted_linear_index(
    candidate_indices: np.ndarray,
    weights: np.ndarray,
    rng: random.Random,
) -> Optional[int]:
    if candidate_indices.size == 0:
        return None
    clipped = np.maximum(weights.astype(np.float64, copy=False), EPS)
    total = float(np.sum(clipped))
    if total <= EPS:
        return int(candidate_indices[rng.randrange(int(candidate_indices.size))])
    draw = rng.random() * total
    cumulative = 0.0
    for index, weight in zip(candidate_indices.tolist(), clipped.tolist()):
        cumulative += float(weight)
        if draw <= cumulative:
            return int(index)
    return int(candidate_indices[-1])


def connected_components(
    mask: np.ndarray,
    offsets: Sequence[Int2],
) -> List[np.ndarray]:
    visited = np.zeros(mask.shape, dtype=bool)
    components: List[np.ndarray] = []
    ny, nx = mask.shape
    for start_y, start_x in np.argwhere(mask):
        y0 = int(start_y)
        x0 = int(start_x)
        if visited[y0, x0]:
            continue
        queue: deque[Int2] = deque([(y0, x0)])
        visited[y0, x0] = True
        coords: List[Int2] = []
        while queue:
            y, x = queue.popleft()
            coords.append((y, x))
            for dy, dx in offsets:
                yy = y + dy
                xx = x + dx
                if (
                    0 <= yy < ny
                    and 0 <= xx < nx
                    and mask[yy, xx]
                    and not visited[yy, xx]
                ):
                    visited[yy, xx] = True
                    queue.append((yy, xx))
        components.append(np.asarray(coords, dtype=np.int32))
    return components


def remove_small_components(
    mask: np.ndarray,
    min_component_pixels: int,
    offsets: Sequence[Int2],
) -> Tuple[np.ndarray, int, int]:
    if min_component_pixels <= 1:
        return mask.copy(), 0, 0
    cleaned = mask.copy()
    removed_components = 0
    removed_pixels = 0
    for component in connected_components(mask, offsets):
        if int(component.shape[0]) >= min_component_pixels:
            continue
        cleaned[component[:, 0], component[:, 1]] = False
        removed_components += 1
        removed_pixels += int(component.shape[0])
    return cleaned, removed_components, removed_pixels


def balance_active_pixels_unconstrained(
    active_mask: np.ndarray,
    target_active_pixels: int,
    rng: random.Random,
) -> None:
    current = int(np.count_nonzero(active_mask))
    while current > target_active_pixels:
        boundary_indices = active_boundary_indices_from_mask(active_mask)
        if not boundary_indices:
            break
        chosen = int(boundary_indices[rng.randrange(len(boundary_indices))])
        active_mask.reshape(-1)[chosen] = False
        current -= 1

    while current < target_active_pixels:
        candidate_indices, neighbor_weights = attached_active_growth_candidates(
            active_mask, ~active_mask
        )
        chosen = choose_weighted_linear_index(candidate_indices, neighbor_weights, rng)
        if chosen is None:
            break
        active_mask.reshape(-1)[chosen] = True
        current += 1

    cleaned, _, removed_pixels = remove_small_components(active_mask, 2, NEIGHBOR_8)
    if removed_pixels > 0:
        active_mask[:, :] = cleaned
        current = int(np.count_nonzero(active_mask))
        while current < target_active_pixels:
            candidate_indices, neighbor_weights = attached_active_growth_candidates(
                active_mask, ~active_mask
            )
            chosen = choose_weighted_linear_index(
                candidate_indices, neighbor_weights, rng
            )
            if chosen is None:
                break
            active_mask.reshape(-1)[chosen] = True
            current += 1


def reconstruct_active_phase(
    params: ReconstructionParams2D,
    rng: random.Random,
) -> np.ndarray:
    shape = grid_shape(params)
    active_mask = np.zeros(shape, dtype=bool)
    domain_area_um2 = params.domain_size_um[0] * params.domain_size_um[1]
    target_area_um2 = domain_area_um2 * params.particle_area_fraction
    target_active_pixels = int(round(active_mask.size * params.particle_area_fraction))
    base_semi_axes_um = np.asarray(params.base_axes_um, dtype=np.float64) / 2.0
    base_major_diameter_um = 2.0 * float(np.max(base_semi_axes_um))
    required_pool_area = (
        target_area_um2
        * (2.0 / max(EPS, 2.0 - params.allowed_overlap))
        * params.particle_pool_factor
    )

    template_cache: Dict[Tuple[float, float, float], np.ndarray] = {}
    particle_records: List[ParticleRecord2D] = []
    pool_area = 0.0
    while pool_area < required_pool_area:
        scale_factor = sample_lognormal_factor(
            params.particle_size_mean_factor,
            params.particle_size_sigma_factor,
            rng,
        )
        semi_axes_um = base_semi_axes_um * scale_factor
        semi_axes_vox = (
            float(semi_axes_um[0] / params.voxel_size_um),
            float(semi_axes_um[1] / params.voxel_size_um),
        )
        angle_deg = sample_orientation_deg(
            params.orientation_mean_deg,
            params.orientation_std_deg,
            rng,
        )
        key = template_key(semi_axes_vox, angle_deg)
        template = template_cache.get(key)
        if template is None:
            template = build_ellipse_template(semi_axes_vox, angle_deg)
            template_cache[key] = template
        area_um2 = math.pi * float(semi_axes_um[0]) * float(semi_axes_um[1])
        particle_records.append(
            ParticleRecord2D(
                particle_id=0,
                analytic_area_um2=area_um2,
                template_offsets=template,
            )
        )
        pool_area += area_um2
        if len(particle_records) > 200000:
            raise RuntimeError("Particle pool exceeded safety limit.")

    particle_records.sort(key=lambda record: record.analytic_area_um2, reverse=True)
    for index, record in enumerate(particle_records, start=1):
        record.particle_id = index
        place_particle(
            active_mask,
            record,
            rng,
            params.allowed_overlap,
            params.max_placement_attempts,
        )
        if int(np.count_nonzero(active_mask)) >= target_active_pixels:
            break

    balance_active_pixels_unconstrained(active_mask, target_active_pixels, rng)
    return active_mask


def grow_cbd_phase(
    active_mask: np.ndarray,
    params: ReconstructionParams2D,
    rng: random.Random,
) -> np.ndarray:
    total_pixels = int(active_mask.size)
    target_cbd_pixels = int(round(total_pixels * params.cbd_area_fraction))
    cbd_mask = np.zeros(active_mask.shape, dtype=bool)
    if target_cbd_pixels <= 0:
        return cbd_mask

    pore_mask = ~active_mask
    active_neighbors = neighbor_count(active_mask, NEIGHBOR_8)
    surface_candidates = pore_mask & (active_neighbors > 0)
    surface_indices = np.flatnonzero(surface_candidates)
    if surface_indices.size == 0:
        return cbd_mask

    seed_count = max(1, int(round(target_cbd_pixels * params.cbd_seed_fraction)))
    seed_count = min(seed_count, int(surface_indices.size), target_cbd_pixels)
    shuffled_surface = surface_indices.tolist()
    rng.shuffle(shuffled_surface)
    cbd_mask.reshape(-1)[np.asarray(shuffled_surface[:seed_count], dtype=np.int64)] = True

    morphology = float(params.cbd_morphology_parameter)
    for _ in range(params.cbd_max_growth_steps):
        if int(np.count_nonzero(cbd_mask)) >= target_cbd_pixels:
            break
        current_solid = active_mask | cbd_mask
        candidates = (~current_solid) & (neighbor_count(current_solid, NEIGHBOR_8) > 0)
        candidate_indices = np.flatnonzero(candidates)
        if candidate_indices.size == 0:
            break
        active_count = neighbor_count(active_mask, NEIGHBOR_8).reshape(-1)
        cbd_count = neighbor_count(cbd_mask, NEIGHBOR_8).reshape(-1)
        raw_weights = (
            1.0
            + active_count[candidate_indices].astype(np.float64)
            + (1.0 + morphology) * cbd_count[candidate_indices].astype(np.float64)
        )
        weights = np.power(np.maximum(raw_weights, EPS), 1.0 + morphology)
        cumulative = np.cumsum(weights)
        offset = int(np.searchsorted(cumulative, rng.random() * float(cumulative[-1])))
        chosen_index = int(candidate_indices[min(offset, candidate_indices.size - 1)])
        cbd_mask.reshape(-1)[chosen_index] = True
    return cbd_mask


def cleanup_and_balance_cbd_phase(
    active_mask: np.ndarray,
    cbd_mask: np.ndarray,
    params: ReconstructionParams2D,
    rng: random.Random,
) -> np.ndarray:
    target_cbd_pixels = int(round(cbd_mask.size * params.cbd_area_fraction))
    cleaned_mask, _, _ = remove_small_components(
        cbd_mask,
        params.cbd_cleanup_min_component_pixels,
        NEIGHBOR_4,
    )

    balanced = cleaned_mask.copy()
    current = int(np.count_nonzero(balanced))
    flat = balanced.reshape(-1)
    active_flat = active_mask.reshape(-1)

    if current < target_cbd_pixels:
        need = target_cbd_pixels - current
        solid = active_mask | balanced
        candidate_mask = (~solid) & (neighbor_count(solid, NEIGHBOR_8) > 0)
        candidate_indices = np.flatnonzero(candidate_mask).tolist()
        rng.shuffle(candidate_indices)
        for index in candidate_indices[:need]:
            if not active_flat[index]:
                flat[index] = True
    elif current > target_cbd_pixels:
        remove_count = current - target_cbd_pixels
        cbd_indices = np.flatnonzero(flat).tolist()
        rng.shuffle(cbd_indices)
        for index in cbd_indices[:remove_count]:
            flat[index] = False

    return balanced


def reconstruct_path(
    end: Int2,
    parent_y: np.ndarray,
    parent_x: np.ndarray,
) -> List[Int2]:
    path: List[Int2] = []
    y, x = end
    while True:
        path.append((y, x))
        py = int(parent_y[y, x])
        px = int(parent_x[y, x])
        if py == y and px == x:
            break
        if py < 0 or px < 0:
            break
        y, x = py, px
    path.reverse()
    return path


def shortest_bridge_path(source_mask: np.ndarray, target_mask: np.ndarray) -> List[Int2]:
    ny, nx = source_mask.shape
    parent_y = np.full(source_mask.shape, -1, dtype=np.int32)
    parent_x = np.full(source_mask.shape, -1, dtype=np.int32)
    queue: deque[Int2] = deque()

    for y, x in np.argwhere(source_mask):
        yy = int(y)
        xx = int(x)
        parent_y[yy, xx] = yy
        parent_x[yy, xx] = xx
        queue.append((yy, xx))

    while queue:
        y, x = queue.popleft()
        if target_mask[y, x] and not source_mask[y, x]:
            return reconstruct_path((y, x), parent_y, parent_x)
        for dy, dx in NEIGHBOR_4:
            yy = y + dy
            xx = x + dx
            if 0 <= yy < ny and 0 <= xx < nx and parent_y[yy, xx] < 0:
                parent_y[yy, xx] = y
                parent_x[yy, xx] = x
                queue.append((yy, xx))
    return []


def largest_nonactive_component_mask(labels: np.ndarray) -> np.ndarray:
    components = connected_components(labels != PHASE_ACTIVE, NEIGHBOR_4)
    mask = np.zeros(labels.shape, dtype=bool)
    if not components:
        return mask
    largest = max(components, key=lambda component: int(component.shape[0]))
    mask[largest[:, 0], largest[:, 1]] = True
    return mask


def find_unreachable_electrolyte_components(
    labels: np.ndarray,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    main_nonactive_mask = largest_nonactive_component_mask(labels)
    unreachable: List[np.ndarray] = []
    for component in connected_components(labels == PHASE_ELECTROLYTE, NEIGHBOR_4):
        if component.size == 0:
            continue
        if not bool(np.any(main_nonactive_mask[component[:, 0], component[:, 1]])):
            unreachable.append(component)
    return main_nonactive_mask, unreachable


def repair_unreachable_electrolyte_components(
    labels: np.ndarray,
) -> np.ndarray:
    repaired = labels.copy()
    max_rounds = int(repaired.size)
    for _ in range(max_rounds):
        main_nonactive_mask, unreachable = find_unreachable_electrolyte_components(
            repaired
        )
        if not unreachable:
            break
        source_component = max(unreachable, key=lambda component: int(component.shape[0]))
        source_mask = np.zeros(repaired.shape, dtype=bool)
        source_mask[source_component[:, 0], source_component[:, 1]] = True
        path = shortest_bridge_path(source_mask, main_nonactive_mask)
        if not path:
            break
        changed = 0
        for y, x in path:
            if repaired[y, x] == PHASE_ACTIVE:
                repaired[y, x] = PHASE_ELECTROLYTE
                changed += 1
        if changed == 0:
            break
    return repaired


def connect_nonactive_network(labels: np.ndarray) -> np.ndarray:
    repaired = labels.copy()
    max_rounds = int(repaired.size)
    for _ in range(max_rounds):
        components = connected_components(repaired != PHASE_ACTIVE, NEIGHBOR_4)
        if len(components) <= 1:
            break
        components.sort(key=lambda component: int(component.shape[0]), reverse=True)
        main = components[0]
        source = np.zeros(repaired.shape, dtype=bool)
        source[main[:, 0], main[:, 1]] = True
        target = np.zeros(repaired.shape, dtype=bool)
        for component in components[1:]:
            target[component[:, 0], component[:, 1]] = True
        path = shortest_bridge_path(source, target)
        if not path:
            break
        changed = 0
        for y, x in path:
            if repaired[y, x] == PHASE_ACTIVE:
                repaired[y, x] = PHASE_CBD
                changed += 1
        if changed == 0:
            break
    return repaired


def is_nonactive_connected(labels: np.ndarray) -> bool:
    mask = labels != PHASE_ACTIVE
    if not np.any(mask):
        return False
    return len(connected_components(mask, NEIGHBOR_4)) == 1


def regrow_active_pixels_with_connectivity(
    labels: np.ndarray,
    target_active_pixels: int,
    rng: random.Random,
) -> np.ndarray:
    repaired = labels.copy()
    attempts = 0
    max_attempts = max(1000, int(repaired.size) * 20)
    while int(np.count_nonzero(repaired == PHASE_ACTIVE)) < target_active_pixels:
        if attempts >= max_attempts:
            break
        attempts += 1
        active_mask = repaired == PHASE_ACTIVE
        candidate_indices, weights = attached_active_growth_candidates(
            active_mask, repaired != PHASE_ACTIVE
        )
        chosen = choose_weighted_linear_index(candidate_indices, weights, rng)
        if chosen is None:
            break
        old_value = int(repaired.reshape(-1)[chosen])
        repaired.reshape(-1)[chosen] = PHASE_ACTIVE
        if is_nonactive_connected(repaired):
            attempts = 0
        else:
            repaired.reshape(-1)[chosen] = old_value
    return repaired


def preserve_active_pixel_count_with_connectivity(
    labels: np.ndarray,
    target_active_pixels: int,
    rng: random.Random,
) -> np.ndarray:
    repaired = labels.copy()
    while int(np.count_nonzero(repaired == PHASE_ACTIVE)) > target_active_pixels:
        candidates = active_boundary_indices_from_mask(repaired == PHASE_ACTIVE)
        if not candidates:
            break
        repaired.reshape(-1)[candidates[rng.randrange(len(candidates))]] = PHASE_ELECTROLYTE
    return regrow_active_pixels_with_connectivity(repaired, target_active_pixels, rng)


def balance_cbd_count_preserving_nonactive(
    labels: np.ndarray,
    target_cbd_pixels: int,
    rng: random.Random,
) -> np.ndarray:
    balanced = labels.copy()
    flat = balanced.reshape(-1)
    current = int(np.count_nonzero(balanced == PHASE_CBD))
    if current > target_cbd_pixels:
        cbd_indices = np.flatnonzero(flat == PHASE_CBD).tolist()
        rng.shuffle(cbd_indices)
        for index in cbd_indices[: current - target_cbd_pixels]:
            flat[index] = PHASE_ELECTROLYTE
    elif current < target_cbd_pixels:
        electrolyte_indices = np.flatnonzero(flat == PHASE_ELECTROLYTE).tolist()
        rng.shuffle(electrolyte_indices)
        for index in electrolyte_indices[: target_cbd_pixels - current]:
            flat[index] = PHASE_CBD
    return balanced


def enforce_nonactive_connectivity(
    labels: np.ndarray,
    target_active_pixels: int,
    target_cbd_pixels: int,
    preserve_active_fraction_after_connectivity: bool,
    rng: random.Random,
) -> np.ndarray:
    repaired = repair_unreachable_electrolyte_components(labels)
    repaired = connect_nonactive_network(repaired)
    if preserve_active_fraction_after_connectivity:
        repaired = preserve_active_pixel_count_with_connectivity(
            repaired, target_active_pixels, rng
        )
    repaired = balance_cbd_count_preserving_nonactive(repaired, target_cbd_pixels, rng)
    return repaired


def phase_counts(labels: np.ndarray) -> Dict[str, float]:
    total = int(labels.size)
    electrolyte = int(np.count_nonzero(labels == PHASE_ELECTROLYTE))
    active = int(np.count_nonzero(labels == PHASE_ACTIVE))
    cbd = int(np.count_nonzero(labels == PHASE_CBD))
    return {
        "total_pixels": float(total),
        "electrolyte_pixels": float(electrolyte),
        "active_pixels": float(active),
        "cbd_pixels": float(cbd),
        "electrolyte_area_fraction": electrolyte / max(1, total),
        "active_area_fraction": active / max(1, total),
        "cbd_area_fraction": cbd / max(1, total),
    }


def write_phase_txt(labels: np.ndarray, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in labels:
            handle.write(" ".join(str(int(value)) for value in row))
            handle.write("\n")


def run_reconstruction(
    params: ReconstructionParams2D,
    output_dir: Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    validate_params(params)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(params.random_seed)

    shape = grid_shape(params)
    target_active_pixels = int(round(shape[0] * shape[1] * params.particle_area_fraction))
    target_cbd_pixels = int(round(shape[0] * shape[1] * params.cbd_area_fraction))

    active_mask = reconstruct_active_phase(params, rng)
    cbd_mask = grow_cbd_phase(active_mask, params, rng)
    cbd_mask = cleanup_and_balance_cbd_phase(active_mask, cbd_mask, params, rng)

    labels = np.zeros(shape, dtype=np.uint8)
    labels[active_mask] = PHASE_ACTIVE
    labels[cbd_mask & ~active_mask] = PHASE_CBD
    labels = enforce_nonactive_connectivity(
        labels,
        target_active_pixels,
        target_cbd_pixels,
        params.preserve_active_fraction_after_connectivity,
        rng,
    )

    counts = phase_counts(labels)
    txt_path = output_dir / "phase_matrix.txt"
    write_phase_txt(labels, txt_path)

    summary = {
        "random_seed": params.random_seed,
        "grid_shape_yx": [int(shape[0]), int(shape[1])],
        "phase_counts": counts,
        "txt_path": str(txt_path),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return labels, summary


def load_params_from_json(config_path: Path) -> ReconstructionParams2D:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must contain an object.")
    return params_from_dict(payload)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal 2D three-phase electrode reconstruction core."
    )
    parser.add_argument("--config", type=Path, required=True, help="Config JSON path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for phase_matrix.txt and summary.json.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        params = load_params_from_json(args.config)
        _, summary = run_reconstruction(params, args.output_dir)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    counts = summary["phase_counts"]
    print(f"Output: {summary['txt_path']}")
    print(
        "Fractions: "
        f"electrolyte={counts['electrolyte_area_fraction']:.6f}, "
        f"active={counts['active_area_fraction']:.6f}, "
        f"cbd={counts['cbd_area_fraction']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
