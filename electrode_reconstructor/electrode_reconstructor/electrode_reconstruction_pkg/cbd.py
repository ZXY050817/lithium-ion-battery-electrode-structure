from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .particles import config_wrap_axes
from .shared import (
    CBDGrowthStats,
    EPS,
    Int3,
    PHASE_ACTIVE,
    PHASE_CBD,
    PHASE_ELECTROLYTE,
    SimulationConfig,
    WrapAxes,
)

def neighborhood_offsets(radius: int) -> Tuple[Int3, ...]:
    offsets: List[Int3] = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offsets.append((dx, dy, dz))
    return tuple(offsets)

CBD_NEIGHBOR_OFFSETS = neighborhood_offsets(1)

CBD_CANDIDATE_OFFSETS = neighborhood_offsets(2)

CBD_ENERGY_NORMALIZER = max(1.0, len(CBD_NEIGHBOR_OFFSETS) / 2.0)

FACE_NEIGHBOR_OFFSETS: Tuple[Int3, ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)

def in_bounds(index: Int3, shape: Int3) -> bool:
    return (
        0 <= index[0] < shape[0]
        and 0 <= index[1] < shape[1]
        and 0 <= index[2] < shape[2]
    )

def remove_small_cbd_components_face_connected(
    cbd_phase: np.ndarray,
    min_voxels: int,
) -> Tuple[np.ndarray, int, int]:
    mask = np.asarray(cbd_phase, dtype=bool)
    if min_voxels <= 1 or not np.any(mask):
        return mask.copy(), 0, 0

    shape: Int3 = (int(mask.shape[0]), int(mask.shape[1]), int(mask.shape[2]))
    visited = np.zeros(mask.shape, dtype=bool)
    kept = np.zeros(mask.shape, dtype=bool)
    removed_components = 0
    removed_voxels = 0

    for x, y, z in np.argwhere(mask):
        start: Int3 = (int(x), int(y), int(z))
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        component: List[Int3] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for offset in FACE_NEIGHBOR_OFFSETS:
                neighbor: Int3 = (
                    current[0] + offset[0],
                    current[1] + offset[1],
                    current[2] + offset[2],
                )
                if not in_bounds(neighbor, shape):
                    continue
                if visited[neighbor] or not mask[neighbor]:
                    continue
                visited[neighbor] = True
                stack.append(neighbor)

        if len(component) >= int(min_voxels):
            for voxel in component:
                kept[voxel] = True
        else:
            removed_components += 1
            removed_voxels += len(component)

    return kept, removed_components, removed_voxels

def cbd_growth_candidates_face_connected(
    cbd_phase: np.ndarray,
    available_phase: np.ndarray,
) -> List[Int3]:
    shape: Int3 = (
        int(cbd_phase.shape[0]),
        int(cbd_phase.shape[1]),
        int(cbd_phase.shape[2]),
    )
    candidates: Set[Int3] = set()
    for x, y, z in np.argwhere(cbd_phase):
        current: Int3 = (int(x), int(y), int(z))
        for offset in FACE_NEIGHBOR_OFFSETS:
            neighbor: Int3 = (
                current[0] + offset[0],
                current[1] + offset[1],
                current[2] + offset[2],
            )
            if in_bounds(neighbor, shape) and available_phase[neighbor]:
                candidates.add(neighbor)
    return list(candidates)

def cbd_surface_erosion_candidates(cbd_phase: np.ndarray) -> List[Int3]:
    shape: Int3 = (
        int(cbd_phase.shape[0]),
        int(cbd_phase.shape[1]),
        int(cbd_phase.shape[2]),
    )
    candidates: List[Int3] = []
    for x, y, z in np.argwhere(cbd_phase):
        current: Int3 = (int(x), int(y), int(z))
        for offset in FACE_NEIGHBOR_OFFSETS:
            neighbor: Int3 = (
                current[0] + offset[0],
                current[1] + offset[1],
                current[2] + offset[2],
            )
            if not in_bounds(neighbor, shape) or not cbd_phase[neighbor]:
                candidates.append(current)
                break
    return candidates

def pop_random_valid_voxel(
    candidates: List[Int3],
    candidate_set: Set[Int3],
    valid_mask: np.ndarray,
    rng: random.Random,
) -> Optional[Int3]:
    while candidates:
        index = rng.randrange(len(candidates))
        chosen = candidates[index]
        candidates[index] = candidates[-1]
        candidates.pop()
        candidate_set.discard(chosen)
        if valid_mask[chosen]:
            return chosen
    return None

def add_growth_frontier_neighbors(
    frontier: List[Int3],
    frontier_set: Set[Int3],
    center: Int3,
    available: np.ndarray,
    shape: Int3,
) -> None:
    for offset in FACE_NEIGHBOR_OFFSETS:
        neighbor: Int3 = (
            center[0] + offset[0],
            center[1] + offset[1],
            center[2] + offset[2],
        )
        if not in_bounds(neighbor, shape):
            continue
        if available[neighbor] and neighbor not in frontier_set:
            frontier_set.add(neighbor)
            frontier.append(neighbor)

def add_erosion_frontier_neighbors(
    frontier: List[Int3],
    frontier_set: Set[Int3],
    center: Int3,
    cbd_phase: np.ndarray,
    shape: Int3,
) -> None:
    for offset in FACE_NEIGHBOR_OFFSETS:
        neighbor: Int3 = (
            center[0] + offset[0],
            center[1] + offset[1],
            center[2] + offset[2],
        )
        if not in_bounds(neighbor, shape):
            continue
        if cbd_phase[neighbor] and neighbor not in frontier_set:
            frontier_set.add(neighbor)
            frontier.append(neighbor)

def balance_cbd_volume_after_cleanup(
    cbd_phase: np.ndarray,
    active_phase: np.ndarray,
    target_voxels: int,
    rng: random.Random,
    wrap_axes: WrapAxes = (True, True, True),
) -> Tuple[np.ndarray, int, int, int]:
    balanced = np.asarray(cbd_phase, dtype=bool).copy()
    active = np.asarray(active_phase, dtype=bool)
    available = ~(active | balanced)
    regrown_voxels = 0
    eroded_voxels = 0
    shape: Int3 = (
        int(balanced.shape[0]),
        int(balanced.shape[1]),
        int(balanced.shape[2]),
    )
    current_voxels = int(np.count_nonzero(balanced))
    growth_frontier = cbd_growth_candidates_face_connected(balanced, available)
    growth_frontier_set: Set[Int3] = set(growth_frontier)
    erosion_frontier = cbd_surface_erosion_candidates(balanced)
    erosion_frontier_set: Set[Int3] = set(erosion_frontier)
    progress_total = abs(int(target_voxels) - current_voxels)
    progress_step = max(1, int(math.ceil(max(1, progress_total) * 0.1)))
    next_progress = progress_step

    while current_voxels < int(target_voxels):
        candidate = pop_random_valid_voxel(
            growth_frontier,
            growth_frontier_set,
            available,
            rng,
        )
        if candidate is None:
            active_neighbor_count = accumulate_neighbor_count(
                active, FACE_NEIGHBOR_OFFSETS, wrap_axes
            )
            seed_candidates_array = np.argwhere(available & (active_neighbor_count > 0))
            if seed_candidates_array.size == 0:
                seed_candidates_array = np.argwhere(available)
            if seed_candidates_array.size == 0:
                break
            chosen_array = seed_candidates_array[
                rng.randrange(int(seed_candidates_array.shape[0]))
            ]
            chosen: Int3 = (
                int(chosen_array[0]),
                int(chosen_array[1]),
                int(chosen_array[2]),
            )
        else:
            chosen = candidate

        balanced[chosen] = True
        available[chosen] = False
        regrown_voxels += 1
        current_voxels += 1
        add_growth_frontier_neighbors(
            growth_frontier,
            growth_frontier_set,
            chosen,
            available,
            shape,
        )
        if chosen not in erosion_frontier_set:
            erosion_frontier_set.add(chosen)
            erosion_frontier.append(chosen)
        if regrown_voxels >= next_progress and progress_total > 0:
            print(
                "CBD清理后再生进度 / CBD cleanup regrowth progress: "
                f"regrown={regrown_voxels}/{progress_total}, "
                f"current={current_voxels}, target={target_voxels}"
            )
            next_progress += progress_step

    while current_voxels > int(target_voxels):
        candidate = pop_random_valid_voxel(
            erosion_frontier,
            erosion_frontier_set,
            balanced,
            rng,
        )
        if candidate is None:
            all_cbd = np.argwhere(balanced)
            if all_cbd.size == 0:
                break
            chosen_array = all_cbd[rng.randrange(int(all_cbd.shape[0]))]
            chosen: Int3 = (
                int(chosen_array[0]),
                int(chosen_array[1]),
                int(chosen_array[2]),
            )
        else:
            chosen = candidate

        balanced[chosen] = False
        available[chosen] = True
        eroded_voxels += 1
        current_voxels -= 1
        add_erosion_frontier_neighbors(
            erosion_frontier,
            erosion_frontier_set,
            chosen,
            balanced,
            shape,
        )
        if chosen not in growth_frontier_set:
            growth_frontier_set.add(chosen)
            growth_frontier.append(chosen)
        if eroded_voxels >= next_progress and progress_total > 0:
            print(
                "CBD清理后剥蚀进度 / CBD cleanup erosion progress: "
                f"eroded={eroded_voxels}/{progress_total}, "
                f"current={current_voxels}, target={target_voxels}"
            )
            next_progress += progress_step

    return balanced, regrown_voxels, eroded_voxels, current_voxels

def cleanup_and_balance_cbd_phase(
    cbd_phase: np.ndarray,
    active_phase: np.ndarray,
    config: SimulationConfig,
    rng: random.Random,
) -> Tuple[np.ndarray, Dict[str, int]]:
    wrap_axes = config_wrap_axes(config)
    target_voxels = int(np.count_nonzero(cbd_phase))
    cleaned, removed_components, removed_voxels = remove_small_cbd_components_face_connected(
        cbd_phase,
        int(config.cbd_growth.cleanup_min_component_voxels),
    )
    balanced, regrown_voxels, eroded_voxels, final_voxels = (
        balance_cbd_volume_after_cleanup(
            cleaned,
            active_phase,
            target_voxels,
            rng,
            wrap_axes,
        )
    )
    return balanced, {
        "removed_components": int(removed_components),
        "removed_voxels": int(removed_voxels),
        "regrown_voxels": int(regrown_voxels),
        "eroded_voxels": int(eroded_voxels),
        "final_voxels": int(final_voxels),
    }

def roll_mask(
    mask: np.ndarray, offset: Int3, wrap_axes: WrapAxes = (True, True, True)
) -> np.ndarray:
    rolled = np.asarray(mask).copy()
    for axis, shift in enumerate(offset):
        if shift == 0:
            continue
        rolled = np.roll(rolled, shift, axis=axis)
        if wrap_axes[axis]:
            continue
        invalid_slice = [slice(None)] * rolled.ndim
        if shift > 0:
            invalid_slice[axis] = slice(0, shift)
        else:
            invalid_slice[axis] = slice(shift, None)
        rolled[tuple(invalid_slice)] = 0
    return rolled

def accumulate_neighbor_count(
    mask: np.ndarray,
    offsets: Sequence[Int3],
    wrap_axes: WrapAxes = (True, True, True),
) -> np.ndarray:
    counts = np.zeros(mask.shape, dtype=np.uint16)
    mask_uint = mask.astype(np.uint8, copy=False)
    for offset in offsets:
        counts += roll_mask(mask_uint, offset, wrap_axes)
    return counts

def dilate_mask(
    mask: np.ndarray,
    offsets: Sequence[Int3],
    wrap_axes: WrapAxes = (True, True, True),
) -> np.ndarray:
    dilated = mask.copy()
    for offset in offsets:
        dilated |= roll_mask(mask, offset, wrap_axes)
    return dilated

def wrap_index(
    coord: Int3,
    offset: Int3,
    shape: Int3,
    wrap_axes: WrapAxes = (True, True, True),
) -> Optional[Int3]:
    values: List[int] = []
    for axis in range(3):
        candidate = int(coord[axis]) + int(offset[axis])
        if wrap_axes[axis]:
            candidate %= int(shape[axis])
        elif candidate < 0 or candidate >= int(shape[axis]):
            return None
        values.append(candidate)
    return (values[0], values[1], values[2])

def flat_index(coord: Int3, shape: Int3) -> int:
    return coord[0] * shape[1] * shape[2] + coord[1] * shape[2] + coord[2]

def unflatten_index(flat_idx: int, shape: Int3) -> Int3:
    yz = shape[1] * shape[2]
    x = flat_idx // yz
    remainder = flat_idx % yz
    y = remainder // shape[2]
    z = remainder % shape[2]
    return (int(x), int(y), int(z))

def fenwick_build(weights: np.ndarray) -> np.ndarray:
    tree = np.zeros(weights.size + 1, dtype=np.float64)
    tree[1:] = weights
    for index in range(1, weights.size + 1):
        parent = index + (index & -index)
        if parent <= weights.size:
            tree[parent] += tree[index]
    return tree

def fenwick_add(tree: np.ndarray, index: int, delta: float) -> None:
    if abs(delta) <= EPS:
        return
    fenwick_index = index + 1
    while fenwick_index < tree.size:
        tree[fenwick_index] += delta
        fenwick_index += fenwick_index & -fenwick_index

def fenwick_find_by_cumulative(tree: np.ndarray, target: float) -> int:
    size = tree.size - 1
    index = 0
    bit = 1 << max(0, size.bit_length() - 1)
    while bit > 0:
        next_index = index + bit
        if next_index <= size and tree[next_index] <= target:
            target -= float(tree[next_index])
            index = next_index
        bit >>= 1
    return index

def build_phase_labels(active_phase: np.ndarray, cbd_phase: np.ndarray) -> np.ndarray:
    labels = np.full(active_phase.shape, PHASE_ELECTROLYTE, dtype=np.uint8)
    labels[active_phase] = PHASE_ACTIVE
    labels[cbd_phase] = PHASE_CBD
    return labels

def cbd_growth_exponent(morphology_parameter: float) -> float:
    return 5.0 * ((0.15 / 5.0) ** morphology_parameter)

def grow_cbd_phase(
    active_phase: np.ndarray,
    config: SimulationConfig,
    rng: random.Random,
) -> Tuple[np.ndarray, CBDGrowthStats]:
    shape = active_phase.shape
    total_voxels = int(active_phase.size)
    target_cbd_voxels = int(round(total_voxels * config.cbd_volume_fraction))
    cbd_phase = np.zeros(shape, dtype=bool)
    stats: CBDGrowthStats = {
        "target_cbd_voxels": target_cbd_voxels,
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

    if target_cbd_voxels <= 0:
        return cbd_phase, stats

    pore_phase = ~active_phase
    wrap_axes = config_wrap_axes(config)
    particle_neighbor_count = accumulate_neighbor_count(
        active_phase, CBD_NEIGHBOR_OFFSETS, wrap_axes
    )
    growth_start = time.time()
    surface_electrolyte = pore_phase & (particle_neighbor_count > 0)
    surface_indices = np.flatnonzero(surface_electrolyte)

    initial_seed_budget = int(
        math.ceil(config.cbd_growth.seed_fraction * max(1, target_cbd_voxels))
    )
    initial_seed_count = min(
        target_cbd_voxels, initial_seed_budget, int(surface_indices.size)
    )
    if initial_seed_count > 0:
        chosen_seed_indices = rng.sample(surface_indices.tolist(), initial_seed_count)
        cbd_phase.reshape(-1)[chosen_seed_indices] = True

    stats["seeded_cbd_voxels"] = int(cbd_phase.sum())
    current_solid = active_phase | cbd_phase
    candidate_mask = (
        dilate_mask(current_solid, CBD_CANDIDATE_OFFSETS, wrap_axes) & (~current_solid)
    )
    cbd_neighbor_count = accumulate_neighbor_count(
        cbd_phase, CBD_NEIGHBOR_OFFSETS, wrap_axes
    )

    morphology_parameter = config.cbd_growth.morphology_parameter
    exponent = cbd_growth_exponent(morphology_parameter)
    particle_bias = 1.0 - morphology_parameter
    flat_active_phase = active_phase.reshape(-1)
    flat_cbd_phase = cbd_phase.reshape(-1)
    flat_particle_neighbor_count = particle_neighbor_count.reshape(-1)
    flat_cbd_neighbor_count = cbd_neighbor_count.reshape(-1)
    candidate_weights = np.zeros(total_voxels, dtype=np.float64)
    initial_candidate_indices = np.flatnonzero(candidate_mask.reshape(-1))
    initial_energies = (
        flat_cbd_neighbor_count[initial_candidate_indices].astype(np.float64)
        + particle_bias
        * flat_particle_neighbor_count[initial_candidate_indices].astype(np.float64)
    ) / CBD_ENERGY_NORMALIZER
    positive_initial_mask = initial_energies > 0.0
    if np.any(positive_initial_mask):
        positive_candidate_indices = initial_candidate_indices[positive_initial_mask]
        candidate_weights[positive_candidate_indices] = np.power(
            initial_energies[positive_initial_mask], exponent
        )
    candidate_tree = fenwick_build(candidate_weights)
    candidate_count = int(np.count_nonzero(candidate_weights > 0.0))
    total_candidate_weight = float(candidate_weights.sum())
    cbd_voxels = int(cbd_phase.sum())
    growth_iterations = 0
    stop_reason = "target_reached"
    progress_step_voxels = max(1, int(math.ceil(target_cbd_voxels * 0.1)))
    next_progress_mark = progress_step_voxels

    print(
        "CBD初始化 / CBD initialization: "
        f"target={target_cbd_voxels}, "
        f"surface_sites={int(surface_indices.size)}, "
        f"seeded={cbd_voxels}, "
        f"candidate_sites={candidate_count}"
    )
    if cbd_voxels > 0:
        seeded_fraction = 100.0 * cbd_voxels / max(1, target_cbd_voxels)
        print(
            "CBD进度 / CBD progress: "
            f"{cbd_voxels}/{target_cbd_voxels} "
            f"({seeded_fraction:.1f}%), "
            f"iterations={growth_iterations}, "
            f"candidate_sites={candidate_count}, "
            f"elapsed={time.time() - growth_start:.2f}s"
        )
    while next_progress_mark <= cbd_voxels:
        next_progress_mark += progress_step_voxels

    while cbd_voxels < target_cbd_voxels:
        if growth_iterations >= config.cbd_growth.max_growth_steps:
            stop_reason = "max_growth_steps_reached"
            break
        if candidate_count <= 0 or total_candidate_weight <= EPS:
            stop_reason = "non_positive_growth_energy"
            break

        growth_iterations += 1
        random_value = rng.random() * total_candidate_weight
        selected_flat = fenwick_find_by_cumulative(candidate_tree, random_value)
        selected_weight = float(candidate_weights[selected_flat])
        if selected_weight <= 0.0:
            stop_reason = "non_positive_growth_energy"
            break

        candidate_weights[selected_flat] = 0.0
        total_candidate_weight -= selected_weight
        candidate_count -= 1
        fenwick_add(candidate_tree, selected_flat, -selected_weight)

        selected_coord = unflatten_index(selected_flat, shape)
        cbd_phase[selected_coord] = True
        cbd_voxels += 1
        flat_cbd_phase[selected_flat] = True

        impacted_candidate_indices: Set[int] = set()
        for offset in CBD_NEIGHBOR_OFFSETS:
            neighbor = wrap_index(selected_coord, offset, shape, wrap_axes)
            if neighbor is None:
                continue
            neighbor_flat = flat_index(neighbor, shape)
            flat_cbd_neighbor_count[neighbor_flat] += 1
            if candidate_weights[neighbor_flat] > 0.0:
                impacted_candidate_indices.add(neighbor_flat)

        for offset in CBD_CANDIDATE_OFFSETS:
            neighbor = wrap_index(selected_coord, offset, shape, wrap_axes)
            if neighbor is None:
                continue
            neighbor_flat = flat_index(neighbor, shape)
            if not flat_active_phase[neighbor_flat] and not flat_cbd_phase[neighbor_flat]:
                impacted_candidate_indices.add(neighbor_flat)

        for candidate_flat in impacted_candidate_indices:
            if flat_active_phase[candidate_flat] or flat_cbd_phase[candidate_flat]:
                continue
            updated_energy = (
                float(flat_cbd_neighbor_count[candidate_flat])
                + particle_bias * float(flat_particle_neighbor_count[candidate_flat])
            ) / CBD_ENERGY_NORMALIZER
            updated_weight = (
                math.pow(updated_energy, exponent) if updated_energy > 0.0 else 0.0
            )
            previous_weight = float(candidate_weights[candidate_flat])
            if updated_weight > 0.0 and previous_weight <= 0.0:
                candidate_count += 1
            elif updated_weight <= 0.0 and previous_weight > 0.0:
                candidate_count -= 1
            weight_delta = updated_weight - previous_weight
            if abs(weight_delta) > EPS:
                candidate_weights[candidate_flat] = updated_weight
                total_candidate_weight += weight_delta
                fenwick_add(candidate_tree, candidate_flat, weight_delta)

        while cbd_voxels >= next_progress_mark and next_progress_mark <= target_cbd_voxels:
            progress_fraction = 100.0 * cbd_voxels / max(1, target_cbd_voxels)
            print(
                "CBD进度 / CBD progress: "
                f"{cbd_voxels}/{target_cbd_voxels} "
                f"({progress_fraction:.1f}%), "
                f"iterations={growth_iterations}, "
                f"candidate_sites={candidate_count}, "
                f"elapsed={time.time() - growth_start:.2f}s"
            )
            next_progress_mark += progress_step_voxels

    grown_cbd_voxels = cbd_voxels - int(stats["seeded_cbd_voxels"])
    stats["grown_cbd_voxels"] = grown_cbd_voxels
    stats["growth_iterations"] = growth_iterations
    stats["candidate_voxels_final"] = candidate_count
    stats["growth_terminated_early"] = cbd_voxels < target_cbd_voxels
    stats["growth_stop_reason"] = stop_reason
    return cbd_phase, stats
