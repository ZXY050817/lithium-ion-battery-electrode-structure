#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script requires numpy. Install it with: pip install numpy") from exc


Float3 = Tuple[float, float, float]
FloatQuad = Tuple[Float3, Float3, Float3, Float3]
Int3 = Tuple[int, int, int]
IntQuad = Tuple[Int3, Int3, Int3, Int3]
PHASE_ELECTROLYTE = 0
PHASE_ACTIVE = 1
PHASE_CBD = 2
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = DEFAULT_PROJECT_ROOT / "Structure_Data"
DEFAULT_VOXEL_SIZE_UM = 0.5
DEFAULT_MIN_CBD_COMPONENT_VOXELS = 20


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert TXT/NPY voxel electrode structures to two separate STL files: "
            "active particles and CBD. The default logic removes isolated CBD "
            "components smaller than 20 voxels and exports only external faces, so "
            "face-adjacent voxels inside the same phase do not create internal walls."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input structure file (.txt slice file or .npy structure file).",
    )
    parser.add_argument(
        "--voxel-size-um",
        type=float,
        default=DEFAULT_VOXEL_SIZE_UM,
        help=f"Voxel size in micrometers. Default: {DEFAULT_VOXEL_SIZE_UM}.",
    )
    parser.add_argument(
        "--initial-dir",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Initial directory for the interactive file chooser. Default: {DEFAULT_INPUT_ROOT}.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix inserted before the phase name in exported STL filenames.",
    )
    parser.add_argument(
        "--min-cbd-component-voxels",
        type=int,
        default=DEFAULT_MIN_CBD_COMPONENT_VOXELS,
        help=(
            "Remove CBD components smaller than this voxel count. Components use "
            "6-neighbor connectivity: only face contact counts; edge and point "
            f"contact do not. Default: {DEFAULT_MIN_CBD_COMPONENT_VOXELS}."
        ),
    )
    parser.add_argument(
        "--cbd-random-seed",
        type=int,
        default=12345,
        help=(
            "Random seed for CBD volume-preserving growth/erosion after small "
            "component cleanup. Default: 12345."
        ),
    )
    return parser.parse_args(argv)


def normalize_user_path(text: str) -> str:
    return text.strip().strip('"').strip("'")


def choose_input_file_interactively(initial_dir: Path) -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Select structure file",
            initialdir=str(initial_dir),
            filetypes=(
                ("Structure files", "*.txt *.npy"),
                ("TXT slice files", "*.txt"),
                ("NPY files", "*.npy"),
                ("All files", "*.*"),
            ),
        )
        root.destroy()
        if selected:
            return Path(selected)
    except Exception:
        pass

    manual_path = normalize_user_path(input("Enter structure file path (.txt or .npy): "))
    if not manual_path:
        raise SystemExit("No input file was selected.")
    return Path(manual_path)


def load_phase_slices_txt(path: Path) -> np.ndarray:
    layers: List[List[List[int]]] = []
    current_layer: List[List[int]] = []

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            if current_layer:
                layers.append(current_layer)
                current_layer = []
            continue

        try:
            row = [int(value) for value in line.split()]
        except ValueError as exc:
            raise ValueError(f"Invalid integer value in {path} at line {line_number}.") from exc
        if row:
            current_layer.append(row)

    if current_layer:
        layers.append(current_layer)
    if not layers:
        raise ValueError(f"No voxel slices were found in {path}.")

    row_count = len(layers[0])
    col_count = len(layers[0][0])
    for layer_index, layer in enumerate(layers, start=1):
        if len(layer) != row_count:
            raise ValueError(
                f"Layer {layer_index} has {len(layer)} rows; expected {row_count}."
            )
        for row_index, row in enumerate(layer, start=1):
            if len(row) != col_count:
                raise ValueError(
                    f"Layer {layer_index}, row {row_index} has {len(row)} columns; "
                    f"expected {col_count}."
                )

    # TXT files are written as z-stacked x:y matrices, so z becomes axis 2.
    return np.stack([np.asarray(layer, dtype=np.uint8) for layer in layers], axis=2)


def decode_phase_array(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array)
    if data.ndim == 3:
        phase = data
    elif data.ndim == 4 and data.shape[-1] == 3:
        phase = np.argmax(data, axis=-1)
    elif data.ndim == 4 and data.shape[0] == 3:
        phase = np.argmax(data, axis=0)
    else:
        raise ValueError(
            "NPY structure must be a 3D phase array, a channel-last 4D array "
            "(Nx, Ny, Nz, 3), or a channel-first 4D array (3, Nx, Ny, Nz)."
        )

    phase = np.asarray(phase, dtype=np.uint8)
    invalid_values = sorted(int(value) for value in np.unique(phase) if value not in {0, 1, 2})
    if invalid_values:
        raise ValueError(f"Structure contains invalid phase values: {invalid_values}.")
    return phase


def load_phase_labels(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return decode_phase_array(load_phase_slices_txt(path))
    if suffix == ".npy":
        return decode_phase_array(np.load(path))
    raise ValueError(f"Unsupported input file type: {path.suffix}. Use .txt or .npy.")


def face_neighbor_offsets() -> List[Int3]:
    return [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]


def remove_small_cbd_components(
    cbd_mask: np.ndarray,
    min_voxels: int,
) -> Tuple[np.ndarray, int, int]:
    mask = np.asarray(cbd_mask, dtype=bool)
    if min_voxels <= 1 or not np.any(mask):
        return mask.copy(), 0, 0

    nx, ny, nz = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    keep = np.zeros(mask.shape, dtype=bool)
    offsets = face_neighbor_offsets()
    removed_components = 0
    removed_voxels = 0

    for start_x, start_y, start_z in np.argwhere(mask):
        start = (int(start_x), int(start_y), int(start_z))
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        component: List[Tuple[int, int, int]] = []

        while stack:
            x, y, z = stack.pop()
            component.append((x, y, z))
            for dx, dy, dz in offsets:
                nx_i = x + dx
                ny_i = y + dy
                nz_i = z + dz
                if not (0 <= nx_i < nx and 0 <= ny_i < ny and 0 <= nz_i < nz):
                    continue
                neighbor = (nx_i, ny_i, nz_i)
                if visited[neighbor] or not mask[neighbor]:
                    continue
                visited[neighbor] = True
                stack.append(neighbor)

        if len(component) >= int(min_voxels):
            for voxel in component:
                keep[voxel] = True
        else:
            removed_components += 1
            removed_voxels += len(component)

    return keep, removed_components, removed_voxels


def random_growth_candidates(
    cbd_mask: np.ndarray,
    electrolyte_mask: np.ndarray,
) -> List[Tuple[int, int, int]]:
    cbd = np.asarray(cbd_mask, dtype=bool)
    electrolyte = np.asarray(electrolyte_mask, dtype=bool)
    nx, ny, nz = cbd.shape
    candidates: set[Tuple[int, int, int]] = set()

    for x, y, z in np.argwhere(cbd):
        x_i = int(x)
        y_i = int(y)
        z_i = int(z)
        for dx, dy, dz in face_neighbor_offsets():
            neighbor = (x_i + dx, y_i + dy, z_i + dz)
            nx_i, ny_i, nz_i = neighbor
            if not (0 <= nx_i < nx and 0 <= ny_i < ny and 0 <= nz_i < nz):
                continue
            if electrolyte[neighbor]:
                candidates.add(neighbor)

    return list(candidates)


def random_surface_erosion_candidates(cbd_mask: np.ndarray) -> List[Tuple[int, int, int]]:
    cbd = np.asarray(cbd_mask, dtype=bool)
    nx, ny, nz = cbd.shape
    candidates: List[Tuple[int, int, int]] = []

    for x, y, z in np.argwhere(cbd):
        x_i = int(x)
        y_i = int(y)
        z_i = int(z)
        is_surface = False
        for dx, dy, dz in face_neighbor_offsets():
            nx_i = x_i + dx
            ny_i = y_i + dy
            nz_i = z_i + dz
            if not (0 <= nx_i < nx and 0 <= ny_i < ny and 0 <= nz_i < nz):
                is_surface = True
                break
            if not cbd[nx_i, ny_i, nz_i]:
                is_surface = True
                break
        if is_surface:
            candidates.append((x_i, y_i, z_i))

    return candidates


def balance_cbd_volume_by_random_growth_erosion(
    cbd_mask: np.ndarray,
    phase_labels: np.ndarray,
    target_voxels: int,
    random_seed: int,
) -> Tuple[np.ndarray, int, int, int]:
    balanced = np.asarray(cbd_mask, dtype=bool).copy()
    electrolyte_mask = phase_labels == PHASE_ELECTROLYTE
    rng = np.random.default_rng(int(random_seed))
    grown_voxels = 0
    eroded_voxels = 0

    while int(np.count_nonzero(balanced)) < int(target_voxels):
        candidates = random_growth_candidates(balanced, electrolyte_mask)
        if not candidates:
            fallback_candidates = list(map(tuple, np.argwhere(electrolyte_mask)))
            if not fallback_candidates:
                break
            candidates = [
                (int(x), int(y), int(z))
                for x, y, z in fallback_candidates
                if not balanced[int(x), int(y), int(z)]
            ]
            if not candidates:
                break

        choice_index = int(rng.integers(0, len(candidates)))
        chosen = candidates[choice_index]
        balanced[chosen] = True
        electrolyte_mask[chosen] = False
        grown_voxels += 1

    while int(np.count_nonzero(balanced)) > int(target_voxels):
        candidates = random_surface_erosion_candidates(balanced)
        if not candidates:
            candidates = [
                (int(x), int(y), int(z))
                for x, y, z in np.argwhere(balanced)
            ]
            if not candidates:
                break

        choice_index = int(rng.integers(0, len(candidates)))
        chosen = candidates[choice_index]
        balanced[chosen] = False
        electrolyte_mask[chosen] = True
        eroded_voxels += 1

    final_voxels = int(np.count_nonzero(balanced))
    return balanced, grown_voxels, eroded_voxels, final_voxels


def write_ascii_triangle(
    handle,
    normal: Float3,
    v0: Float3,
    v1: Float3,
    v2: Float3,
) -> None:
    handle.write(
        f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n"
    )
    handle.write("    outer loop\n")
    handle.write(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
    handle.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
    handle.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
    handle.write("    endloop\n")
    handle.write("  endfacet\n")


def scaled_quad(coords: IntQuad, voxel_size_um: float) -> FloatQuad:
    h = float(voxel_size_um)
    return (
        (float(coords[0][0]) * h, float(coords[0][1]) * h, float(coords[0][2]) * h),
        (float(coords[1][0]) * h, float(coords[1][1]) * h, float(coords[1][2]) * h),
        (float(coords[2][0]) * h, float(coords[2][1]) * h, float(coords[2][2]) * h),
        (float(coords[3][0]) * h, float(coords[3][1]) * h, float(coords[3][2]) * h),
    )


def write_quad(
    handle,
    normal: Float3,
    quad: FloatQuad,
) -> None:
    write_ascii_triangle(handle, normal, quad[0], quad[1], quad[2])
    write_ascii_triangle(handle, normal, quad[0], quad[2], quad[3])


def write_voxel_surface_stl(
    occupied: np.ndarray,
    voxel_size_um: float,
    output_path: Path,
    solid_name: str,
) -> int:
    mask = np.asarray(occupied, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("occupied mask must be 3D.")

    face_specs: List[Tuple[Float3, Int3, IntQuad]] = [
        ((1.0, 0.0, 0.0), (1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        ((-1.0, 0.0, 0.0), (-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
        ((0.0, 1.0, 0.0), (0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
        ((0.0, -1.0, 0.0), (0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        ((0.0, 0.0, 1.0), (0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
        ((0.0, 0.0, -1.0), (0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
    ]

    nx, ny, nz = mask.shape
    face_count = 0
    with output_path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"solid {solid_name}\n")
        for x, y, z in np.argwhere(mask):
            x_i = int(x)
            y_i = int(y)
            z_i = int(z)
            for normal, direction, quad_offsets in face_specs:
                neighbor_x = x_i + direction[0]
                neighbor_y = y_i + direction[1]
                neighbor_z = z_i + direction[2]
                if (
                    0 <= neighbor_x < nx
                    and 0 <= neighbor_y < ny
                    and 0 <= neighbor_z < nz
                    and mask[neighbor_x, neighbor_y, neighbor_z]
                ):
                    continue

                quad_coords: IntQuad = (
                    (
                        x_i + quad_offsets[0][0],
                        y_i + quad_offsets[0][1],
                        z_i + quad_offsets[0][2],
                    ),
                    (
                        x_i + quad_offsets[1][0],
                        y_i + quad_offsets[1][1],
                        z_i + quad_offsets[1][2],
                    ),
                    (
                        x_i + quad_offsets[2][0],
                        y_i + quad_offsets[2][1],
                        z_i + quad_offsets[2][2],
                    ),
                    (
                        x_i + quad_offsets[3][0],
                        y_i + quad_offsets[3][1],
                        z_i + quad_offsets[3][2],
                    ),
                )
                quad = scaled_quad(quad_coords, voxel_size_um)
                write_quad(handle, normal, quad)
                face_count += 1
        handle.write(f"endsolid {solid_name}\n")

    return face_count


def export_phase_stl(
    phase_mask: np.ndarray,
    voxel_size_um: float,
    input_path: Path,
    output_suffix: str,
    phase_name: str,
) -> Optional[Path]:
    voxel_count = int(np.count_nonzero(phase_mask))
    if voxel_count == 0:
        print(f"Warning: {phase_name} has no voxels; STL skipped.")
        return None

    output_path = input_path.parent / f"{input_path.stem}{output_suffix}_{phase_name}.stl"
    face_count = write_voxel_surface_stl(
        phase_mask,
        voxel_size_um,
        output_path,
        phase_name,
    )
    print(
        f"  Exported {phase_name}: {voxel_count} voxels, "
        f"{face_count} external voxel faces, {face_count * 2} triangles"
    )
    return output_path


def export_structure_stls(
    phase_labels: np.ndarray,
    voxel_size_um: float,
    input_path: Path,
    output_suffix: str,
    min_cbd_component_voxels: int,
    cbd_random_seed: int,
) -> List[Path]:
    active_mask = phase_labels == PHASE_ACTIVE
    cbd_mask = phase_labels == PHASE_CBD
    original_cbd_voxels = int(np.count_nonzero(cbd_mask))
    cleaned_cbd_mask, removed_components, removed_voxels = remove_small_cbd_components(
        cbd_mask,
        min_cbd_component_voxels,
    )
    balanced_cbd_mask, grown_voxels, eroded_voxels, final_cbd_voxels = (
        balance_cbd_volume_by_random_growth_erosion(
            cleaned_cbd_mask,
            phase_labels,
            original_cbd_voxels,
            cbd_random_seed,
        )
    )
    print(
        "CBD cleanup: "
        f"input={original_cbd_voxels}, "
        f"kept={int(np.count_nonzero(cleaned_cbd_mask))}, "
        f"removed_components={removed_components}, "
        f"removed_voxels={removed_voxels}, "
        f"threshold={int(min_cbd_component_voxels)}, "
        "connectivity=face-only"
    )
    print(
        "CBD volume balance: "
        f"target={original_cbd_voxels}, "
        f"grown={grown_voxels}, "
        f"eroded={eroded_voxels}, "
        f"final={final_cbd_voxels}, "
        f"random_seed={int(cbd_random_seed)}"
    )

    outputs: List[Path] = []
    active_path = export_phase_stl(
        active_mask,
        voxel_size_um,
        input_path,
        output_suffix,
        "active_phase",
    )
    if active_path is not None:
        outputs.append(active_path)

    cbd_path = export_phase_stl(
        balanced_cbd_mask,
        voxel_size_um,
        input_path,
        output_suffix,
        "cbd_phase",
    )
    if cbd_path is not None:
        outputs.append(cbd_path)

    if not outputs:
        raise ValueError("No active or CBD voxels were found, so no STL files were created.")
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_path = args.input if args.input is not None else choose_input_file_interactively(args.initial_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if float(args.voxel_size_um) <= 0.0:
        raise ValueError("voxel_size_um must be positive.")
    if int(args.min_cbd_component_voxels) < 1:
        raise ValueError("min_cbd_component_voxels must be at least 1.")

    phase_labels = load_phase_labels(input_path)
    outputs = export_structure_stls(
        phase_labels,
        float(args.voxel_size_um),
        input_path,
        str(args.output_suffix).strip(),
        int(args.min_cbd_component_voxels),
        int(args.cbd_random_seed),
    )

    print(f"Input file: {input_path}")
    print(f"Voxel size: {float(args.voxel_size_um)} um")
    print(f"Grid shape: {tuple(int(value) for value in phase_labels.shape)}")
    print("Exported STL files:")
    for output in outputs:
        print(f"  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
