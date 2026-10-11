from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .batch import random_batch_domain_size_um, random_batch_max_major_axis_limit_um
from .shared import (
    CBDGrowthConfig,
    DEFAULT_INVERSE_DIAMETER_BIN_START_UM,
    DEFAULT_INVERSE_DIAMETER_BIN_WIDTH_UM,
    DEFAULT_INVERSE_DIAMETER_PROBABILITIES,
    DEFAULT_OUTPUT_BASE_DIR,
    DEFAULT_SINGLE_OUTPUT_BASE_DIR,
    Float3,
    OrientationDistributionConfig,
    RANDOM_BATCH_MAX_MAJOR_AXIS_VOX,
    RandomBatchConfig,
    SimulationConfig,
    SizeDistributionConfig,
    default_random_seed,
    format_float3,
)
from .reconstructor import run_random_batch_reconstruction, run_reconstruction

def prompt_text(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default

def prompt_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")

def parse_float_list(text: str) -> List[float]:
    parts = [
        part.strip()
        for part in text.replace(";", ",").split(",")
        if part.strip()
    ]
    if not parts:
        raise ValueError("No numeric values were provided.")
    return [float(part) for part in parts]

def prompt_float_list(prompt: str, default: Sequence[float]) -> List[float]:
    default_text = ", ".join(f"{value:g}" for value in default)
    while True:
        raw = input(f"{prompt} [{default_text}]: ").strip()
        if not raw:
            return [float(value) for value in default]
        try:
            return parse_float_list(raw)
        except ValueError:
            print("Please enter a comma-separated list of valid numbers.")

def prompt_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return int(default)
        try:
            return int(raw)
        except ValueError:
            print("Please enter a valid integer.")

def prompt_triplet(prompt: str, default: Float3) -> Float3:
    default_text = ", ".join(f"{value:g}" for value in default)
    while True:
        raw = input(f"{prompt} [{default_text}]: ").strip()
        if not raw:
            return default
        parts = [part.strip() for part in raw.replace(";", ",").split(",")]
        if len(parts) != 3:
            print("Please enter exactly 3 comma-separated values.")
            continue
        try:
            return (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError:
            print("Please enter valid numeric values.")

def prompt_choice(prompt: str, choices: Sequence[str], default: str) -> str:
    normalized_choices = [str(choice) for choice in choices]
    choice_text = ", ".join(
        f"{index}. {choice}"
        for index, choice in enumerate(normalized_choices, start=1)
    )
    allowed = {choice.lower(): choice.lower() for choice in normalized_choices}
    default_lower = default.lower()
    default_index = next(
        (
            index
            for index, choice in enumerate(normalized_choices, start=1)
            if choice.lower() == default_lower
        ),
        None,
    )
    default_hint = (
        f"{default_index}. {default}"
        if default_index is not None
        else default
    )
    while True:
        raw = input(f"{prompt} ({choice_text}) [{default_hint}]: ").strip().lower()
        if not raw:
            return default_lower
        if raw.isdigit():
            choice_index = int(raw)
            if 1 <= choice_index <= len(normalized_choices):
                return normalized_choices[choice_index - 1].lower()
        if raw in allowed:
            return allowed[raw]
        print(f"Please enter a number or option name: {choice_text}")

def interactive_random_batch_config(
    default_output_dir: Path,
    electrode_mode: str,
) -> Tuple[RandomBatchConfig, Path]:
    print("Random batch generation mode enabled.")
    print(
        "This mode samples parameters internally and loops over many structures."
    )
    gradient_slice_count = 5
    gradient_slice1_major_diameter_um = 4.0
    gradient_sliceN_major_diameter_um = 2.0
    if electrode_mode == "gradient":
        gradient_slice_count = prompt_int(
            "Number of z-direction slices",
            5,
        )
        gradient_slice1_major_diameter_um = prompt_float(
            "Slice1 base major particle diameter (um)",
            4.0,
        )
        gradient_sliceN_major_diameter_um = prompt_float(
            "SliceN base major particle diameter (um)",
            2.0,
        )
    batch_output_dir = Path(
        prompt_text(
            "Batch output root directory",
            str(default_output_dir),
        )
    )

    batch_cfg = RandomBatchConfig(
        structure_count=prompt_int(
            "Number of random structures", 2000
        ),
        grid_shape_vox=(64, 64, 64),
        voxel_size_um=0.5,
        max_major_axis_um=10.0,
        per_structure_timeout_s=prompt_float(
            "Per-structure timeout (s)",
            30.0,
        ),
        electrode_mode=electrode_mode,
        gradient_slice_count=gradient_slice_count,
        gradient_slice1_major_diameter_um=gradient_slice1_major_diameter_um,
        gradient_sliceN_major_diameter_um=gradient_sliceN_major_diameter_um,
        output_prefix_base=prompt_text(
            "Batch output prefix base", "electrode_random"
        ),
        batch_seed=prompt_int(
            "Batch random seed", default_random_seed()
        ),
    )
    batch_cfg.validate()
    print(
        "Fixed constraints: "
        f"grid={batch_cfg.grid_shape_vox}, voxel={batch_cfg.voxel_size_um:.3f} um, "
        f"domain={format_float3(random_batch_domain_size_um(batch_cfg))} um, "
        f"max_major_axis={random_batch_max_major_axis_limit_um(batch_cfg):.3f} um "
        f"({RANDOM_BATCH_MAX_MAJOR_AXIS_VOX:.1f} vox)"
    )
    print("Randomized distributions: size=lognormal, orientation=normal")
    if batch_cfg.electrode_mode == "gradient":
        print(
            "Gradient settings: "
            f"slices={batch_cfg.gradient_slice_count}, "
            f"slice1_major={batch_cfg.gradient_slice1_major_diameter_um:.3f} um, "
            f"sliceN_major={batch_cfg.gradient_sliceN_major_diameter_um:.3f} um"
        )
    print(
        "Note: The base particle major axis is limited to 8 voxels, but actual particles "
        "can still exceed 8 voxels after lognormal size scaling."
    )
    print(f"Batch output root directory: {batch_output_dir}")
    return batch_cfg, batch_output_dir

def interactive_config(electrode_mode: str) -> SimulationConfig:
    print("Enter the reconstruction parameters. Press Enter to keep defaults.")

    domain_size_um = prompt_triplet(
        "Domain size in um (x, y, z)",
        (20.0, 20.0, 20.0),
    )

    gradient_slice1_particle_vf = 0.4
    gradient_sliceN_particle_vf = 0.4
    while True:
        particle_vf = prompt_float("Particle volume fraction", 0.4)
        cbd_vf = prompt_float("CBD volume fraction", 0.1)
        if electrode_mode == "gradient":
            gradient_slice1_particle_vf = prompt_float(
                "slice1 particle volume fraction",
                particle_vf,
            )
            gradient_sliceN_particle_vf = prompt_float(
                "sliceN particle volume fraction",
                particle_vf,
            )
            if gradient_slice1_particle_vf + cbd_vf > 1.0 + 1.0e-9:
                print(
                    "slice1 particle volume fraction plus CBD volume fraction must be at most 1."
                )
                continue
            if gradient_sliceN_particle_vf + cbd_vf > 1.0 + 1.0e-9:
                print(
                    "sliceN particle volume fraction plus CBD volume fraction must be at most 1."
                )
                continue
            electrolyte_vf = max(0.0, 1.0 - particle_vf - cbd_vf)
            break
        electrolyte_vf = 1.0 - particle_vf - cbd_vf
        if electrolyte_vf >= -1.0e-9:
            electrolyte_vf = max(0.0, electrolyte_vf)
            break
        print(
            "Particle and CBD volume fractions must sum to at most 1."
        )

    base_axes_um = prompt_triplet(
        "Base particle diameters in um (long, middle, short)",
        (10.0, 6.0, 5.0),
    )
    gradient_slice_count = 5
    gradient_slice1_major_diameter_um = float(max(base_axes_um))
    gradient_sliceN_major_diameter_um = float(max(base_axes_um) * 0.5)
    if electrode_mode == "gradient":
        gradient_slice_count = prompt_int(
            "Number of z-direction slices",
            5,
        )
        gradient_slice1_major_diameter_um = prompt_float(
            "Slice1 base major particle diameter (um)",
            float(max(base_axes_um)),
        )
        gradient_sliceN_major_diameter_um = prompt_float(
            "SliceN base major particle diameter (um)",
            float(max(base_axes_um) * 0.5),
        )

    size_mode = prompt_choice(
        "Particle size distribution mode",
        ("homogeneous", "lognormal", "inverse_histogram"),
        "homogeneous",
    )
    size_distribution = SizeDistributionConfig(mode=size_mode)
    if size_mode == "lognormal":
        size_distribution.parameterization = prompt_choice(
            "Lognormal parameterization", ("factor_space", "log_space"), "factor_space"
        )
        size_distribution.mean = prompt_float(
            "Lognormal mean",
            1.0 if size_distribution.parameterization == "factor_space" else 0.0,
        )
        size_distribution.sigma = prompt_float("Lognormal sigma", 0.25)
    elif size_mode == "inverse_histogram":
        size_distribution.inverse_bin_start_um = prompt_float(
            "Inverse-sampling first diameter-bin start (um)",
            DEFAULT_INVERSE_DIAMETER_BIN_START_UM,
        )
        size_distribution.inverse_bin_width_um = prompt_float(
            "Inverse-sampling diameter bin width (um)",
            DEFAULT_INVERSE_DIAMETER_BIN_WIDTH_UM,
        )
        size_distribution.inverse_probabilities = prompt_float_list(
            "Inverse-sampling diameter-bin probabilities",
            DEFAULT_INVERSE_DIAMETER_PROBABILITIES,
        )
    if size_mode in {"lognormal", "inverse_histogram"}:
        size_distribution.placement_order = prompt_choice(
            "Particle placement order",
            ("interleaved", "random", "descending"),
            "interleaved",
        )
        size_distribution.placement_size_bins = prompt_int(
            "Interleaved placement size-bin count",
            10,
        )

    orientation_mode = prompt_choice(
        "Particle orientation mode", ("homogeneous", "normal"), "homogeneous"
    )
    orientation_distribution = OrientationDistributionConfig(mode=orientation_mode)
    fixed_angles_deg = (0.0, 0.0, 0.0)
    if orientation_mode == "homogeneous":
        fixed_angles_deg = prompt_triplet(
            "Fixed rotation angles in degrees (x, y, z)", (0.0, 0.0, 0.0)
        )
    else:
        orientation_distribution.mean_deg = prompt_triplet(
            "Rotation mean in degrees (x, y, z)", (90.0, 60.0, 30.0)
        )
        orientation_distribution.std_deg = prompt_triplet(
            "Rotation standard deviation in degrees (x, y, z)", (10.0, 10.0, 10.0)
        )

    allowed_overlap = prompt_float("Allowed overlap ratio", 0.2)

    cbd_growth = CBDGrowthConfig()
    if cbd_vf > 0.0:
        cbd_growth.morphology_parameter = prompt_float(
            "CBD morphology parameter omega", 0.5
        )
    export_particle_histogram = (
        prompt_choice(
            "Export particle size histogram",
            ("no", "yes"),
            "no",
        )
        == "yes"
    )
    cfg = SimulationConfig(
        domain_size_um=domain_size_um,
        voxel_size_um=0.5,
        particle_volume_fraction=particle_vf,
        cbd_volume_fraction=cbd_vf,
        electrolyte_volume_fraction=electrolyte_vf,
        electrode_mode=electrode_mode,
        gradient_slice_count=gradient_slice_count,
        gradient_slice1_particle_volume_fraction=gradient_slice1_particle_vf,
        gradient_sliceN_particle_volume_fraction=gradient_sliceN_particle_vf,
        gradient_slice1_major_diameter_um=gradient_slice1_major_diameter_um,
        gradient_sliceN_major_diameter_um=gradient_sliceN_major_diameter_um,
        allowed_overlap=allowed_overlap,
        base_axes_um=base_axes_um,
        axis_definition="diameter",
        size_distribution=size_distribution,
        orientation_distribution=orientation_distribution,
        cbd_growth=cbd_growth,
        fixed_angles_deg=fixed_angles_deg,
        random_seed=default_random_seed(),
        max_placement_attempts=100000,
        perturbation_fraction=0.3,
        particle_pool_factor=1.15,
        export_particle_histogram=export_particle_histogram,
        output_prefix=prompt_text("Output prefix", "electrode_reconstruction"),
    )
    cfg.validate()
    return cfg

def example_config_dict() -> Dict[str, object]:
    return {
        "domain_size_um": [20.0, 20.0, 20.0],
        "voxel_size_um": 0.5,
        "particle_volume_fraction": 0.4,
        "cbd_volume_fraction": 0.1,
        "electrolyte_volume_fraction": 0.5,
        "electrode_mode": "uniform",
        "gradient_slice_count": 5,
        "gradient_slice1_particle_volume_fraction": 0.4,
        "gradient_sliceN_particle_volume_fraction": 0.4,
        "gradient_slice1_major_diameter_um": 10.0,
        "gradient_sliceN_major_diameter_um": 5.0,
        "allowed_overlap": 0.2,
        "base_axes_um": [10.0, 6.0, 5.0],
        "axis_definition": "diameter",
        "size_distribution": {
            "mode": "homogeneous",
            "parameterization": "factor_space",
            "mean": 1.0,
            "sigma": 0.25,
            "rank_trim_enabled": True,
            "rank_trim_lower": 0.15,
            "rank_trim_upper": 0.95,
            "inverse_bin_start_um": DEFAULT_INVERSE_DIAMETER_BIN_START_UM,
            "inverse_bin_width_um": DEFAULT_INVERSE_DIAMETER_BIN_WIDTH_UM,
            "inverse_probabilities": DEFAULT_INVERSE_DIAMETER_PROBABILITIES,
            "placement_order": "interleaved",
            "placement_size_bins": 10,
        },
        "orientation_distribution": {
            "mode": "homogeneous",
            "mean_deg": [0.0, 0.0, 0.0],
            "std_deg": [10.0, 10.0, 10.0],
        },
        "cbd_growth": {
            "morphology_parameter": 0.5,
            "seed_fraction": 0.15,
            "max_growth_steps": 500000,
            "cleanup_min_component_voxels": 20
        },
        "fixed_angles_deg": [0.0, 0.0, 0.0],
        "random_seed": None,
        "max_placement_attempts": 100000,
        "perturbation_fraction": 0.3,
        "particle_pool_factor": 1.15,
        "export_stl": True,
        "export_step": False,
        "export_ascii_stl": True,
        "export_voxel_npy": True,
        "export_particle_excel": True,
        "export_particle_histogram": False,
        "export_structure_txt": True,
        "export_summary_json": False,
        "output_prefix": "electrode_case_01",
    }

def write_example_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(example_config_dict(), handle, indent=2)
        handle.write("\n")

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a voxelized porous electrode microstructure with optional CBD growth."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a JSON config file. If omitted, interactive input is used.",
    )
    parser.add_argument(
        "--write-example-config",
        type=Path,
        help="Write an example JSON config file and exit.",
    )
    parser.add_argument(
        "--random-batch",
        action="store_true",
        help="Generate a random batch of structures internally instead of using a single config.",
    )
    parser.add_argument(
        "--batch-count",
        type=int,
        default=2000,
        help="Number of structures to generate in random batch mode. Default: 2000",
    )
    parser.add_argument(
        "--batch-seed",
        type=int,
        help="Optional master random seed for random batch mode.",
    )
    parser.add_argument(
        "--batch-timeout-seconds",
        type=float,
        default=30.0,
        help=(
            "Per-structure timeout in random batch mode. "
            "If a structure exceeds this time, it is skipped and regenerated. "
            "Default: 30"
        ),
    )
    parser.add_argument(
        "--batch-output-prefix",
        type=str,
        default="electrode_random",
        help="Output prefix base used in random batch mode. Default: electrode_random",
    )
    parser.add_argument(
        "--batch-output-dir",
        type=Path,
        help=(
            "Output root directory used in random batch mode. "
            f"Default: {DEFAULT_OUTPUT_BASE_DIR}"
        ),
    )
    return parser.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.write_example_config:
        write_example_config(args.write_example_config)
        print(f"Example config written to: {args.write_example_config}")
        return 0

    base_dir = DEFAULT_SINGLE_OUTPUT_BASE_DIR
    if args.config and args.random_batch:
        raise ValueError("--config and --random-batch cannot be used together.")

    if args.random_batch:
        batch_output_dir = (
            args.batch_output_dir
            if args.batch_output_dir is not None
            else DEFAULT_OUTPUT_BASE_DIR
        )
        batch_config = RandomBatchConfig(
            structure_count=args.batch_count,
            grid_shape_vox=(64, 64, 64),
            voxel_size_um=0.5,
            max_major_axis_um=10.0,
            per_structure_timeout_s=args.batch_timeout_seconds,
            output_prefix_base=args.batch_output_prefix,
            batch_seed=args.batch_seed,
        )
        run_random_batch_reconstruction(batch_config, base_dir=batch_output_dir)
    elif args.config:
        with args.config.open("r", encoding="utf-8") as handle:
            config_data = json.load(handle)
        config = SimulationConfig.from_dict(config_data)
        run_reconstruction(config, base_dir=base_dir)
    else:
        run_mode = prompt_choice(
            "Run mode",
            ("single", "random_batch"),
            "single",
        )
        electrode_mode = prompt_choice(
            "Electrode structure mode",
            ("uniform", "gradient"),
            "uniform",
        )
        if run_mode == "random_batch":
            batch_config, batch_output_dir = interactive_random_batch_config(
                DEFAULT_OUTPUT_BASE_DIR,
                electrode_mode,
            )
            run_random_batch_reconstruction(batch_config, base_dir=batch_output_dir)
        else:
            config = interactive_config(electrode_mode)
            run_reconstruction(config, base_dir=base_dir)
    return 0

