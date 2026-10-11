from __future__ import annotations

import math
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict

import numpy as np

Float3 = Tuple[float, float, float]

Int3 = Tuple[int, int, int]

WrapAxes = Tuple[bool, bool, bool]

CenterBounds3 = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]

EPS = 1.0e-12

PHASE_ELECTROLYTE = 0

PHASE_ACTIVE = 1

PHASE_CBD = 2

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SINGLE_OUTPUT_BASE_DIR = DEFAULT_PROJECT_ROOT / "toCOMSOL"

DEFAULT_OUTPUT_BASE_DIR = DEFAULT_PROJECT_ROOT / "Structure_Data"

RANDOM_BATCH_MAX_MAJOR_AXIS_VOX = 8.0

RANDOM_BATCH_CBD_VOLUME_FRACTION_RANGE = (0.02, 0.10)

RANDOM_BATCH_ELECTROLYTE_VOLUME_FRACTION_RANGE = (0.20, 0.48)

RANDOM_BATCH_ELECTROLYTE_BETA_SHAPE = (1.10, 2.40)

DEFAULT_PARTICLE_HISTOGRAM_BIN_WIDTH_UM = 2.0

DEFAULT_INVERSE_DIAMETER_BIN_WIDTH_UM = 2.0

DEFAULT_INVERSE_DIAMETER_BIN_START_UM = 0.0

DEFAULT_INVERSE_DIAMETER_PROBABILITIES = [
    0.082,
    0.034,
    0.197,
    0.197,
    0.082,
    0.064,
    0.094,
    0.100,
    0.039,
    0.034,
    0.039,
    0.002,
    0.009,
    0.003,
    0.022
]

STEP_FACE_SPECS = [
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
    ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
]

@dataclass
class SizeDistributionConfig:
    mode: str = "homogeneous"
    parameterization: str = "factor_space"
    mean: float = 1.0
    sigma: float = 0.25
    rank_trim_enabled: bool = True
    rank_trim_lower: float = 0.15
    rank_trim_upper: float = 0.95
    inverse_bin_start_um: float = DEFAULT_INVERSE_DIAMETER_BIN_START_UM
    inverse_bin_width_um: float = DEFAULT_INVERSE_DIAMETER_BIN_WIDTH_UM
    inverse_probabilities: List[float] = field(
        default_factory=lambda: list(DEFAULT_INVERSE_DIAMETER_PROBABILITIES)
    )
    placement_order: str = "interleaved"
    placement_size_bins: int = 10

@dataclass
class OrientationDistributionConfig:
    mode: str = "homogeneous"
    mean_deg: Float3 = (0.0, 0.0, 0.0)
    std_deg: Float3 = (0.0, 0.0, 0.0)

@dataclass
class CBDGrowthConfig:
    morphology_parameter: float = 0.5
    seed_fraction: float = 0.15
    max_growth_steps: int = 500000
    cleanup_min_component_voxels: int = 20

class CBDGrowthStats(TypedDict):
    target_cbd_voxels: int
    seeded_cbd_voxels: int
    grown_cbd_voxels: int
    growth_iterations: int
    candidate_voxels_final: int
    growth_terminated_early: bool
    growth_stop_reason: str
    cleanup_removed_components: int
    cleanup_removed_voxels: int
    cleanup_regrown_voxels: int
    cleanup_eroded_voxels: int
    cleanup_final_voxels: int

class GradientSliceSpec(TypedDict):
    slice_index: int
    z_start: int
    z_end: int
    thickness_vox: int
    particle_volume_fraction: float
    major_diameter_um: float
    base_axes_um: Float3
    target_particle_voxels: int
    target_particle_volume_um3: float
    center_bounds: CenterBounds3
    min_z_inclusive: int

@dataclass
class SimulationConfig:
    domain_size_um: Float3 = (20.0, 20.0, 20.0)
    voxel_size_um: float = 0.5
    particle_volume_fraction: float = 0.4
    cbd_volume_fraction: float = 0.1
    electrolyte_volume_fraction: float = 0.5
    electrode_mode: str = "uniform"
    gradient_slice_count: int = 5
    gradient_slice1_particle_volume_fraction: float = 0.4
    gradient_sliceN_particle_volume_fraction: float = 0.4
    gradient_slice1_major_diameter_um: float = 10.0
    gradient_sliceN_major_diameter_um: float = 5.0
    allowed_overlap: float = 0.2
    base_axes_um: Float3 = (10.0, 6.0, 5.0)
    axis_definition: str = "diameter"
    size_distribution: SizeDistributionConfig = field(
        default_factory=SizeDistributionConfig
    )
    orientation_distribution: OrientationDistributionConfig = field(
        default_factory=OrientationDistributionConfig
    )
    cbd_growth: CBDGrowthConfig = field(default_factory=CBDGrowthConfig)
    fixed_angles_deg: Float3 = (0.0, 0.0, 0.0)
    random_seed: Optional[int] = None
    max_placement_attempts: int = 100000
    perturbation_fraction: float = 0.3
    particle_pool_factor: float = 1.15
    export_stl: bool = True
    export_step: bool = False
    export_ascii_stl: bool = True
    export_voxel_npy: bool = True
    export_particle_excel: bool = True
    export_particle_histogram: bool = False
    export_structure_txt: bool = True
    export_summary_json: bool = False
    output_prefix: str = "electrode_reconstruction"

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SimulationConfig":
        sd_raw = data.get("size_distribution", {})
        sd_data = (
            {str(key): value for key, value in sd_raw.items()}
            if isinstance(sd_raw, dict)
            else {}
        )
        size_distribution = SizeDistributionConfig(**sd_data)
        od_raw = data.get("orientation_distribution", {})
        od_data = (
            {str(key): value for key, value in od_raw.items()}
            if isinstance(od_raw, dict)
            else {}
        )
        orientation_distribution = OrientationDistributionConfig(**od_data)
        cbd_raw = data.get("cbd_growth", {})
        cbd_data = (
            {str(key): value for key, value in cbd_raw.items()}
            if isinstance(cbd_raw, dict)
            else {}
        )
        cbd_growth = CBDGrowthConfig(**cbd_data)
        random_seed_raw = data.get("random_seed", cls.random_seed)
        particle_volume_fraction = to_float(
            data.get("particle_volume_fraction", cls.particle_volume_fraction),
            "particle_volume_fraction",
        )
        cbd_volume_fraction = to_float(
            data.get("cbd_volume_fraction", cls.cbd_volume_fraction),
            "cbd_volume_fraction",
        )
        electrolyte_default = (
            cls.electrolyte_volume_fraction
            if "electrolyte_volume_fraction" in data
            else 1.0 - particle_volume_fraction - cbd_volume_fraction
        )
        cfg = cls(
            domain_size_um=as_float3(
                data.get("domain_size_um", cls.domain_size_um), "domain_size_um"
            ),
            voxel_size_um=to_float(
                data.get("voxel_size_um", cls.voxel_size_um), "voxel_size_um"
            ),
            particle_volume_fraction=particle_volume_fraction,
            cbd_volume_fraction=cbd_volume_fraction,
            electrolyte_volume_fraction=to_float(
                data.get("electrolyte_volume_fraction", electrolyte_default),
                "electrolyte_volume_fraction",
            ),
            electrode_mode=str(data.get("electrode_mode", cls.electrode_mode)),
            gradient_slice_count=to_int(
                data.get("gradient_slice_count", cls.gradient_slice_count),
                "gradient_slice_count",
            ),
            gradient_slice1_particle_volume_fraction=to_float(
                data.get(
                    "gradient_slice1_particle_volume_fraction",
                    particle_volume_fraction,
                ),
                "gradient_slice1_particle_volume_fraction",
            ),
            gradient_sliceN_particle_volume_fraction=to_float(
                data.get(
                    "gradient_sliceN_particle_volume_fraction",
                    particle_volume_fraction,
                ),
                "gradient_sliceN_particle_volume_fraction",
            ),
            gradient_slice1_major_diameter_um=to_float(
                data.get(
                    "gradient_slice1_major_diameter_um",
                    cls.gradient_slice1_major_diameter_um,
                ),
                "gradient_slice1_major_diameter_um",
            ),
            gradient_sliceN_major_diameter_um=to_float(
                data.get(
                    "gradient_sliceN_major_diameter_um",
                    cls.gradient_sliceN_major_diameter_um,
                ),
                "gradient_sliceN_major_diameter_um",
            ),
            allowed_overlap=to_float(
                data.get("allowed_overlap", cls.allowed_overlap), "allowed_overlap"
            ),
            base_axes_um=as_float3(
                data.get("base_axes_um", cls.base_axes_um), "base_axes_um"
            ),
            axis_definition=str(data.get("axis_definition", cls.axis_definition)),
            size_distribution=size_distribution,
            orientation_distribution=orientation_distribution,
            cbd_growth=cbd_growth,
            fixed_angles_deg=as_float3(
                data.get("fixed_angles_deg", cls.fixed_angles_deg),
                "fixed_angles_deg",
            ),
            random_seed=(
                None
                if random_seed_raw is None
                else to_int(random_seed_raw, "random_seed")
            ),
            max_placement_attempts=to_int(
                data.get(
                    "max_placement_attempts", cls.max_placement_attempts
                ),
                "max_placement_attempts",
            ),
            perturbation_fraction=to_float(
                data.get("perturbation_fraction", cls.perturbation_fraction),
                "perturbation_fraction",
            ),
            particle_pool_factor=to_float(
                data.get("particle_pool_factor", cls.particle_pool_factor),
                "particle_pool_factor",
            ),
            export_stl=to_bool(data.get("export_stl", cls.export_stl), "export_stl"),
            export_step=to_bool(data.get("export_step", cls.export_step), "export_step"),
            export_ascii_stl=to_bool(
                data.get("export_ascii_stl", cls.export_ascii_stl),
                "export_ascii_stl",
            ),
            export_voxel_npy=to_bool(
                data.get("export_voxel_npy", cls.export_voxel_npy),
                "export_voxel_npy",
            ),
            export_particle_excel=to_bool(
                data.get(
                    "export_particle_excel",
                    data.get("export_particle_csv", cls.export_particle_excel),
                ),
                "export_particle_excel",
            ),
            export_particle_histogram=to_bool(
                data.get(
                    "export_particle_histogram",
                    cls.export_particle_histogram,
                ),
                "export_particle_histogram",
            ),
            export_structure_txt=to_bool(
                data.get("export_structure_txt", cls.export_structure_txt),
                "export_structure_txt",
            ),
            export_summary_json=to_bool(
                data.get("export_summary_json", cls.export_summary_json),
                "export_summary_json",
            ),
            output_prefix=str(data.get("output_prefix", cls.output_prefix)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        self.domain_size_um = as_float3(self.domain_size_um, "domain_size_um")
        sorted_axes = sorted(
            (float(value) for value in self.base_axes_um), reverse=True
        )
        self.base_axes_um = (
            sorted_axes[0],
            sorted_axes[1],
            sorted_axes[2],
        )
        self.fixed_angles_deg = normalize_angles_deg(self.fixed_angles_deg)
        self.orientation_distribution.mean_deg = normalize_angles_deg(
            self.orientation_distribution.mean_deg
        )
        self.orientation_distribution.std_deg = as_float3(
            self.orientation_distribution.std_deg, "orientation std_deg"
        )
        if self.random_seed is None:
            self.random_seed = default_random_seed()

        if any(value <= 0.0 for value in self.domain_size_um):
            raise ValueError("All domain dimensions must be positive.")
        if self.voxel_size_um <= 0.0:
            raise ValueError("voxel_size_um must be positive.")
        if any(value <= 0.0 for value in self.base_axes_um):
            raise ValueError("All ellipsoid axes must be positive.")
        if not (0.0 <= self.cbd_volume_fraction <= 1.0):
            raise ValueError("cbd_volume_fraction must be within [0, 1].")
        self.electrode_mode = str(self.electrode_mode).lower()
        if self.electrode_mode not in {"uniform", "gradient"}:
            raise ValueError("electrode_mode must be either 'uniform' or 'gradient'.")
        if self.gradient_slice_count <= 0:
            raise ValueError("gradient_slice_count must be positive.")
        depth_vox = max(1, int(round(self.domain_size_um[2] / self.voxel_size_um)))
        if self.electrode_mode == "gradient" and self.gradient_slice_count > depth_vox:
            raise ValueError(
                "gradient_slice_count must not exceed the number of voxels along z."
            )
        if not (
            0.0 <= self.gradient_slice1_particle_volume_fraction <= 1.0
        ):
            raise ValueError(
                "gradient_slice1_particle_volume_fraction must be within [0, 1]."
            )
        if not (
            0.0 <= self.gradient_sliceN_particle_volume_fraction <= 1.0
        ):
            raise ValueError(
                "gradient_sliceN_particle_volume_fraction must be within [0, 1]."
            )
        if self.gradient_slice1_major_diameter_um <= 0.0:
            raise ValueError("gradient_slice1_major_diameter_um must be positive.")
        if self.gradient_sliceN_major_diameter_um <= 0.0:
            raise ValueError("gradient_sliceN_major_diameter_um must be positive.")
        if self.electrode_mode == "gradient":
            if (
                self.gradient_slice1_particle_volume_fraction
                + self.cbd_volume_fraction
                > 1.0 + 1.0e-9
            ):
                raise ValueError(
                    "gradient slice1 particle volume fraction plus cbd_volume_fraction must be <= 1."
                )
            if (
                self.gradient_sliceN_particle_volume_fraction
                + self.cbd_volume_fraction
                > 1.0 + 1.0e-9
            ):
                raise ValueError(
                    "gradient sliceN particle volume fraction plus cbd_volume_fraction must be <= 1."
                )
            slice_particle_values: List[float] = []
            if self.gradient_slice_count == 1:
                slice_particle_values = [
                    float(self.gradient_slice1_particle_volume_fraction)
                ]
            else:
                step = (
                    float(self.gradient_sliceN_particle_volume_fraction)
                    - float(self.gradient_slice1_particle_volume_fraction)
                ) / float(self.gradient_slice_count - 1)
                slice_particle_values = [
                    float(self.gradient_slice1_particle_volume_fraction)
                    + float(index) * step
                    for index in range(self.gradient_slice_count)
                ]
            base_thickness = depth_vox // self.gradient_slice_count
            remainder = depth_vox % self.gradient_slice_count
            slice_weights = [
                base_thickness + (1 if index < remainder else 0)
                for index in range(self.gradient_slice_count)
            ]
            weighted_particle_sum = sum(
                value * float(weight)
                for value, weight in zip(slice_particle_values, slice_weights)
            )
            self.particle_volume_fraction = weighted_particle_sum / float(depth_vox)
            self.electrolyte_volume_fraction = (
                1.0 - self.particle_volume_fraction - self.cbd_volume_fraction
            )
        if not (0.0 <= self.particle_volume_fraction <= 1.0):
            raise ValueError("particle_volume_fraction must be within [0, 1].")
        if not (0.0 <= self.electrolyte_volume_fraction <= 1.0):
            raise ValueError("electrolyte_volume_fraction must be within [0, 1].")
        if not math.isclose(
            self.particle_volume_fraction
            + self.cbd_volume_fraction
            + self.electrolyte_volume_fraction,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "particle_volume_fraction + cbd_volume_fraction + electrolyte_volume_fraction must equal 1."
            )
        if not (0.0 <= self.allowed_overlap < 1.0):
            raise ValueError("allowed_overlap must be within [0, 1).")
        if self.axis_definition not in {"semi_axis", "diameter"}:
            raise ValueError(
                "axis_definition must be either 'semi_axis' or 'diameter'."
            )
        if self.max_placement_attempts <= 0:
            raise ValueError("max_placement_attempts must be positive.")
        if self.perturbation_fraction <= 0.0:
            raise ValueError("perturbation_fraction must be positive.")
        if self.particle_pool_factor < 1.0:
            raise ValueError("particle_pool_factor must be >= 1.0.")

        size_mode = self.size_distribution.mode.lower()
        if size_mode == "inverse":
            size_mode = "inverse_histogram"
        if size_mode not in {"homogeneous", "lognormal", "inverse_histogram"}:
            raise ValueError(
                "size_distribution.mode must be 'homogeneous', 'lognormal', or 'inverse_histogram'."
            )
        self.size_distribution.mode = size_mode

        parameterization = self.size_distribution.parameterization.lower()
        if parameterization not in {"factor_space", "log_space"}:
            raise ValueError(
                "size_distribution.parameterization must be 'factor_space' or 'log_space'."
            )
        self.size_distribution.parameterization = parameterization
        if size_mode == "lognormal":
            if parameterization == "factor_space":
                if self.size_distribution.mean <= 0.0:
                    raise ValueError(
                        "For factor_space lognormal input, mean must be positive."
                    )
                if self.size_distribution.sigma < 0.0:
                    raise ValueError(
                        "For factor_space lognormal input, sigma must be non-negative."
                    )
            else:
                if self.size_distribution.sigma < 0.0:
                    raise ValueError(
                        "For log_space lognormal input, sigma must be non-negative."
                    )
            lower = float(self.size_distribution.rank_trim_lower)
            upper = float(self.size_distribution.rank_trim_upper)
            if not (0.0 <= lower < upper <= 1.0):
                raise ValueError(
                    "size_distribution rank trim bounds must satisfy 0 <= lower < upper <= 1."
                )
        if size_mode == "inverse_histogram":
            if self.size_distribution.inverse_bin_width_um <= 0.0:
                raise ValueError(
                    "size_distribution.inverse_bin_width_um must be positive."
                )
            if self.size_distribution.inverse_bin_start_um < 0.0:
                raise ValueError(
                    "size_distribution.inverse_bin_start_um must be non-negative."
                )
            inverse_probabilities = [
                float(value)
                for value in self.size_distribution.inverse_probabilities
            ]
            if not inverse_probabilities:
                raise ValueError(
                    "size_distribution.inverse_probabilities must not be empty."
                )
            if any(
                (not math.isfinite(value)) or value < 0.0
                for value in inverse_probabilities
            ):
                raise ValueError(
                    "size_distribution.inverse_probabilities must be finite non-negative numbers."
                )
            if sum(inverse_probabilities) <= 0.0:
                raise ValueError(
                    "size_distribution.inverse_probabilities must contain at least one positive value."
                )
            self.size_distribution.inverse_probabilities = inverse_probabilities
        placement_order = self.size_distribution.placement_order.lower()
        if placement_order not in {"descending", "random", "interleaved"}:
            raise ValueError(
                "size_distribution.placement_order must be 'descending', 'random', or 'interleaved'."
            )
        self.size_distribution.placement_order = placement_order
        if self.size_distribution.placement_size_bins < 1:
            raise ValueError("size_distribution.placement_size_bins must be >= 1.")
        if self.cbd_growth.cleanup_min_component_voxels < 1:
            raise ValueError("cbd_growth.cleanup_min_component_voxels must be >= 1.")

        orientation_mode = self.orientation_distribution.mode.lower()
        if orientation_mode not in {"homogeneous", "normal"}:
            raise ValueError(
                "orientation_distribution.mode must be 'homogeneous' or 'normal'."
            )
        self.orientation_distribution.mode = orientation_mode
        if any(value < 0.0 for value in self.orientation_distribution.std_deg):
            raise ValueError("Orientation standard deviations must be non-negative.")

        if not (0.0 <= self.cbd_growth.morphology_parameter <= 1.0):
            raise ValueError(
                "cbd_growth.morphology_parameter must be within [0, 1]."
            )
        if self.cbd_growth.seed_fraction < 0.0:
            raise ValueError("cbd_growth.seed_fraction must be non-negative.")
        if self.cbd_growth.max_growth_steps <= 0:
            raise ValueError("cbd_growth.max_growth_steps must be positive.")

@dataclass
class RandomBatchConfig:
    structure_count: int = 2000
    grid_shape_vox: Int3 = (64, 64, 64)
    voxel_size_um: float = 0.5
    max_major_axis_um: float = 10.0
    per_structure_timeout_s: float = 30.0
    electrode_mode: str = "uniform"
    gradient_slice_count: int = 5
    gradient_slice1_major_diameter_um: float = 4.0
    gradient_sliceN_major_diameter_um: float = 2.0
    output_prefix_base: str = "electrode_random"
    batch_seed: Optional[int] = None

    def validate(self) -> None:
        self.grid_shape_vox = (
            int(self.grid_shape_vox[0]),
            int(self.grid_shape_vox[1]),
            int(self.grid_shape_vox[2]),
        )
        if self.structure_count <= 0:
            raise ValueError("structure_count must be positive.")
        if any(value <= 0 for value in self.grid_shape_vox):
            raise ValueError("grid_shape_vox must contain only positive integers.")
        if self.voxel_size_um <= 0.0:
            raise ValueError("voxel_size_um must be positive.")
        if self.max_major_axis_um <= 0.0:
            raise ValueError("max_major_axis_um must be positive.")
        if not math.isfinite(self.per_structure_timeout_s):
            raise ValueError("per_structure_timeout_s must be finite.")
        if self.per_structure_timeout_s <= 0.0:
            raise ValueError("per_structure_timeout_s must be positive.")
        self.electrode_mode = str(self.electrode_mode).lower()
        if self.electrode_mode not in {"uniform", "gradient", "mixed"}:
            raise ValueError(
                "electrode_mode must be 'uniform', 'gradient', or 'mixed'."
            )
        if self.gradient_slice_count <= 0:
            raise ValueError("gradient_slice_count must be positive.")
        if self.gradient_slice_count > int(self.grid_shape_vox[2]):
            raise ValueError(
                "gradient_slice_count must not exceed the number of voxels along z."
            )
        if self.gradient_slice1_major_diameter_um <= 0.0:
            raise ValueError("gradient_slice1_major_diameter_um must be positive.")
        if self.gradient_sliceN_major_diameter_um <= 0.0:
            raise ValueError("gradient_sliceN_major_diameter_um must be positive.")
        effective_max_major_axis_um = min(
            float(self.max_major_axis_um),
            float(self.voxel_size_um) * float(RANDOM_BATCH_MAX_MAJOR_AXIS_VOX),
        )
        if self.electrode_mode in {"gradient", "mixed"}:
            if self.gradient_slice1_major_diameter_um > effective_max_major_axis_um + 1.0e-9:
                raise ValueError(
                    "gradient_slice1_major_diameter_um exceeds the allowed random-batch base major diameter limit."
                )
            if self.gradient_sliceN_major_diameter_um > effective_max_major_axis_um + 1.0e-9:
                raise ValueError(
                    "gradient_sliceN_major_diameter_um exceeds the allowed random-batch base major diameter limit."
                )
        if self.batch_seed is None:
            self.batch_seed = default_random_seed()
        self.output_prefix_base = sanitize_output_stem(self.output_prefix_base)

@dataclass
class ParticleRecord:
    particle_id: int
    scale_factor: float
    semi_axes_um: Float3
    semi_axes_vox: Float3
    angles_deg: Float3
    analytic_volume_um3: float
    template_voxels: int
    template: np.ndarray = field(repr=False)
    placed: bool = False
    center_vox: Optional[Int3] = None
    attempts: int = 0
    overlap_ratio: float = 0.0

def as_float3(values: object, name: str) -> Float3:
    if isinstance(values, (tuple, list)) and len(values) == 3:
        return (float(values[0]), float(values[1]), float(values[2]))
    raise ValueError(f"{name} must contain exactly 3 values.")

def to_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, got boolean.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{name} must not be empty.")
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be numeric.") from exc
    raise ValueError(f"{name} must be numeric.")

def to_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got boolean.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{name} must be an integer.")
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{name} must not be empty.")
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer.") from exc
    raise ValueError(f"{name} must be an integer.")

def to_bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{name} must be boolean (true/false).")

def normalize_angles_deg(values: Sequence[float]) -> Float3:
    if len(values) != 3:
        raise ValueError("Angle input must contain exactly 3 values.")
    normalized = []
    for angle in values:
        angle = float(angle)
        normalized.append(((angle + 180.0) % 360.0) - 180.0)
    return (normalized[0], normalized[1], normalized[2])

def sanitize_output_stem(output_prefix: str) -> str:
    stem = Path(output_prefix).name.strip()
    if stem in {"", ".", ".."}:
        return "electrode_reconstruction"
    return stem

def create_unique_run_output_dir(base_dir: Path) -> Path:
    date_prefix = datetime.now().strftime("%Y%m%d")
    index = 1
    while True:
        candidate = base_dir / f"{date_prefix}_{index:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        index += 1

def default_random_seed() -> int:
    date_value = int(datetime.now().strftime("%Y%m%d"))
    hex_text = format(date_value, "X")
    digits_only = "".join(character for character in hex_text if character.isdigit())
    return int(digits_only) if digits_only else date_value

def format_float3(values: Sequence[float]) -> str:
    return "(" + ", ".join(f"{float(value):.3f}" for value in values) + ")"
