from __future__ import annotations

import random
from typing import Tuple

from .shared import (
    CBDGrowthConfig,
    Float3,
    OrientationDistributionConfig,
    RANDOM_BATCH_CBD_VOLUME_FRACTION_RANGE,
    RANDOM_BATCH_ELECTROLYTE_BETA_SHAPE,
    RANDOM_BATCH_ELECTROLYTE_VOLUME_FRACTION_RANGE,
    RANDOM_BATCH_MAX_MAJOR_AXIS_VOX,
    RandomBatchConfig,
    SimulationConfig,
    SizeDistributionConfig,
)

def random_batch_domain_size_um(batch_config: RandomBatchConfig) -> Float3:
    return (
        float(batch_config.grid_shape_vox[0] * batch_config.voxel_size_um),
        float(batch_config.grid_shape_vox[1] * batch_config.voxel_size_um),
        float(batch_config.grid_shape_vox[2] * batch_config.voxel_size_um),
    )

def random_batch_max_major_axis_limit_um(batch_config: RandomBatchConfig) -> float:
    voxel_limited_um = float(batch_config.voxel_size_um) * float(
        RANDOM_BATCH_MAX_MAJOR_AXIS_VOX
    )
    return float(min(float(batch_config.max_major_axis_um), voxel_limited_um))

def sample_random_base_axes_um(
    rng: random.Random, max_major_axis_um: float
) -> Float3:
    min_major_axis_um = max_major_axis_um * 0.40
    long_axis_um = rng.uniform(min_major_axis_um, max_major_axis_um)
    middle_axis_um = long_axis_um * rng.uniform(0.45, 1.0)
    short_axis_um = middle_axis_um * rng.uniform(0.45, 1.0)
    sorted_axes = sorted(
        (long_axis_um, middle_axis_um, short_axis_um), reverse=True
    )
    return (
        float(sorted_axes[0]),
        float(sorted_axes[1]),
        float(sorted_axes[2]),
    )

def sample_low_porosity_electrolyte_volume_fraction(rng: random.Random) -> float:
    low, high = RANDOM_BATCH_ELECTROLYTE_VOLUME_FRACTION_RANGE
    alpha, beta = RANDOM_BATCH_ELECTROLYTE_BETA_SHAPE
    biased_fraction = rng.betavariate(alpha, beta)
    return float(low + (high - low) * biased_fraction)

def sample_random_volume_fractions(rng: random.Random) -> Tuple[float, float, float]:
    electrolyte_vf = sample_low_porosity_electrolyte_volume_fraction(rng)
    cbd_vf = rng.uniform(*RANDOM_BATCH_CBD_VOLUME_FRACTION_RANGE)
    particle_vf = 1.0 - electrolyte_vf - cbd_vf
    if particle_vf <= 0.0:
        raise ValueError("Randomly sampled volume fractions produced non-positive particle phase.")
    return float(particle_vf), float(cbd_vf), float(electrolyte_vf)

def generate_random_batch_simulation_config(
    batch_config: RandomBatchConfig,
    rng: random.Random,
    structure_index: int,
) -> SimulationConfig:
    sampled_electrode_mode = str(batch_config.electrode_mode).lower()
    if sampled_electrode_mode == "mixed":
        sampled_electrode_mode = "gradient" if rng.random() < 0.5 else "uniform"

    particle_vf, cbd_vf, electrolyte_vf = sample_random_volume_fractions(rng)
    effective_max_major_axis_um = random_batch_max_major_axis_limit_um(batch_config)
    cfg = SimulationConfig(
        domain_size_um=random_batch_domain_size_um(batch_config),
        voxel_size_um=batch_config.voxel_size_um,
        particle_volume_fraction=particle_vf,
        cbd_volume_fraction=cbd_vf,
        electrolyte_volume_fraction=electrolyte_vf,
        electrode_mode=sampled_electrode_mode,
        gradient_slice_count=batch_config.gradient_slice_count,
        gradient_slice1_major_diameter_um=batch_config.gradient_slice1_major_diameter_um,
        gradient_sliceN_major_diameter_um=batch_config.gradient_sliceN_major_diameter_um,
        allowed_overlap=rng.uniform(0.05, 0.38),
        base_axes_um=sample_random_base_axes_um(rng, effective_max_major_axis_um),
        axis_definition="diameter",
        size_distribution=SizeDistributionConfig(
            mode="lognormal",
            parameterization="factor_space",
            mean=rng.uniform(0.75, 1.25),
            sigma=rng.uniform(0.05, 0.45),
        ),
        orientation_distribution=OrientationDistributionConfig(
            mode="normal",
            mean_deg=(
                rng.uniform(-180.0, 180.0),
                rng.uniform(-180.0, 180.0),
                rng.uniform(-180.0, 180.0),
            ),
            std_deg=(
                rng.uniform(5.0, 90.0),
                rng.uniform(5.0, 90.0),
                rng.uniform(5.0, 90.0),
            ),
        ),
        cbd_growth=CBDGrowthConfig(
            morphology_parameter=rng.uniform(0.05, 0.95),
            seed_fraction=0.15,
            max_growth_steps=500000,
        ),
        fixed_angles_deg=(0.0, 0.0, 0.0),
        random_seed=rng.randint(1, 2**31 - 1),
        max_placement_attempts=rng.randint(120000, 260000),
        perturbation_fraction=rng.uniform(0.05, 0.65),
        particle_pool_factor=rng.uniform(1.25, 1.8),
        export_stl=False,
        export_step=False,
        export_ascii_stl=False,
        export_voxel_npy=True,
        export_particle_excel=False,
        export_particle_histogram=False,
        export_structure_txt=True,
        export_summary_json=False,
        output_prefix=f"{batch_config.output_prefix_base}_{structure_index:04d}",
    )
    cfg.validate()
    return cfg
