"""Reproducible synthetic examples; does not modify existing structure data."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from electrode_structure_factors import (
    compute_chord_length_distribution,
    compute_spherical_contact_distribution,
    compute_two_point_correlation,
)


def main():
    n = 32
    x, y, z = np.indices((n, n, n))
    examples = {
        "Periodic slabs": np.where(x % 8 < 4, 0, 1),
        "Periodic pore spheres": np.where(sum((v % 16 - 8) ** 2 for v in (x, y, z)) < 6 ** 2, 0, 1),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for name, labels in examples.items():
        chord = compute_chord_length_distribution(labels, .5)
        contact = compute_spherical_contact_distribution(labels, .5)
        two_point = compute_two_point_correlation(labels, .5)
        for ax, distribution in zip(axes[:2], (chord, contact)):
            ax.step([0, *distribution.values_um], [0, *distribution.cdf], where="post", label=name)
            ax.set_ylim(0, 1.05)
        axes[2].plot(two_point.radii_um, two_point.pore_pore_probability, label=name)
    for ax, title, xlabel, ylabel in zip(axes,
            ["Pore chord lengths (x)", "Solid-to-pore contact distances", "Pore-pore correlation"],
            ["Chord length (um)", "Distance (um)", "Radial shell upper edge (um)"],
            ["Empirical CDF", "Empirical CDF", "Joint probability"]):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Synthetic periodic 32 x 32 x 32 structures | voxel size 0.5 um")
    destination = Path(__file__).with_name("distribution_descriptors_demo.png")
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
