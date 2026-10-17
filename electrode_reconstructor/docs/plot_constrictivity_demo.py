"""Regenerate the synthetic CPSD/MIP illustration: python -B this_file.py."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from electrode_structure_factors import compute_constrictivity


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for col, neck in enumerate((False, True)):
        labels = np.ones((25, 17, 17), dtype=np.int8)
        labels[:, 3:14, 3:14] = 0
        if neck:
            labels[3:6] = 1
            labels[3:6, 7:10, 7:10] = 0
        result = compute_constrictivity(labels, .5)
        axes[0, col].imshow((labels[:, :, 8] == 0).T, origin='lower', cmap='Blues',
                            extent=(0, 12.5, 0, 8.5), vmin=0, vmax=1)
        axes[0, col].set(title='Narrow throat' if neck else 'Uniform channel',
                         xlabel='x (um); intrusion from left', ylabel='y (um)')
        ax = axes[1, col]
        porosity = result.cpsd[0]
        ax.plot(result.radii_um, np.array(result.cpsd)/porosity, 'o-', ms=3,
                label='CPSD / porosity', color='#2463a6')
        ax.plot(result.radii_um, np.array(result.mip)/porosity, 's--', ms=3,
                label='MIP / porosity', color='#d65f24')
        ax.axhline(.5, color='#666666', ls=':', label='50% of all pore volume')
        ax.axvline(result.r_min_um, color='#d65f24', alpha=.3)
        ax.axvline(result.r_max_um, color='#2463a6', alpha=.3)
        ax.set(xlabel='Probe radius (um); points = EDT critical radii',
               ylabel='Fraction of pore volume', ylim=(-.03, 1.08),
               title=f'r_min={result.r_min_um:g}, r_max={result.r_max_um:g} um; beta={result.constrictivity_beta:.4f}')
        ax.legend(fontsize=8, loc='lower left')
    fig.suptitle('Constrictivity v2: 26 neighbors, critical radii, x-low inlet', fontsize=14)
    fig.savefig(Path(__file__).with_name('constrictivity_demo.png'), dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
