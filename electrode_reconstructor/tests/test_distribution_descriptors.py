"""Independent numerical checks and end-to-end export coverage."""
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import electrode_structure_factors as f
from test_constrictivity import output_directory


def write_structure(path, labels):
    path.write_text("\n\n".join("\n".join(" ".join(map(str, row))
                    for row in labels[:, :, z]) for z in range(labels.shape[2])), encoding="utf-8")


class DescriptorTests(unittest.TestCase):
    def test_chords_merge_seam_and_equal_chord_weight(self):
        labels = np.array([0, 0, 1, 0, 1, 1, 0], dtype=np.int8)[:, None, None]
        r = f.compute_chord_length_distribution(labels, .5)
        self.assertEqual(r.values_um, [.5, 1.5])
        self.assertEqual(r.counts, [1, 1])
        self.assertEqual(r.probability_mass, [.5, .5])
        self.assertEqual(r.mean_um, 1)
        self.assertEqual(r.std_um, .5)
        self.assertEqual(r.quantiles_um[9], .5)  # inverse ECDF median, not interpolation
        self.assertEqual(r.quantiles_um[10], 1.5)
        self.assertEqual(r.cdf[-1], 1)
        self.assertEqual(r.status, "ok")

    def test_chords_axis_scaling_and_periodic_translation(self):
        labels = np.random.default_rng(10).integers(0, 3, (9, 7, 5))
        base = f.compute_chord_length_distribution(labels, 1)
        other = f.compute_chord_length_distribution(labels.transpose(2, 1, 0), .2, axis=2)
        np.testing.assert_allclose(other.values_um, np.array(base.values_um) * .2)
        self.assertEqual(base.counts, other.counts)
        rolled = f.compute_chord_length_distribution(np.roll(labels, 4, axis=0), 1)
        self.assertEqual(base.counts, rolled.counts)
        self.assertEqual(base.values_um, rolled.values_um)

    def test_full_pore_line_is_counted_once_and_flagged(self):
        r = f.compute_chord_length_distribution(np.zeros((5, 3, 2)), .5)
        self.assertEqual(r.values_um, [2.5])
        self.assertEqual(r.sample_count, 6)
        self.assertEqual(r.window_spanning_line_count, 6)
        self.assertEqual(r.status, "window_censored")

    def test_contact_direction_and_periodic_seam(self):
        # One pore plane: solid distances on the length-five torus are 1,2,2,1.
        labels = np.ones((5, 2, 3), dtype=np.int8)
        labels[0] = 0
        labels[2] = 2  # CBD belongs to the solid complement too.
        r = f.compute_spherical_contact_distribution(labels, .5)
        self.assertEqual(r.values_um, [.5, 1])
        self.assertEqual(r.counts, [12, 12])
        self.assertEqual(r.sample_count, 24)
        self.assertEqual(r.mean_um, .75)
        self.assertEqual(r.std_um, .25)
        self.assertEqual(r.distance_direction, "solid_to_pore")

    def test_contact_against_all_pairs_minimum_image_distance(self):
        labels = np.random.default_rng(3).integers(0, 3, (5, 4, 7))
        points, targets = np.argwhere(labels != 0), np.argwhere(labels == 0)
        distances = []
        for p in points:
            distances.append(min(math.sqrt(sum(min(abs(int(a-b)), n-abs(int(a-b))) ** 2
                                 for a, b, n in zip(p, q, labels.shape))) for q in targets))
        r = f.compute_spherical_contact_distribution(labels, 1)
        values, counts = np.unique(distances, return_counts=True)
        np.testing.assert_allclose(r.values_um, values)
        np.testing.assert_array_equal(r.counts, counts)
        self.assertAlmostEqual(r.mean_um, np.mean(distances))

    def test_two_point_against_direct_pair_counting(self):
        for shape in [(5, 4, 3), (4, 4, 4), (1, 1, 1)]:
            labels = np.random.default_rng(4).integers(0, 3, shape)
            pore = labels == 0
            r = f.compute_two_point_correlation(labels, .7)
            groups = [[] for _ in r.radii_um]
            for shift in np.ndindex(shape):
                radius = math.sqrt(sum(min(s, n-s) ** 2 for s, n in zip(shift, shape)))
                probability = np.mean(pore & np.roll(pore, shift, axis=(0, 1, 2)))
                groups[math.ceil(radius)].append(probability)
            self.assertEqual(r.displacement_counts, [len(g) for g in groups])
            np.testing.assert_allclose(r.pore_pore_probability, [np.mean(g) for g in groups], atol=1e-14)
            self.assertEqual(r.pore_pore_probability[0], float(pore.mean()))
            weighted = sum(c*n for c, n in zip(r.pore_pore_probability, r.displacement_counts)) / pore.size
            self.assertAlmostEqual(weighted, pore.mean() ** 2)
            for axis, name in enumerate("xyz"):
                expected = [np.mean(pore & np.roll(pore, shift, axis=axis))
                            for shift in range(shape[axis] // 2 + 1)]
                np.testing.assert_allclose(r.axial_pore_pore_probability[name], expected, atol=1e-14)

    def test_two_point_translation_permutation_and_units(self):
        labels = np.random.default_rng(9).integers(0, 2, (7, 6, 5))
        base = f.compute_two_point_correlation(labels, 1)
        other = f.compute_two_point_correlation(np.roll(labels, (2, 1, 3), (0, 1, 2)).transpose(2, 1, 0), .5)
        np.testing.assert_allclose(base.pore_pore_probability, other.pore_pore_probability)
        np.testing.assert_allclose(other.radii_um, np.array(base.radii_um) * .5)
        np.testing.assert_allclose(base.axial_pore_pore_probability['x'], other.axial_pore_pore_probability['z'])

    def test_degenerate_phases_and_strict_json(self):
        for label, status, probability in [(0, "no_solid", 1), (1, "no_pore", 0)]:
            labels = np.full((3, 3, 3), label)
            contact = f.compute_spherical_contact_distribution(labels, 1)
            self.assertEqual(contact.status, status)
            self.assertIsNone(contact.mean_um)
            self.assertEqual(contact.quantiles_um, [None] * 19)
            correlation = f.compute_two_point_correlation(labels, 1)
            self.assertEqual(correlation.pore_pore_probability, [probability] * len(correlation.radii_um))
            json.dumps(f.asdict(contact), allow_nan=False)
        empty = f.compute_chord_length_distribution(np.ones((3, 3, 3)), 1)
        self.assertEqual(empty.status, "no_pore")
        self.assertEqual(empty.sample_count, 0)

    def test_invalid_inputs_and_custom_pore_label(self):
        funcs = [f.compute_chord_length_distribution, f.compute_spherical_contact_distribution,
                 f.compute_two_point_correlation]
        labels = np.array([0, 1, 2, 0, 1])[:, None, None]
        for fn in funcs:
            for h in [0, -1, float('nan'), float('inf')]:
                with self.assertRaises(ValueError):
                    fn(labels, h)
            for bad in [np.zeros((2, 2)), np.zeros((0, 2, 2))]:
                with self.assertRaises(ValueError):
                    fn(bad, 1)
            base, custom = fn(labels, 1), fn(np.where(labels == 0, 7, 8), 1, pore_label=7)
            self.assertEqual(base.status, custom.status)
        with self.assertRaises(ValueError):
            f.compute_chord_length_distribution(labels, 1, axis=3)

    def assert_export(self, directory, expected_axis):
        data = json.loads((directory / 'example_structure_factors.json').read_text(encoding='utf-8'))
        self.assertTrue(set(f.MULTITASK_FACTOR_NAMES) <= data.keys())
        self.assertIn('constrictivity_beta', data)
        chord = data['chord_length_distribution']
        contact = data['spherical_contact_distribution']
        self.assertEqual(chord['axis'], expected_axis)
        self.assertEqual(len(chord['quantiles_um']), 19)
        self.assertEqual(len(contact['quantiles_um']), 19)
        self.assertEqual(contact['distance_direction'], 'solid_to_pore')
        self.assertEqual(data['two_point_correlation']['pore_pore_probability'][0], .5)
        with zipfile.ZipFile(directory / 'structure_factor_example_phase_slices.xlsx') as z:
            xml = z.read('xl/worksheets/sheet1.xml').decode()
            for label in ['Constrictivity', 'Chord Length', 'Spherical Contact', 'Two-Point Correlation']:
                self.assertIn(label, xml)
            self.assertIn(str(chord['mean_um']), xml)

    def test_single_and_batch_cli_exports(self):
        with output_directory() as directory:
            txt = directory / 'example_phase_slices.txt'
            labels = np.ones((6, 4, 4), dtype=np.int8)
            labels[::2] = 0
            write_structure(txt, labels)
            env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            for args, axis in [([str(txt), '--chord-axis', 'z'], 'z'),
                               (['--batch-root', str(directory), '--batch-output-format', 'both', '--chord-axis', 'y'], 'y')]:
                p = subprocess.run([sys.executable, '-B', f.__file__, *args],
                                   capture_output=True, text=True, encoding='utf-8', env=env, timeout=60)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                self.assert_export(directory, axis)

    def test_integrated_pipeline_exports_and_skip_policy(self):
        spec = importlib.util.spec_from_file_location('electrode_kit', Path(f.__file__).with_name('electrode-kit.py'))
        kit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kit)
        self.assertEqual(kit.parse_args(['--chord-axis', 'z']).chord_axis, 'z')
        with output_directory() as directory:
            labels = np.ones((6, 4, 4), dtype=np.int8)
            labels[::2] = 0
            write_structure(directory / 'example_phase_slices.txt', labels)
            kwargs = dict(split_root=directory, split_name='test', voxel_size_um=.5, solid_value=1,
                          liquid_values=(0, 2), unknown_cluster_threshold=.2, write_json=True,
                          write_xlsx=True, chord_axis=2)
            kit.extract_split_structure_factors(**kwargs)
            self.assert_export(directory, 'z')
            target = directory / 'example_structure_factors.json'
            original = target.read_bytes()
            kit.extract_split_structure_factors(**{**kwargs, 'chord_axis': 1})
            self.assertEqual(target.read_bytes(), original)
            kit.extract_split_structure_factors(**{**kwargs, 'chord_axis': 1, 'skip_existing_outputs': False})
            self.assert_export(directory, 'y')


if __name__ == '__main__':
    unittest.main()
