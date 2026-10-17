import sys
from pathlib import Path
from contextlib import contextmanager
from uuid import uuid4
import unittest
import json
import zipfile
from collections import deque
from itertools import product
import subprocess
import os
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from constrictivity import compute_constrictivity, _distance_to, _inlet_connected
import electrode_structure_factors as factors


def channel(neck=False):
    a = np.ones((25, 17, 17), dtype=np.int8)
    a[:, 3:14, 3:14] = 0
    if neck:
        a[3:6] = 1
        a[3:6, 7:10, 7:10] = 0
    return a


@contextmanager
def output_directory():
    # Python 3.13 mkdtemp's private Windows ACL can exclude sandbox tokens.
    directory = Path(__file__).resolve().parent / ("output_" + uuid4().hex)
    directory.mkdir(mode=0o755)
    try:
        yield directory
    finally:
        for name in ("example_phase_slices.txt", "structure_factor_example_phase_slices.xlsx", "result.json", "example_structure_factors.json"):
            (directory / name).unlink(missing_ok=True)
        directory.rmdir()



def brute_connected(mask, axis, connectivity, side):
    offsets = [v for v in product((-1, 0, 1), repeat=3)
               if 0 < sum(abs(x) for x in v) <= {6: 1, 18: 2, 26: 3}[connectivity]]
    reached = np.zeros_like(mask)
    face = 0 if side == "low" else mask.shape[axis] - 1
    pending = deque(tuple(p) for p in np.argwhere(mask) if p[axis] == face)
    for point in pending:
        reached[point] = True
    while pending:
        point = pending.popleft()
        for offset in offsets:
            q = [p + d for p, d in zip(point, offset)]
            if not 0 <= q[axis] < mask.shape[axis]:
                continue
            q = tuple(v if a == axis else v % mask.shape[a] for a, v in enumerate(q))
            if mask[q] and not reached[q]:
                reached[q] = True
                pending.append(q)
    return reached


def brute_distances(points, targets, shape, periodic):
    if not len(targets):
        return np.full(len(points), np.inf)
    delta = abs(points[:, None, :] - targets[None, :, :])
    for a in periodic:
        delta[:, :, a] = np.minimum(delta[:, :, a], shape[a] - delta[:, :, a])
    return np.sqrt((delta * delta).sum(axis=2).min(axis=1))

class ConstrictivityTests(unittest.TestCase):
    def test_straight_and_bottleneck(self):
        straight = compute_constrictivity(channel(), 1)
        neck = compute_constrictivity(channel(True), 1)
        self.assertEqual(straight.constrictivity_beta, 1)
        self.assertGreater(neck.constrictivity_beta, 0)
        self.assertLess(neck.constrictivity_beta, straight.constrictivity_beta)
        self.assertTrue(np.all(np.array(neck.mip) <= np.array(neck.cpsd)))

    def test_scale_and_axis(self):
        a = channel(True)
        x = compute_constrictivity(a, 1)
        z = compute_constrictivity(a.transpose(2, 1, 0), 0.5, axis=2)
        self.assertEqual(x.constrictivity_beta, z.constrictivity_beta)
        self.assertEqual(x.r_min_um / 2, z.r_min_um)
        self.assertEqual(x.r_max_um / 2, z.r_max_um)

    def test_degenerate_and_closed_pores(self):
        for value, status in [(0, 'no_solid_interface'), (1, 'no_pores')]:
            r = compute_constrictivity(np.full((4, 4, 4), value), 1)
            self.assertEqual(r.status, status)
            self.assertIsNone(r.constrictivity_beta)
        a = np.ones((9, 9, 9), dtype=int)
        a[3:7, 3:7, 3:7] = 0
        r = compute_constrictivity(a, 1)
        self.assertEqual(r.constrictivity_beta, 0)
        self.assertEqual(r.status, 'insufficient_inlet_access')

    def test_periodic_distances_against_brute_force(self):
        rng = np.random.default_rng(9)
        for shape in [(3, 4, 5), (1, 3, 6), (9, 2, 3)]:
            target = rng.random(shape) < .15
            target.flat[0] = True
            points = np.argwhere(target)
            for axes in [(0, 1, 2), (1, 2)]:
                actual = _distance_to(target, axes)
                for index in np.ndindex(shape):
                    delta = abs(points - np.array(index))
                    for axis in axes:
                        delta[:, axis] = np.minimum(delta[:, axis], shape[axis] - delta[:, axis])
                    self.assertAlmostEqual(actual[index], np.sqrt((delta ** 2).sum(1).min()))

    def test_transverse_wrap_but_no_transport_wrap(self):
        a = np.zeros((5, 5, 5), dtype=bool)
        a[0:3, 0, 2] = True
        a[2, -1, 2] = True
        a[-1, 0, 2] = True
        reached = _inlet_connected(a, 0)
        self.assertTrue(reached[2, -1, 2])
        self.assertFalse(reached[-1, 0, 2])

    def test_invalid_input(self):
        for h in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                compute_constrictivity(channel(), h)
        with self.assertRaises(ValueError):
            compute_constrictivity(np.zeros((2, 2)), 1)

    def test_critical_radius_not_rounded_to_integer(self):
        # A five-voxel plus cross-section has center clearance sqrt(2).
        labels = np.ones((7, 7, 7), dtype=int)
        y, z = np.indices((7, 7))
        labels[:, (y - 3) ** 2 + (z - 3) ** 2 <= 1] = 0
        result = compute_constrictivity(labels, .5)
        self.assertAlmostEqual(result.r_max_um, np.sqrt(2) * .5)
        self.assertAlmostEqual(result.r_min_um, np.sqrt(2) * .5)
        self.assertEqual(result.constrictivity_beta, 1)

    def test_connectivity_all_axes_and_sides_against_bfs(self):
        rng = np.random.default_rng(27)
        for shape in [(5, 4, 6), (1, 4, 5)]:
            mask = rng.random(shape) < .18
            for axis in (0, 1, 2):
                for side in ("low", "high"):
                    for conn in (6, 18, 26):
                        np.testing.assert_array_equal(
                            _inlet_connected(mask, axis, conn, side),
                            brute_connected(mask, axis, conn, side))

    def test_diagonal_periodic_corner(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[0, 0, 0] = mask[1, 4, 4] = True
        self.assertTrue(_inlet_connected(mask, 0, 26)[1, 4, 4])
        self.assertFalse(_inlet_connected(mask, 0, 18)[1, 4, 4])
        self.assertFalse(_inlet_connected(mask, 0, 6)[1, 4, 4])

    def test_bounded_halo_matches_full_periodic_edt(self):
        rng = np.random.default_rng(89)
        for shape in [(13, 14, 15), (1, 7, 9)]:
            mask = rng.random(shape) < .008
            mask.flat[-1] = True
            for axes in [(0, 1, 2), (1, 2), (0, 2), ()]:
                full = _distance_to(mask, axes)
                for radius in (0, 1, np.sqrt(2), 3, 8):
                    truncated = _distance_to(mask, axes, max_distance=radius)
                    expected = np.where(full <= radius, full, np.inf)
                    np.testing.assert_array_equal(truncated, expected)

    def test_curves_against_direct_sphere_enumeration(self):
        # Independent all-pairs distances + BFS: no EDT, labels or halo logic.
        rng = np.random.default_rng(51)
        pore = rng.random((5, 6, 7)) < .85
        points = np.argwhere(pore)
        clearance = brute_distances(points, np.argwhere(~pore), pore.shape, (0, 1, 2))
        for axis, side, conn in [(0, "low", 26), (2, "high", 6), (1, "low", 18)]:
            result = compute_constrictivity(np.where(pore, 0, 1), 1, axis=axis,
                                           connectivity=conn, inlet_side=side)
            cpsd, mip = [], []
            for radius in result.radii_um:
                centers = np.zeros_like(pore)
                selected = points[clearance >= radius]
                if len(selected):
                    centers[tuple(selected.T)] = True
                accessible = brute_connected(centers, axis, conn, side)
                cpsd.append(np.count_nonzero(brute_distances(
                    points, selected, pore.shape, (0, 1, 2)) <= radius) / pore.size)
                mip.append(np.count_nonzero(brute_distances(
                    points, np.argwhere(accessible), pore.shape,
                    tuple(a for a in range(3) if a != axis)) <= radius) / pore.size)
            np.testing.assert_allclose(result.cpsd, cpsd, rtol=0, atol=0)
            np.testing.assert_allclose(result.mip, mip, rtol=0, atol=0)
            self.assertTrue(np.all(np.array(mip) <= np.array(cpsd)))
            expected_max = max(r for r, v in zip(result.radii_um, cpsd) if v >= pore.mean()/2)
            self.assertEqual(result.r_max_um, expected_max)

    def test_high_inlet_equals_flipped_low_inlet(self):
        a = channel(True)
        high = compute_constrictivity(a, 1, inlet_side="high")
        flipped = compute_constrictivity(a[::-1], 1)
        np.testing.assert_array_equal(high.mip, flipped.mip)
        self.assertGreater(high.r_min_um, compute_constrictivity(a, 1).r_min_um)

    def test_half_volume_tie_and_isolated_pore_denominator(self):
        labels = np.ones((9, 7, 7), dtype=int)
        labels[0, 3, 3] = labels[4, 3, 3] = 0
        self.assertEqual(compute_constrictivity(labels, 1).r_min_um, 1)
        labels[5, 3, 3] = 0
        result = compute_constrictivity(labels, 1)
        self.assertEqual(result.r_min_um, 0)
        self.assertEqual(result.status, 'insufficient_inlet_access')

    def test_invalid_connectivity_direction_and_inlet(self):
        for kw in [dict(connectivity=8), dict(axis=3), dict(inlet_side="both")]:
            with self.assertRaises(ValueError):
                compute_constrictivity(channel(), 1, **kw)

    def test_pipeline_and_exports(self):
        a = channel(True)
        with output_directory() as d:
            txt = Path(d) / 'example_phase_slices.txt'
            txt.write_text('\n\n'.join('\n'.join(' '.join(map(str, row)) for row in a[:, :, z]) for z in range(a.shape[2])), encoding='utf-8')
            output, area, fractions, agg, ec, lc, diameter, tau, constr, chord, contact, two_point = factors.process_single_structure_file(txt, .5, 1, (0, 2), .2,
                constrictivity_axis=2, constrictivity_connectivity=18, constrictivity_inlet="high")
            payload = factors.build_multitask_structure_factor_payload(fractions, area, ec, lc, diameter, constr)
            dest = Path(d) / 'result.json'
            factors.write_multitask_structure_factor_json(dest, payload)
            loaded = json.loads(dest.read_text(encoding='utf-8'))
            self.assertEqual(loaded['constrictivity_beta'], constr.constrictivity_beta)
            self.assertTrue(set(factors.MULTITASK_FACTOR_NAMES) <= loaded.keys())
            details = loaded['constrictivity_details']
            self.assertEqual(details['axis'], 'z')
            self.assertEqual(details['inlet_side'], 'high')
            self.assertEqual(details['connectivity'], 18)
            self.assertEqual(details['method'], 'edt_critical_radii_closed_dilation_v2')
            self.assertIsNone(details['radius_step_um'])
            with zipfile.ZipFile(output) as z:
                self.assertIn('Constrictivity', z.read('xl/worksheets/sheet1.xml').decode())
            completed = subprocess.run([
                sys.executable, "-B", str(Path(factors.__file__).resolve()),
                "--batch-root", str(d), "--batch-output-format", "both",
                "--constrictivity-axis", "y", "--constrictivity-connectivity", "6",
                "--constrictivity-inlet", "high",
            ], capture_output=True, text=True, encoding="utf-8", timeout=60,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            batch = json.loads((Path(d) / 'example_structure_factors.json').read_text(encoding='utf-8'))
            self.assertEqual(batch['constrictivity_details']['axis'], 'y')
            self.assertEqual(batch['constrictivity_details']['connectivity'], 6)
            self.assertEqual(batch['constrictivity_details']['inlet_side'], 'high')


if __name__ == '__main__':
    unittest.main()
