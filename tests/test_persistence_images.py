"""Correctness and I/O checks for the Python persistence-image implementation."""

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from persistence_images import (
    Bounds,
    compute_bounds,
    load_dataset,
    persistence_image,
    persistence_image_reference,
    read_diagram,
    remove_infinite_class,
    to_scaled_diagrams,
)
from persistence_images.cli import main


class CoreTests(unittest.TestCase):
    def test_single_point_closed_form_and_row_major(self):
        bounds = Bounds(0.0, 2.0, 0.0, 4.0)
        arr = persistence_image([[1.0, 1.0]], bounds, 2, 3, 1.0, 2.0)
        expected = [
            math.exp(-0.5 * (((x - 1.0) / 1.0) ** 2 + ((y - 1.0) / 2.0) ** 2))
            for x in [0.5, 1.5]
            for y in [2.0 / 3.0, 2.0, 10.0 / 3.0]
        ]
        self.assertEqual(arr.shape, (6,))
        np.testing.assert_allclose(arr, expected, rtol=2e-15, atol=1e-15)

    def test_optimized_matches_direct_reference(self):
        rng = np.random.default_rng(734)
        for n in [0, 1, 2, 3, 10, 128]:
            dgm = rng.uniform([-0.8, 0.0], [2.5, 3.0], size=(n, 2))
            bounds = Bounds(-1.0, 3.0, -0.5, 4.0)
            expected = persistence_image_reference(dgm, bounds, 11, 7, 0.42, 1.12)
            for batch_size in [1, 3, 12, 4096]:
                with self.subTest(n=n, batch_size=batch_size):
                    observed = persistence_image(dgm, bounds, 11, 7, 0.42, 1.12,
                                                 batch_size=batch_size)
                    np.testing.assert_allclose(observed, expected, rtol=3e-13, atol=2e-14)

    def test_degenerate_bounds_and_empty(self):
        bounds = Bounds(2.0, 2.0, 1.0, 1.0)
        arr = persistence_image([[2.0, 1.0]], bounds, 3, 2, 0.1, 0.2)
        np.testing.assert_array_equal(arr, np.ones(6))
        empty = persistence_image([], bounds, 3, 2)
        np.testing.assert_array_equal(empty, np.zeros(6))
        self.assertIsNone(compute_bounds([[], np.empty((0, 2))]))
        self.assertEqual(compute_bounds([[], [[2.0, 1.0], [3.0, 5.0]]]),
                         Bounds(2.0, 3.0, 1.0, 5.0))

    def test_no_weighting_and_no_normalization(self):
        bounds = Bounds(0, 0, 1, 1)
        one = persistence_image([[0, 1]], bounds, 1, 1, 0.3, 0.4)
        two = persistence_image([[0, 1], [0, 1]], bounds, 1, 1, 0.3, 0.4)
        np.testing.assert_array_equal(one, [1.0])
        np.testing.assert_array_equal(two, [2.0])

    def test_reject_invalid_parameters(self):
        bounds = Bounds(0, 1, 0, 1)
        with self.assertRaises(ValueError):
            persistence_image([[0, 0]], bounds, nx=0)
        with self.assertRaises(ValueError):
            persistence_image([[0, 0]], bounds, sigma_x=0)
        with self.assertRaises(ValueError):
            persistence_image([[0, 0]], bounds, batch_size=0)
        with self.assertRaises(ValueError):
            persistence_image([[0, 0, 0]], bounds)


class IoAndCliTests(unittest.TestCase):
    def test_remove_h0_uses_min_birth_tie_max_death(self):
        arr = np.array([[0.2, 2.0], [0.1, 0.4], [0.1, 3.0], [0.8, 5.0]])
        expected = np.array([[0.2, 2.0], [0.1, 0.4], [0.8, 5.0]])
        np.testing.assert_array_equal(remove_infinite_class(arr), expected)
        np.testing.assert_array_equal(remove_infinite_class([]), np.empty((0, 2)))

    def test_read_diagram_converts_to_persistence(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.csv"
            path.write_text("0.3,2.1,unused\n0.1,3.0\n", encoding="utf-8")
            np.testing.assert_allclose(read_diagram(path), [[0.3, 1.8], [0.1, 2.9]])
            np.testing.assert_allclose(read_diagram(path, remove_h0=True), [[0.3, 1.8]])

    def test_dataset_spacing_and_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            d = root / "nested"
            d.mkdir()
            (d / "a.csv").write_text("0.5,1.5\n")
            (d / "b.csv").write_text("1,2\n")
            (d / "irrelevant.txt").write_text("irrelevant")
            metadata = root / "spacing.csv"
            metadata.write_text("filename,notes,spacing\na.tif,,2\n")
            output = root / "images.csv"
            output.write_text("not,a,diagram\n")
            dataset = load_dataset(root, metadata, output)
            self.assertEqual([s.path.stem for s in dataset], ["a", "b"])
            self.assertEqual([s.voxel_spacing for s in dataset], [2.0, 1.0])
            np.testing.assert_allclose(to_scaled_diagrams(dataset), [
                [[1.0, 2.0]], [[1.0, 1.0]]
            ])

    def test_end_to_end_cli_agrees_with_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            diagram_dir = root / "diagrams"
            diagram_dir.mkdir()
            (diagram_dir / "im1.csv").write_text("0,2\n0.4,0.7\n")
            (diagram_dir / "im2.csv").write_text("1,2\n1.2,1.9\n")
            metadata = root / "spacings.csv"
            metadata.write_text("filename,other,spacing\nim1.tif,,1.4\nim2.tif,,0.75\n")
            output = root / "images.csv"
            code = main(["--input", str(diagram_dir), "--spacing", str(metadata),
                         "--output", str(output), "--nx", "4", "--ny", "3",
                         "--sigma-x", "0.5", "--sigma-y", "0.8",
                         "--batch-size", "1"])
            self.assertEqual(code, 0)
            with output.open(newline="") as f:
                result = list(csv.reader(f))
            self.assertEqual([row[0] for row in result], ["im1", "im2"])
            scaled = [np.array([[0, 2], [0.4, 0.3]]) * 1.4,
                      np.array([[1, 1], [1.2, 0.7]]) * 0.75]
            bounds = compute_bounds(scaled)
            for row, diagram, spacing in zip(result, scaled, [1.4, 0.75]):
                ref = persistence_image_reference(diagram, bounds, 4, 3,
                                                  0.5 * spacing, 0.8 * spacing)
                np.testing.assert_allclose(np.array(row[1:], dtype=float), ref,
                                           rtol=3e-14, atol=1e-14)

    def test_cli_empty_dataset_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = root / "spacing.csv"
            metadata.write_text("filename,unused,spacing\n")
            out = root / "out.csv"
            code = main(["--input", str(root), "--spacing", str(metadata),
                         "--output", str(out)])
            self.assertEqual(code, 0)
            self.assertFalse(out.exists())


class EngineTests(unittest.TestCase):
    def test_factorization_quantized_matches_reference(self):
        rng = np.random.default_rng(2026)
        coords = rng.integers([0, 0], [16, 8], size=(900, 2)).astype(float) / [16, 8]
        bounds = Bounds(-0.5, 1.5, -0.2, 1.2)
        expected = persistence_image_reference(coords, bounds, 35, 37, 0.22, 0.11)
        for engine in ("auto", "factorized", "dense"):
            with self.subTest(engine=engine):
                actual = persistence_image(coords, bounds, 35, 37, 0.22, 0.11, engine=engine)
                np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-12)

    def test_auto_agrees_with_dense_for_continuous_data(self):
        rng = np.random.default_rng(321)
        diagram = rng.random((1200, 2))
        bounds = Bounds(0, 1, 0, 1)
        dense = persistence_image(diagram, bounds, 33, 32, 0.3, 0.5, engine="dense")
        auto = persistence_image(diagram, bounds, 33, 32, 0.3, 0.5, engine="auto")
        np.testing.assert_array_equal(auto, dense)

    def test_same_duplicate_points_count_with_multiplicity(self):
        bounds = Bounds(-1, 1, -1, 1)
        points = np.repeat([[0.25, 0.5]], 512, axis=0)
        for engine in ("dense", "factorized", "auto"):
            value = persistence_image(points, bounds, 32, 32, 0.6, 0.3, engine=engine)
            one = persistence_image(points[:1], bounds, 32, 32, 0.6, 0.3)
            np.testing.assert_allclose(value, 512 * one, rtol=1e-13, atol=1e-12)

    def test_invalid_engine_and_allocation_cap(self):
        with self.assertRaises(ValueError):
            persistence_image([[1, 1]], Bounds(0, 1, 0, 1), engine="unknown")
        # Full disjoint coordinate set causes 1.5e3**2 entries: over 2e6 cap.
        points = np.column_stack([np.arange(1500), np.arange(1500)+0.25])
        with self.assertRaises(ValueError):
            persistence_image(points, Bounds(0, 1500, 0, 1501), engine="factorized")
        # Auto must fall back instead of allocating O(n**2) memory.
        auto = persistence_image(points, Bounds(0, 1500, 0, 1501), 32, 32, engine="auto")
        self.assertEqual(len(auto), 1024)

class StreamingTests(unittest.TestCase):
    def test_stream_and_default_agree(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dgmdir = root / "diagrams"
            dgmdir.mkdir()
            (dgmdir / "a.csv").write_text("0.2,1.4\n0.2,1.4\n0.3,0.7\n")
            (dgmdir / "b.csv").write_text("0.1,0.9\n0.5,1.3\n")
            spacing = root / "spacing.csv"
            spacing.write_text("filename,ignored,spacing\na.tif,,1.2\nb.tif,,0.5\n")
            dense = root / "dense.csv"
            streamed = root / "stream.csv"
            shared = ["--input", str(dgmdir), "--spacing", str(spacing),
                      "--nx", "33", "--ny", "32", "--engine", "auto"]
            self.assertEqual(main(shared + ["--output", str(dense)]), 0)
            self.assertEqual(main(shared + ["--output", str(streamed), "--stream"]), 0)
            self.assertEqual(dense.read_text(), streamed.read_text())

class SparseTests(unittest.TestCase):
    @unittest.skipUnless(__import__("importlib").util.find_spec("numba"), "requires optional Numba")
    def test_error_bound_and_zero_extent(self):
        rng = np.random.default_rng(98)
        for shape, sigma in [((20, 25), 0.02), ((32, 27), 0.5), ((1, 17), 0.12)]:
            points = rng.random((200, 2))
            bounds = Bounds(0, 1, 0, 1)
            ref = persistence_image(points, bounds, *shape, sigma, sigma, engine="dense")
            out = persistence_image(points, bounds, *shape, sigma, sigma,
                                    engine="sparse", absolute_error=1e-8)
            self.assertLessEqual(float(np.max(np.abs(ref - out))), 1e-8 + 1e-12)
        bounds = Bounds(0.5, 0.5, 0.5, 0.5)
        points = rng.random((33, 2))
        for shape in ((4, 5), (1, 1)):
            ref = persistence_image(points, bounds, *shape, 0.01, 0.01, engine="dense")
            out = persistence_image(points, bounds, *shape, 0.01, 0.01,
                                    engine="sparse", absolute_error=1e-8)
            self.assertLessEqual(float(np.max(np.abs(ref - out))), 1e-8 + 1e-12)

    def test_sparse_invalid_tolerance(self):
        with self.assertRaises(ValueError):
            persistence_image([[0, 1]], Bounds(0, 1, 0, 1),
                              engine="sparse", absolute_error=0)

if __name__ == "__main__":
    unittest.main()

# These tests deliberately appear after the original unittest.main guard;
# pytest discovers them normally, and `python -m unittest discover` also does.
from persistence_images import SharedGaussianBasis, batch_persistence_images

class SharedBasisTests(unittest.TestCase):
    def test_cached_basis_matches_individual_at_mixed_spacings(self):
        rng = np.random.default_rng(817)
        diagrams = []
        spacing = [1.0, 1.0, 0.5, 1.0, 0.5, 2.0]
        for s in spacing:
            coords = rng.integers(0, 12, size=(800, 2)).astype(float) * s / 12
            diagrams.append(coords)
        bounds = compute_bounds(diagrams)
        for engine in ('auto', 'shared', 'dense'):
            images = batch_persistence_images(diagrams, bounds, 36, 35, 0.2, 0.3,
                                              spacings=spacing, engine=engine)
            self.assertEqual(len(images), len(diagrams))
            for diagram, sp, image in zip(diagrams, spacing, images):
                reference = persistence_image_reference(diagram, bounds, 36, 35,
                                                        0.2 * sp, 0.3 * sp)
                np.testing.assert_allclose(image, reference, rtol=1e-12, atol=1e-11)

    def test_manual_basis_rejects_unseen_coordinates_and_caps(self):
        bounds = Bounds(-1, 2, -1, 3)
        diagrams = [np.array([[0, 1], [0, 1], [1, 2.0]])]
        b = SharedGaussianBasis.from_diagrams(diagrams, bounds, 4, 5, 0.2, 0.3)
        ref = persistence_image_reference(diagrams[0], bounds, 4, 5, 0.2, 0.3)
        np.testing.assert_allclose(b.transform(diagrams[0]), ref, atol=1e-13)
        with self.assertRaises(ValueError):
            b.transform([[0.33, 1]])
        with self.assertRaises(ValueError):
            SharedGaussianBasis.from_diagrams(diagrams, bounds, 4, 5, 0.2, 0.3, max_cells=1)
        np.testing.assert_array_equal(b.transform([]), np.zeros(20))

    def test_default_auto_is_exact_and_sparse_optin_is_bounded(self):
        rng = np.random.default_rng(919)
        diagram = rng.random((300, 2))
        bounds = Bounds(0, 1, 0, 1)
        exact = persistence_image(diagram, bounds, 64, 60, .006, .008, engine='dense')
        auto_exact = persistence_image(diagram, bounds, 64, 60, .006, .008, engine='auto')
        np.testing.assert_array_equal(auto_exact, exact)
        approx = persistence_image(diagram, bounds, 64, 60, .006, .008, engine='auto',
                                   absolute_error=1e-9)
        self.assertLessEqual(float(np.max(np.abs(approx - exact))), 1e-9 + 2e-12)

    def test_mismatched_spacings(self):
        with self.assertRaises(ValueError):
            batch_persistence_images([[[0, 1]]], Bounds(0, 1, 0, 1), spacings=[])
        with self.assertRaises(ValueError):
            batch_persistence_images([[[0, 1]]], Bounds(0, 1, 0, 1), spacings=[-2])

    def test_shared_cli_matches_dense_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dgms = root / 'dgms'; dgms.mkdir()
            for k in range(4):
                (dgms / f'im{k}.csv').write_text('0,1\n0,1\n0.5,2\n')
            metadata = root / 'spacing.csv'
            metadata.write_text('filename,note,spacing\n' +
                                ''.join(f'im{k}.tif,,{1.0 if k < 3 else 0.5}\n' for k in range(4)))
            files = []
            for engine in ('shared', 'dense'):
                out = root / f'{engine}.csv'
                self.assertEqual(main(['--input', str(dgms), '--spacing', str(metadata),
                                       '--output', str(out), '--nx', '25', '--ny', '26',
                                       '--engine', engine]), 0)
                with out.open(newline="") as f:
                    files.append(list(csv.reader(f)))
            for a, b in zip(*files):
                self.assertEqual(a[0], b[0])
                np.testing.assert_allclose(np.array(a[1:], dtype=float),
                                           np.array(b[1:], dtype=float), atol=2e-12)
