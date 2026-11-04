from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from futsal_markerless_analysis.cli import main
from futsal_markerless_analysis.adapters.sports2d import read_sports2d_trc
from futsal_markerless_analysis.geometry import load_projection_matrices, triangulate_point
from futsal_markerless_analysis.kinematics import (
    ankle_dorsiflexion_angle,
    angular_velocity,
    bilateral_difference,
    knee_flexion_angle,
    signed_symmetry_index,
    vector_velocity,
)
from futsal_markerless_analysis.metrics import coordinate_agreement, trajectory_association
from futsal_markerless_analysis.pipeline import process_trial, read_tracks_csv, write_result
from futsal_markerless_analysis.schema import homologate_landmark
from futsal_markerless_analysis.synthetic import SYNTHETIC_CALIBRATION, generate_example
from futsal_markerless_analysis.temporal import (
    curate_2d_point,
    filter_common_trace,
    interpolate_short_gaps,
    normalize_to_phase,
)


class ScientificPipelineTests(unittest.TestCase):
    def test_dlt_reconstructs_known_axes_and_point(self):
        matrices = load_projection_matrices(SYNTHETIC_CALIBRATION)
        expected = np.array([0.31, -0.22, 1.15])
        observed = {
            camera: (matrices[camera] @ np.r_[expected, 1.0])[:2]
            for camera in matrices
        }
        result = triangulate_point(observed, matrices)
        np.testing.assert_allclose(result["xyz"], expected, atol=1e-10)
        self.assertLess(result["max_reprojection_error_px"], 1e-10)

    def test_anatomical_laterality_homologation(self):
        self.assertEqual(homologate_landmark("LEFT_FOOT_INDEX", "left"), "foot")
        self.assertIsNone(homologate_landmark("RIGHT_KNEE", "left"))
        self.assertEqual(homologate_landmark("L_Ankle", "left"), "ankle")

    def test_knee_angle_matches_documented_sagittal_geometry(self):
        # H=(0,0,1), K=(0,0,0), A=(-sin(theta),0,-cos(theta));
        # +X is anterior, so posterior ankle displacement gives positive flexion.
        expected = np.array([0.0, 30.0, 45.0, 90.0, -10.0])
        radians = np.deg2rad(expected)
        hip = np.tile([0.0, 0.0, 1.0], (len(expected), 1))
        knee = np.zeros((len(expected), 3))
        ankle = np.column_stack((-np.sin(radians), np.zeros(len(expected)), -np.cos(radians)))
        np.testing.assert_allclose(knee_flexion_angle(hip, knee, ankle), expected, atol=1e-10)

    def test_ankle_neutral_dorsiflexion_and_plantarflexion_sign(self):
        knee = np.array([[0.0, 0.0, 1.0]] * 3)
        ankle = np.array([[0.0, 0.0, 0.5]] * 3)
        foot = np.array([[0.2, 0.0, 0.5], [0.2, 0.0, 0.6], [0.2, 0.0, 0.4]])
        result = ankle_dorsiflexion_angle(knee, ankle, foot)
        self.assertAlmostEqual(result[0], 0.0, places=8)
        self.assertGreater(result[1], 0.0)
        self.assertLess(result[2], 0.0)

    def test_velocity_uses_sampling_frequency_and_keeps_missing(self):
        positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [np.nan] * 3])
        velocity = vector_velocity(positions, 10.0)
        np.testing.assert_allclose(velocity[:3, 0], [1.0, 1.0, 1.0])
        self.assertTrue(np.isnan(velocity[3]).all())
        angular = angular_velocity([0.0, 10.0, 20.0], 10.0)
        np.testing.assert_allclose(angular, [100.0, 100.0, 100.0])

    def test_angular_velocity_unwraps_the_minus_180_to_180_boundary(self):
        result = angular_velocity([170.0, 179.0, -179.0, -170.0], sampling_hz=10.0)
        np.testing.assert_allclose(result, [90.0, 55.0, 55.0, 90.0])

    def test_position_angle_velocity_and_phase_curve_keep_knee_sign(self):
        angles = np.array([0.0, 30.0, 45.0, 60.0, 90.0])
        radians = np.deg2rad(angles)
        hip = np.tile([0.0, 0.0, 1.0], (len(angles), 1))
        knee = np.zeros((len(angles), 3))
        ankle = np.column_stack((-np.sin(radians), np.zeros(len(angles)), -np.cos(radians)))
        measured = knee_flexion_angle(hip, knee, ankle)
        angular_speed = angular_velocity(measured, sampling_hz=10.0)
        phase = normalize_to_phase(np.arange(len(angles)), measured, 0, len(angles) - 1, 101)
        np.testing.assert_allclose(measured, angles, atol=1e-10)
        self.assertTrue(np.all(angular_speed > 0.0))
        self.assertGreater(phase[-1], phase[0])
        self.assertTrue(np.all(np.diff(phase) >= 0.0))

    def test_bilateral_differences_preserve_oriented_angle_sign(self):
        np.testing.assert_allclose(
            bilateral_difference([30.0, 0.0], [10.0, 20.0]), [20.0, -20.0]
        )

    def test_short_gap_is_interpolated_but_long_and_edge_gaps_remain(self):
        values = np.array([np.nan, 0.0, np.nan, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan, 9.0])
        result = interpolate_short_gaps(values, max_gap=2)
        self.assertTrue(np.isnan(result[0]))
        self.assertAlmostEqual(result[2], 1.0)
        self.assertTrue(np.isnan(result[4:9]).all())

    def test_2d_outlier_mask_removes_both_coordinates_and_bridges_small_gap(self):
        xy = np.column_stack((np.arange(31, dtype=float), 2.0 * np.arange(31, dtype=float)))
        xy[15] = [1000.0, -1000.0]
        result = curate_2d_point(xy)
        self.assertTrue(np.isfinite(result[15]).all())
        self.assertLess(abs(result[15, 0] - 15.0), 2.0)
        self.assertLess(abs(result[15, 1] - 30.0), 3.0)

    def test_common_filter_preserves_missing_interval(self):
        x = np.sin(np.linspace(0, 2 * np.pi, 100))
        x[40:45] = np.nan
        result = filter_common_trace(x, sampling_hz=240.0, cutoff_hz=18.0, order=1, lowess_span=0.10)
        self.assertTrue(np.isnan(result[40:45]).all())
        self.assertTrue(np.isfinite(result[:40]).all())
        self.assertTrue(np.isfinite(result[45:]).all())

    def test_normalization_resamples_declared_window_and_preserves_gap(self):
        frames = np.arange(11)
        values = frames.astype(float)
        values[4:6] = np.nan
        normalized = normalize_to_phase(frames, values, 0, 10, 11)
        np.testing.assert_allclose(normalized[:4], [0, 1, 2, 3])
        self.assertTrue(np.isnan(normalized[4:6]).all())
        np.testing.assert_allclose(normalized[6:], np.arange(6, 11))

    def test_signed_symmetry_has_direction_and_zero_denominator_is_missing(self):
        result = signed_symmetry_index([1.5, 0.5, 0.0], [1.0, 1.0, 0.0])
        self.assertAlmostEqual(result[0], 40.0)
        self.assertAlmostEqual(result[1], -200.0 / 3.0)
        self.assertTrue(np.isnan(result[2]))

    def test_metrics_and_temporal_association(self):
        reference = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        prediction = reference + np.array([0.1, -0.2])
        result = coordinate_agreement(reference, prediction, normalization_scale=2.0)
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["axis_bias"][0], 0.1)
        self.assertAlmostEqual(result["axis_bias"][1], -0.2)
        self.assertGreater(result["ccc_by_axis"][0], 0.9)
        assoc = trajectory_association([0, 1, 2, 3], [0, 2, 4, 6])
        self.assertAlmostEqual(assoc["pearson_r"], 1.0)

    def test_full_synthetic_pipeline_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            tracks, calibration = generate_example(output / "inputs", frames=40)
            result = process_trial(
                read_tracks_csv(tracks), json.loads(calibration.read_text()),
                selected_side="right", source="markerless", sampling_hz=240.0,
                impact_frame=39, window_start_frame=0, cameras=["CAM_1", "CAM_2"],
            )
            write_result(result, output / "results")
            expected = {"tracks_2d_processed.csv", "kinematics_3d.csv",
                        "kinematics_normalized_101.csv", "reprojection_diagnostics.csv", "run_metadata.json"}
            self.assertTrue(expected.issubset({path.name for path in (output / "results").iterdir()}))
            self.assertEqual(len(result["kinematics"]), 40)
            self.assertEqual(len(result["normalized"]["phase_percent"]), 101)
            self.assertFalse(result["metadata"]["legacy_dvideow_equivalence_validated"])

    def test_sports2d_trc_exports_side_filtered_canonical_pixel_tracks(self):
        trc = "\n".join([
            "PathFileType\t4\t(X/Y/Z)\tsample.trc",
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
            "240\t240\t2\t3\tm\t240\t0\t2",
            "Frame#\tTime\tRHip\t\t\tLHip\t\t\tRBigToe\t\t\t",
            "\t\tX1\tY1\tZ1\tX2\tY2\tZ2\tX3\tY3\tZ3",
            "0\t0.5\t100\t200\t0\t300\t400\t0\t110\t210\t0",
            "1\t0.5041666667\t101\t201\t0\t301\t401\t0\t0\t0\t0",
        ])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.trc"
            path.write_text(trc, encoding="utf-8")
            rows = read_sports2d_trc(path, camera="CAM_2", side="right", frame_rate=240.0)
        self.assertEqual([(r["frame_index"], r["landmark"]) for r in rows],
                         [(120, "RHip"), (120, "RBigToe"), (121, "RHip"), (121, "RBigToe")])
        self.assertTrue(np.isnan(rows[-1]["x_px"]))
        self.assertEqual(rows[0]["camera"], "CAM_2")
        self.assertEqual(rows[0]["x_px"], 100.0)

    def test_cli_demo_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            code = main(["demo", "--output", str(Path(temp) / "demo"), "--frames", "32"])
            self.assertEqual(code, 0)
            self.assertTrue((Path(temp) / "demo" / "results" / "kinematics_3d.csv").is_file())


if __name__ == "__main__":
    unittest.main()
