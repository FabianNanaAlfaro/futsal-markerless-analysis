import unittest

from src.biomechanics import angle_at_vertex, derived_kinematics, symmetry_index, velocity_from_positions


class BiomechanicsTest(unittest.TestCase):
    def test_velocity_uses_sampling_rate(self):
        result = velocity_from_positions([[0, 0, 0], [1, 0, 0], [2, 0, 0]], sampling_hz=10)
        self.assertAlmostEqual(result[1][0], 10.0)

    def test_angle_is_geometric_and_missing_safe(self):
        self.assertAlmostEqual(angle_at_vertex([1, 0, 0], [0, 0, 0], [0, 1, 0]), 90.0)
        self.assertIsNone(angle_at_vertex(None, [0, 0, 0], [0, 1, 0]))

    def test_symmetry_is_explicit(self):
        self.assertAlmostEqual(symmetry_index(3, 2), 40.0)
        self.assertAlmostEqual(symmetry_index(2, 3), -40.0)
        self.assertAlmostEqual(symmetry_index(2, 2), 0.0)
        self.assertIsNone(symmetry_index(0, 0))

    def test_derived_angles_use_thesis_signs(self):
        result = derived_kinematics({
            "hip": [[0.0, 0.0, 1.0]],
            "knee": [[0.0, 0.0, 0.0]],
            "ankle": [[-0.5, 0.0, -(3.0 ** 0.5) / 2.0]],
            "foot": [[-0.5 + (3.0 ** 0.5) / 2.0, 0.0, -(3.0 ** 0.5) / 2.0 - 0.5]],
        }, sampling_hz=240.0)
        self.assertAlmostEqual(result["angles_deg"]["knee"][0], 30.0)
        self.assertAlmostEqual(result["angles_deg"]["ankle"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
