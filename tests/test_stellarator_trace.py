import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from stellaratorTraceClass import (
    CylindricalTarget,
    bin_limiter_heat_load,
    summarize_limiter_result,
    trace_to_target,
)


class RadialField:
    def B_xyz(self, points, chunk_size=256):
        pts = np.asarray(points, dtype=float)
        single = pts.ndim == 1
        pts = np.atleast_2d(pts)
        B = np.zeros_like(pts)
        R = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        B[:, 0] = pts[:, 0] / R
        B[:, 1] = pts[:, 1] / R
        if single:
            return B[0]
        return B


class StellaratorTraceTest(unittest.TestCase):
    def test_trace_to_cylindrical_target(self):
        target = CylindricalTarget(radius=1.5, z_min=-0.1, z_max=0.1)

        result = trace_to_target(
            RadialField(),
            [1.0, 0.0, 0.0],
            target,
            step_m=0.05,
            max_steps=20,
        )

        self.assertTrue(result["hit"])
        self.assertAlmostEqual(result["connection_length_m"], 0.5, delta=2.0e-3)
        self.assertAlmostEqual(result["hit_xyz"][0], 1.5, delta=1.0e-6)
        self.assertAlmostEqual(result["abs_bdotn"], 1.0, delta=1.0e-10)

    def test_heat_load_bins_and_normalizes_power(self):
        target = CylindricalTarget(radius=2.0, z_min=-1.0, z_max=1.0)
        rows = [
            {"hit": True, "hit_phi": 0.1, "hit_z": -0.5, "weight": 1.0,
             "connection_length_m": 1.0, "abs_bdotn": 0.5},
            {"hit": True, "hit_phi": 1.1, "hit_z": 0.5, "weight": 3.0,
             "connection_length_m": 3.0, "abs_bdotn": 0.25},
            {"hit": False, "hit_phi": np.nan, "hit_z": np.nan, "weight": 100.0,
             "connection_length_m": 10.0, "abs_bdotn": np.nan},
        ]

        heat = bin_limiter_heat_load(rows, target, total_power_w=400.0, n_phi=4, n_z=4)
        summary = summarize_limiter_result(rows, heat, total_power_w=400.0)

        self.assertAlmostEqual(np.sum(heat["power_w"]), 400.0)
        self.assertEqual(summary["n_traces"], 3)
        self.assertEqual(summary["n_hits"], 2)
        self.assertAlmostEqual(summary["hit_fraction"], 2.0/3.0)
        self.assertGreater(summary["peak_q_mw_m2"], 0.0)


if __name__ == "__main__":
    unittest.main()
