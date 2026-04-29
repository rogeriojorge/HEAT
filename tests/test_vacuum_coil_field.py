import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from vacuumFieldClass import VacuumCoilField


def write_json(data):
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(data, handle)
    handle.close()
    return handle.name


class VacuumCoilFieldTest(unittest.TestCase):
    def test_reads_essos_json_and_evaluates_axis_field(self):
        current = 1.0e5
        radius = 1.2
        data = {
            "nfp": 1,
            "stellsym": False,
            "order": 1,
            "n_segments": 400,
            "dofs_curves": [
                [
                    [0.0, 0.0, radius],
                    [0.0, radius, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ],
            "dofs_currents": [current],
        }
        field = VacuumCoilField.from_json(write_json(data))

        B = field.B_xyz([[0.0, 0.0, 0.0]])[0]
        expected_Bz = 4.0 * np.pi * 1.0e-7 * current / (2.0 * radius)

        self.assertEqual(field.n_coils, 1)
        self.assertAlmostEqual(B[0], 0.0, places=10)
        self.assertAlmostEqual(B[1], 0.0, places=10)
        self.assertAlmostEqual(B[2], expected_Bz, delta=expected_Bz * 1.0e-4)

    def test_reads_minimal_simsopt_json(self):
        current = 7.5e4
        radius = 0.9
        data = {
            "@module": "simsopt._core.json",
            "@class": "SIMSON",
            "@version": "test",
            "graph": {"$type": "ref", "value": "BiotSavart1"},
            "simsopt_objs": {
                "DOF1": {
                    "@class": "DOFs",
                    "x": {
                        "data": [
                            0.0, 0.0, radius,
                            0.0, radius, 0.0,
                            0.0, 0.0, 0.0,
                        ]
                    },
                },
                "CurveXYZFourier1": {
                    "@class": "CurveXYZFourier",
                    "quadpoints": {"data": np.linspace(0.0, 1.0, 400, endpoint=False).tolist()},
                    "order": 1,
                    "dofs": {"$type": "ref", "value": "DOF1"},
                },
                "Current1": {"@class": "Current", "current": current},
                "Coil1": {
                    "@class": "Coil",
                    "curve": {"$type": "ref", "value": "CurveXYZFourier1"},
                    "current": {"$type": "ref", "value": "Current1"},
                },
                "BiotSavart1": {
                    "@class": "BiotSavart",
                    "coils": [{"$type": "ref", "value": "Coil1"}],
                    "points": {"data": []},
                },
            },
        }

        field = VacuumCoilField.from_json(write_json(data))
        B = field.B_xyz([0.0, 0.0, 0.0])
        expected_Bz = 4.0 * np.pi * 1.0e-7 * current / (2.0 * radius)

        self.assertEqual(field.metadata["format"], "simsopt")
        self.assertAlmostEqual(B[2], expected_Bz, delta=expected_Bz * 1.0e-4)

    def test_trace_field_line_returns_requested_steps(self):
        data = {
            "nfp": 1,
            "stellsym": False,
            "order": 1,
            "n_segments": 100,
            "dofs_curves": [
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ],
            "dofs_currents": [1.0e5],
        }
        field = VacuumCoilField.from_json(write_json(data))
        trace = field.trace_field_line([0.0, 0.0, 0.0], step_m=0.01, n_steps=5)

        self.assertEqual(trace.shape, (6, 3))
        self.assertGreater(trace[-1, 2], trace[0, 2])

    def test_trace_poincare_detects_toroidal_plane_hits(self):
        field = AnalyticToroidalField()

        hits = field.trace_poincare(
            [1.0, 0.0, 0.0],
            target_phi=np.pi/2.0,
            period=np.pi/2.0,
            n_hits=3,
            step_m=0.05,
            max_steps=200,
        )
        phi = np.arctan2(hits[:, 1], hits[:, 0])
        nearest = np.pi/2.0 + np.round((phi - np.pi/2.0)/(np.pi/2.0))*(np.pi/2.0)

        self.assertEqual(hits.shape, (3, 3))
        self.assertTrue(np.allclose(np.sqrt(hits[:, 0]**2 + hits[:, 1]**2), 1.0, atol=1.0e-3))
        self.assertTrue(np.allclose(phi - nearest, 0.0, atol=2.0e-3))


class AnalyticToroidalField(VacuumCoilField):
    def __init__(self):
        pass

    def B_xyz(self, points, chunk_size=256):
        pts = np.asarray(points, dtype=float)
        single = pts.ndim == 1
        pts = np.atleast_2d(pts)
        R = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        B = np.zeros_like(pts)
        B[:, 0] = -pts[:, 1] / R
        B[:, 1] = pts[:, 0] / R
        if single:
            return B[0]
        return B


if __name__ == "__main__":
    unittest.main()
