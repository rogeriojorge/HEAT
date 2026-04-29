import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from vacuumFieldClass import VacuumCoilField
from vmecEquilibriumClass import VmecEquilibrium


VMEC_FILE = Path("~/local/simsopt/tests/test_files/wout_LandremanPaul2021_QA_lowres.nc").expanduser()
COIL_FILE = Path("~/local/ESSOS/examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json").expanduser()


def has_landreman_paul_files():
    return VMEC_FILE.exists() and COIL_FILE.exists() and VmecEquilibrium.is_vmec_wout(str(VMEC_FILE))


@unittest.skipUnless(has_landreman_paul_files(), "Landreman-Paul VMEC/coil files with netCDF4 are required")
class VmecEquilibriumTest(unittest.TestCase):
    def test_vmec_reader_exposes_profiles_and_boundary(self):
        vmec = VmecEquilibrium(str(VMEC_FILE))

        self.assertEqual(vmec.nfp, 2)
        self.assertGreater(vmec.g["psiSep"], vmec.g["psiAxis"])
        self.assertAlmostEqual(vmec.g["Ip"], 0.0, delta=1.0e-8)
        self.assertAlmostEqual(vmec.g["pressure_axis"], 0.0, delta=1.0e-12)
        self.assertEqual(vmec.g["Nlcfs"], len(vmec.g["lcfs"]))
        self.assertGreater(vmec.g["lcfs"][:, 0].max(), vmec.g["lcfs"][:, 0].min())

    def test_vmec_estimates_flux_label_near_constructed_surface_point(self):
        vmec = VmecEquilibrium(str(VMEC_FILE))
        xyz = vmec.surface_xyz(0.5, 1.0, 0.3).reshape(3)

        s, distance = vmec.estimate_flux_label_xyz(
            xyz,
            s_samples=121,
            theta_samples=361,
        )

        self.assertAlmostEqual(s, 0.5, delta=0.02)
        self.assertLess(distance, 3.0e-3)

    def test_coil_field_attaches_vmec_and_preserves_surfaces(self):
        field = VacuumCoilField.from_json(str(COIL_FILE), vmec_file=str(VMEC_FILE))

        self.assertIsNotNone(field.vmec)
        self.assertIn("psiSep", field.g)
        results = field.fieldline_surface_drift(
            [0.25, 0.75],
            n_hits=4,
            step_m=0.025,
            max_steps=3000,
            s_samples=121,
            theta_samples=361,
        )

        for result in results:
            self.assertEqual(len(result["s_hits"]), 4)
            self.assertLess(result["max_abs_s_drift"], 0.03)
            self.assertLess(result["max_surface_distance_m"], 3.0e-3)

    def test_mhd_terminal_helper_loads_coils_with_companion_vmec(self):
        import MHDClass

        mhd = MHDClass.setupForTerminalUse(
            gFile=str(COIL_FILE),
            shot=0,
            time=0.0,
            vmecFile=str(VMEC_FILE),
        )

        self.assertEqual(mhd.determineEQFiletype(str(COIL_FILE)), "coiljson")
        self.assertEqual(mhd.determineEQFiletype(str(VMEC_FILE)), "vmec")
        self.assertIsNotNone(mhd.ep.vmec)
        self.assertGreater(mhd.ep.g["Nlcfs"], 0)


if __name__ == "__main__":
    unittest.main()
