#vmecEquilibriumClass.py
#Description: VMEC wout reader for stellarator metadata and surfaces

import os
import numpy as np

try:
    from netCDF4 import Dataset
except Exception:
    Dataset = None


class VmecEquilibrium:
    """
    Lightweight VMEC ``wout`` reader for stellarator flux-surface metadata.

    The class intentionally focuses on geometry and profiles that HEAT needs
    before a full 3D equilibrium/field backend exists:

    * toroidal/poloidal flux profiles;
    * pressure, rotational transform, and net plasma current;
    * physical VMEC boundary / LCFS surfaces;
    * nearest-surface estimates of normalized toroidal flux ``s``.

    It does not evaluate the VMEC magnetic field.  In the current stellarator
    path, the companion coil JSON provides ``B(x,y,z)`` while this class
    provides the plasma-boundary and profile context inside the VMEC boundary.
    """

    kind = "vmec_equilibrium"

    def __init__(self, filename, lcfs_samples=361):
        if Dataset is None:
            raise ImportError("netCDF4 is required to read VMEC wout files")
        self.source_file = filename
        self._surface_grid_cache = {}

        with Dataset(filename) as ds:
            if not _is_vmec_dataset(ds):
                raise ValueError("Unsupported VMEC file. Expected a VMEC wout netCDF.")

            self.nfp = _scalar(ds, "nfp", int)
            self.ns = _scalar(ds, "ns", int)
            self.mpol = _scalar(ds, "mpol", int)
            self.ntor = _scalar(ds, "ntor", int)
            self.mnmax = _scalar(ds, "mnmax", int)
            self.xm = _array(ds, "xm")
            self.xn = _array(ds, "xn")
            self.rmnc = _array(ds, "rmnc")
            self.zmns = _array(ds, "zmns")

            self.phi = _array(ds, "phi")
            self.chi = _array(ds, "chi", default=np.zeros_like(self.phi))
            self.presf = _array(ds, "presf", default=np.zeros_like(self.phi))
            self.pres = _array(ds, "pres", default=np.zeros_like(self.phi))
            self.iotaf = _array(ds, "iotaf", default=np.zeros_like(self.phi))
            self.iotas = _array(ds, "iotas", default=np.zeros_like(self.phi))

            self.ctor = _scalar(ds, "ctor", float, default=0.0)
            self.b0 = _scalar(ds, "b0", float, default=0.0)
            self.rbtor = _scalar(ds, "rbtor", float, default=0.0)
            self.rbtor0 = _scalar(ds, "rbtor0", float, default=0.0)
            self.Rmajor_p = _scalar(ds, "Rmajor_p", float, default=np.nan)
            self.Aminor_p = _scalar(ds, "Aminor_p", float, default=np.nan)
            self.volume_p = _scalar(ds, "volume_p", float, default=np.nan)
            self.betatotal = _scalar(ds, "betatotal", float, default=np.nan)

        self.s_full = _normalized_profile_coordinate(self.phi)
        self.psiAxis = float(self.phi[0])
        self.psiSep = float(self.phi[-1])
        self.lcfs = np.column_stack(self.surface_cyl(
            1.0, np.linspace(0.0, 2.0*np.pi, lcfs_samples, endpoint=False), 0.0
        ))
        self.g = self._build_g_dict()

    @staticmethod
    def is_vmec_wout(filename):
        if Dataset is None or not os.path.exists(filename):
            return False
        try:
            with Dataset(filename) as ds:
                return _is_vmec_dataset(ds)
        except Exception:
            return False

    def _build_g_dict(self):
        edge_iota = float(self.iotaf[-1]) if len(self.iotaf) else np.nan
        pressure_axis = float(np.nanmax(self.presf)) if len(self.presf) else 0.0
        bt0 = self.b0
        if bt0 == 0.0 and self.rbtor0 != 0.0 and np.isfinite(self.Rmajor_p) and self.Rmajor_p != 0.0:
            bt0 = self.rbtor0 / self.Rmajor_p
        return {
            "psiAxis": self.psiAxis,
            "psiSep": self.psiSep,
            "Ip": self.ctor,
            "Bt0": bt0,
            "R0": self.Rmajor_p,
            "a": self.Aminor_p,
            "Nlcfs": len(self.lcfs),
            "lcfs": self.lcfs,
            "wall": self.lcfs,
            "nfp": self.nfp,
            "edge_iota": edge_iota,
            "pressure_axis": pressure_axis,
            "volume": self.volume_p,
            "betatotal": self.betatotal,
            "phi": self.phi,
            "chi": self.chi,
            "presf": self.presf,
            "pres": self.pres,
            "iotaf": self.iotaf,
            "iotas": self.iotas,
            "s": self.s_full,
        }

    def toroidal_flux(self, s):
        return np.interp(s, self.s_full, self.phi)

    def poloidal_flux(self, s):
        return np.interp(s, self.s_full, self.chi)

    def pressure(self, s, half_mesh=False):
        values = self.pres if half_mesh else self.presf
        return np.interp(s, self.s_full, values)

    def iota(self, s, half_mesh=False):
        values = self.iotas if half_mesh else self.iotaf
        return np.interp(s, self.s_full, values)

    def surface_cyl(self, s, theta, phi):
        """
        Return cylindrical ``R, Z`` on VMEC surfaces.

        ``s`` is normalized toroidal flux, and ``theta``/``phi`` are VMEC
        poloidal/toroidal angles in radians.  VMEC's ``xn`` values already
        include field-period multiplication, so the phase is ``m*theta-n*phi``.
        """
        s_arr, theta_arr, phi_arr = np.broadcast_arrays(
            np.asarray(s, dtype=float),
            np.asarray(theta, dtype=float),
            np.asarray(phi, dtype=float),
        )
        flat_s = np.clip(s_arr.ravel(), 0.0, 1.0)
        flat_theta = theta_arr.ravel()
        flat_phi = phi_arr.ravel()
        r_modes = self._interp_modes(self.rmnc, flat_s)
        z_modes = self._interp_modes(self.zmns, flat_s)
        angle = flat_theta[:, None]*self.xm[None, :] - flat_phi[:, None]*self.xn[None, :]
        R = np.sum(r_modes*np.cos(angle), axis=1)
        Z = np.sum(z_modes*np.sin(angle), axis=1)
        return R.reshape(s_arr.shape), Z.reshape(s_arr.shape)

    def surface_xyz(self, s, theta, phi):
        R, Z = self.surface_cyl(s, theta, phi)
        X = R*np.cos(phi)
        Y = R*np.sin(phi)
        return np.stack((X, Y, Z), axis=-1)

    def boundary_cyl(self, phi=0.0, ntheta=361):
        theta = np.linspace(0.0, 2.0*np.pi, ntheta, endpoint=False)
        return self.surface_cyl(np.ones_like(theta), theta, phi)

    def estimate_flux_label(self, R, Z, phi=0.0, s_samples=101,
                            theta_samples=361):
        """
        Estimate VMEC normalized toroidal flux by nearest surface grid point.

        This is a robust diagnostic/integration helper, not a high-order VMEC
        coordinate inversion.  It is good enough for Poincare surface checks and
        preliminary HEAT point-cloud labeling.
        """
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        single = R.ndim == 0
        R, Z = np.broadcast_arrays(R, Z)
        grid = self._surface_grid(float(np.mod(phi, 2.0*np.pi/self.nfp)),
                                  s_samples, theta_samples)
        flat_R = R.ravel()
        flat_Z = Z.ravel()
        s_est = np.zeros_like(flat_R)
        distance = np.zeros_like(flat_R)
        for i, (r, z) in enumerate(zip(flat_R, flat_Z)):
            idx = np.argmin((grid["R"] - r)**2 + (grid["Z"] - z)**2)
            s_est[i] = grid["s"][idx]
            distance[i] = np.sqrt((grid["R"][idx] - r)**2 + (grid["Z"][idx] - z)**2)
        if single:
            return float(s_est[0]), float(distance[0])
        return s_est.reshape(R.shape), distance.reshape(R.shape)

    def estimate_flux_label_xyz(self, xyz, s_samples=101, theta_samples=361):
        xyz = np.asarray(xyz, dtype=float)
        single = xyz.ndim == 1
        pts = np.atleast_2d(xyz)
        R = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        Z = pts[:, 2]
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        s = np.zeros(len(pts))
        d = np.zeros(len(pts))
        for i in range(len(pts)):
            s[i], d[i] = self.estimate_flux_label(
                R[i], Z[i], phi[i],
                s_samples=s_samples,
                theta_samples=theta_samples,
            )
        if single:
            return float(s[0]), float(d[0])
        return s, d

    def fieldline_surface_drift(self, field, s_values, theta0=0.0, phi0=0.0,
                                n_hits=8, step_m=0.02, max_steps=6000,
                                s_samples=201, theta_samples=721):
        """
        Trace coil-field Poincare hits and compare them to VMEC flux surfaces.

        Returns one dictionary per starting surface with the estimated ``s`` for
        each hit and summary drift metrics.
        """
        results = []
        for s0 in np.atleast_1d(s_values).astype(float):
            start = self.surface_xyz(s0, theta0, phi0).reshape(3)
            hits = field.trace_poincare(
                start,
                target_phi=phi0,
                period=2.0*np.pi/self.nfp,
                n_hits=n_hits,
                step_m=step_m,
                max_steps=max_steps,
            )
            if len(hits) == 0:
                s_hit = np.array([])
                distance = np.array([])
            else:
                s_hit, distance = self.estimate_flux_label_xyz(
                    hits,
                    s_samples=s_samples,
                    theta_samples=theta_samples,
                )
                s_hit = np.asarray(s_hit)
                distance = np.asarray(distance)
            drift = np.abs(s_hit - s0)
            results.append({
                "s0": s0,
                "start_xyz": start,
                "hits_xyz": hits,
                "s_hits": s_hit,
                "surface_distance_m": distance,
                "mean_abs_s_drift": float(np.mean(drift)) if len(drift) else np.nan,
                "max_abs_s_drift": float(np.max(drift)) if len(drift) else np.nan,
                "max_surface_distance_m": float(np.max(distance)) if len(distance) else np.nan,
            })
        return results

    def _interp_modes(self, modes, s_flat):
        out = np.empty((len(s_flat), modes.shape[1]), dtype=float)
        for i in range(modes.shape[1]):
            out[:, i] = np.interp(s_flat, self.s_full, modes[:, i])
        return out

    def _surface_grid(self, phi, s_samples, theta_samples):
        key = (round(phi, 12), int(s_samples), int(theta_samples))
        if key not in self._surface_grid_cache:
            s = np.linspace(0.0, 1.0, int(s_samples))
            theta = np.linspace(0.0, 2.0*np.pi, int(theta_samples), endpoint=False)
            ss, tt = np.meshgrid(s, theta, indexing="ij")
            R, Z = self.surface_cyl(ss, tt, phi)
            self._surface_grid_cache[key] = {
                "R": R.ravel(),
                "Z": Z.ravel(),
                "s": ss.ravel(),
            }
        return self._surface_grid_cache[key]


def _is_vmec_dataset(ds):
    required = {"rmnc", "zmns", "xm", "xn", "phi", "nfp", "ns"}
    return required.issubset(set(ds.variables.keys()))


def _scalar(ds, name, dtype=float, default=None):
    if name not in ds.variables:
        if default is None:
            raise KeyError("VMEC variable missing: " + name)
        return default
    value = np.asarray(ds.variables[name][:]).reshape(())
    return dtype(value.item())


def _array(ds, name, default=None):
    if name not in ds.variables:
        if default is None:
            raise KeyError("VMEC variable missing: " + name)
        return np.asarray(default, dtype=float)
    return np.asarray(ds.variables[name][:], dtype=float)


def _normalized_profile_coordinate(phi):
    phi = np.asarray(phi, dtype=float)
    span = phi[-1] - phi[0]
    if span == 0.0:
        return np.linspace(0.0, 1.0, len(phi))
    return (phi - phi[0]) / span
