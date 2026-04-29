#vacuumFieldClass.py
#Description: Vacuum coil-field readers and Biot-Savart evaluator

import json
import numpy as np


MU0_OVER_4PI = 1.0e-7


class VacuumCoilField:
    """
    Dependency-light Biot-Savart field evaluator for Fourier coil JSON files.

    Supported inputs:
      - ESSOS coil JSON with ``dofs_curves`` and ``dofs_currents``.
      - SIMSOPT ``BiotSavart`` JSON using ``CurveXYZFourier`` coils.

    Coordinates are SI Cartesian meters and Tesla.  The field is evaluated from
    the coils only, so it is valid on either side of any user-defined LCFS.  It
    does not provide flux coordinates or plasma current.
    """

    kind = "vacuum_coil_field"

    def __init__(self, gamma, gamma_dash, currents, source_file=None, metadata=None,
                 singularity_tol=1.0e-10, vmec=None):
        self.gamma = np.asarray(gamma, dtype=float)
        self.gamma_dash = np.asarray(gamma_dash, dtype=float)
        self.currents = np.asarray(currents, dtype=float)
        self.source_file = source_file
        self.metadata = metadata or {}
        self.singularity_tol = singularity_tol
        self.vmec = vmec
        if vmec is not None:
            self.g = vmec.g

        if self.gamma.ndim != 3 or self.gamma.shape[-1] != 3:
            raise ValueError("gamma must have shape (n_coils, n_segments, 3)")
        if self.gamma_dash.shape != self.gamma.shape:
            raise ValueError("gamma_dash must match gamma shape")
        if self.currents.shape[0] != self.gamma.shape[0]:
            raise ValueError("currents length must match number of coils")

    @classmethod
    def from_json(cls, filename, vmec_file=None):
        with open(filename, "r") as f:
            data = json.load(f)

        if _is_essos_json(data):
            gamma, gamma_dash, currents, metadata = _load_essos_json(data)
        elif _is_simsopt_json(data):
            gamma, gamma_dash, currents, metadata = _load_simsopt_json(data)
        else:
            raise ValueError(
                "Unsupported vacuum coil JSON. Expected ESSOS dofs_curves/"
                "dofs_currents or SIMSOPT BiotSavart JSON."
            )
        vmec = None
        if vmec_file is not None:
            from vmecEquilibriumClass import VmecEquilibrium
            vmec = VmecEquilibrium(vmec_file)
            metadata["vmec_file"] = vmec_file
        return cls(gamma, gamma_dash, currents, source_file=filename,
                   metadata=metadata, vmec=vmec)

    @staticmethod
    def is_coil_json(filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
        except Exception:
            return False
        return _is_essos_json(data) or _is_simsopt_json(data)

    @property
    def n_coils(self):
        return self.gamma.shape[0]

    @property
    def n_segments(self):
        return self.gamma.shape[1]

    def B_xyz(self, points, chunk_size=256):
        """
        Evaluate B at Cartesian points.

        Parameters
        ----------
        points : array_like
            Shape ``(3,)`` or ``(n, 3)`` in meters.
        chunk_size : int
            Number of evaluation points processed at once.
        """
        pts = np.asarray(points, dtype=float)
        single = pts.ndim == 1
        pts = np.atleast_2d(pts)

        out = np.zeros((pts.shape[0], 3), dtype=float)
        coeff = self.currents[:, None, None] * MU0_OVER_4PI

        for lo in range(0, pts.shape[0], chunk_size):
            hi = min(lo + chunk_size, pts.shape[0])
            diff = pts[lo:hi, None, None, :] - self.gamma[None, :, :, :]
            norm = np.linalg.norm(diff, axis=-1)
            norm = np.maximum(norm, self.singularity_tol)
            dB = np.cross(self.gamma_dash[None, :, :, :], diff) / norm[..., None]**3
            out[lo:hi] = np.mean(coeff[None, :, :, :] * dB, axis=2).sum(axis=1)

        if single:
            return out[0]
        return out

    def B_cyl(self, R, Z, phi):
        """Evaluate B and return cylindrical components ``BR, Bt, BZ``."""
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        phi = np.asarray(phi, dtype=float)
        R, Z, phi = np.broadcast_arrays(R, Z, phi)
        pts = np.column_stack((R.ravel() * np.cos(phi.ravel()),
                               R.ravel() * np.sin(phi.ravel()),
                               Z.ravel()))
        Bxyz = self.B_xyz(pts)
        c = np.cos(phi.ravel())
        s = np.sin(phi.ravel())
        BR = Bxyz[:, 0] * c + Bxyz[:, 1] * s
        Bt = -Bxyz[:, 0] * s + Bxyz[:, 1] * c
        BZ = Bxyz[:, 2]
        return BR.reshape(R.shape), Bt.reshape(R.shape), BZ.reshape(R.shape)

    def trace_field_line(self, start_xyz, step_m=0.01, n_steps=1000,
                         direction=1.0):
        """
        Trace a vacuum field line with fixed arclength steps using RK4.

        This is a lightweight diagnostic tracer.  It does not do wall
        intersections, connection lengths, or adaptive step-size control.
        """
        xyz = np.asarray(start_xyz, dtype=float)
        trace = np.zeros((n_steps + 1, 3), dtype=float)
        trace[0] = xyz
        h = float(step_m) * np.sign(direction)

        for i in range(n_steps):
            trace[i + 1] = self._rk4_step(trace[i], h)
        return trace

    def trace_poincare(self, start_xyz, target_phi=0.0, period=2.0*np.pi,
                       n_hits=10, step_m=0.02, max_steps=100000,
                       direction=1.0):
        """
        Trace a field line and return intersections with toroidal planes.

        ``target_phi`` is the toroidal plane in radians.  ``period`` can be set
        to ``2*pi/nfp`` for stellarator field-period symmetry.  The returned
        array has shape ``(n_hits, 3)`` in Cartesian meters.
        """
        y = np.asarray(start_xyz, dtype=float)
        h = float(step_m) * np.sign(direction)
        target_phi = float(target_phi)
        period = float(period)
        hits = []
        prev_phi = np.arctan2(y[1], y[0])
        initial_plane = _nearest_plane(prev_phi, target_phi, period)

        for _ in range(int(max_steps)):
            y_next = self._rk4_step(y, h)
            phi_next = _unwrap_near(np.arctan2(y_next[1], y_next[0]), prev_phi)
            crossings = _plane_crossings(prev_phi, phi_next, target_phi, period)
            for plane in crossings:
                if len(hits) == 0 and abs(plane - initial_plane) < 1.0e-12:
                    continue
                alpha = (plane - prev_phi) / (phi_next - prev_phi)
                if 1.0e-12 <= alpha <= 1.0:
                    hits.append(y + alpha*(y_next - y))
                    if len(hits) >= n_hits:
                        return np.asarray(hits)
            y = y_next
            prev_phi = phi_next
        return np.asarray(hits)

    def estimate_flux_label_xyz(self, xyz, s_samples=101, theta_samples=361):
        if self.vmec is None:
            raise ValueError("No companion VMEC equilibrium is attached to this field")
        return self.vmec.estimate_flux_label_xyz(
            xyz,
            s_samples=s_samples,
            theta_samples=theta_samples,
        )

    def fieldline_surface_drift(self, s_values, **kwargs):
        if self.vmec is None:
            raise ValueError("No companion VMEC equilibrium is attached to this field")
        return self.vmec.fieldline_surface_drift(self, s_values, **kwargs)

    def _rk4_step(self, y, h):
        k1 = self._bhat(y)
        k2 = self._bhat(y + 0.5 * h * k1)
        k3 = self._bhat(y + 0.5 * h * k2)
        k4 = self._bhat(y + h * k3)
        return y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def _bhat(self, xyz):
        B = self.B_xyz(xyz)
        norm = np.linalg.norm(B)
        if norm == 0.0:
            raise ValueError("Cannot trace field line through zero magnetic field")
        return B / norm


def _is_essos_json(data):
    return all(k in data for k in ("dofs_curves", "dofs_currents", "nfp",
                                   "stellsym", "n_segments"))


def _is_simsopt_json(data):
    if data.get("@class") != "SIMSON":
        return False
    objs = data.get("simsopt_objs", {})
    return any(obj.get("@class") == "BiotSavart" for obj in objs.values())


def _load_essos_json(data):
    base_dofs = np.asarray(data["dofs_curves"], dtype=float)
    base_currents = np.asarray(data["dofs_currents"], dtype=float)
    n_segments = int(data["n_segments"])
    nfp = int(data["nfp"])
    stellsym = bool(data["stellsym"])

    gamma = []
    gamma_dash = []
    currents = []
    for k in range(nfp):
        phi = 2.0 * np.pi * k / nfp
        for flip in ([False, True] if stellsym else [False]):
            for dofs, current in zip(base_dofs, base_currents):
                g, gd = _fourier_curve(dofs, n_segments=n_segments)
                gamma.append(_rotate_points(g, phi, flip))
                gamma_dash.append(_rotate_points(gd, phi, flip))
                currents.append(-current if flip else current)

    metadata = {
        "format": "essos",
        "nfp": nfp,
        "stellsym": stellsym,
        "order": int(data.get("order", base_dofs.shape[-1] // 2)),
    }
    return np.asarray(gamma), np.asarray(gamma_dash), np.asarray(currents), metadata


def _load_simsopt_json(data):
    objs = data["simsopt_objs"]
    biot_savart = next(obj for obj in objs.values()
                       if obj.get("@class") == "BiotSavart")
    coil_refs = [_ref_value(ref) for ref in biot_savart["coils"]]

    gamma = []
    gamma_dash = []
    currents = []
    for coil_ref in coil_refs:
        coil = objs[coil_ref]
        g, gd = _simsopt_curve(objs, _ref_value(coil["curve"]))
        gamma.append(g)
        gamma_dash.append(gd)
        currents.append(_simsopt_current(objs, _ref_value(coil["current"])))

    metadata = {
        "format": "simsopt",
        "version": data.get("@version"),
    }
    return np.asarray(gamma), np.asarray(gamma_dash), np.asarray(currents), metadata


def _simsopt_curve(objs, name):
    obj = objs[name]
    cls = obj.get("@class")
    if cls == "CurveXYZFourier":
        order = int(obj["order"])
        x = np.asarray(objs[_ref_value(obj["dofs"])]["x"]["data"], dtype=float)
        dofs = x.reshape((3, 2 * order + 1))
        quadpoints = np.asarray(obj["quadpoints"]["data"], dtype=float)
        return _fourier_curve(dofs, quadpoints=quadpoints)
    if cls == "RotatedCurve":
        g, gd = _simsopt_curve(objs, _ref_value(obj["curve"]))
        phi = float(obj["phi"])
        flip = bool(obj["flip"])
        return _rotate_points(g, phi, flip), _rotate_points(gd, phi, flip)
    raise ValueError("Unsupported SIMSOPT curve class: " + str(cls))


def _simsopt_current(objs, name):
    obj = objs[name]
    cls = obj.get("@class")
    if cls == "Current":
        return float(obj["current"])
    if cls == "ScaledCurrent":
        return float(obj["scale"]) * _simsopt_current(objs, _ref_value(obj["current_to_scale"]))
    raise ValueError("Unsupported SIMSOPT current class: " + str(cls))


def _ref_value(ref):
    if not isinstance(ref, dict) or ref.get("$type") != "ref":
        raise ValueError("Expected SIMSOPT reference object")
    return ref["value"]


def _fourier_curve(dofs, n_segments=None, quadpoints=None):
    dofs = np.asarray(dofs, dtype=float)
    if dofs.shape[0] != 3 or dofs.shape[1] % 2 != 1:
        raise ValueError("Fourier dofs must have shape (3, 2*order+1)")
    if quadpoints is None:
        if n_segments is None:
            raise ValueError("n_segments or quadpoints must be provided")
        quadpoints = np.linspace(0.0, 1.0, int(n_segments), endpoint=False)
    else:
        quadpoints = np.asarray(quadpoints, dtype=float)

    order = dofs.shape[1] // 2
    gamma = np.einsum("i,k->ki", dofs[:, 0], np.ones_like(quadpoints))
    gamma_dash = np.zeros_like(gamma)
    for k in range(1, order + 1):
        sin = np.sin(2.0 * np.pi * k * quadpoints)
        cos = np.cos(2.0 * np.pi * k * quadpoints)
        gamma += np.einsum("i,k->ki", dofs[:, 2 * k - 1], sin)
        gamma += np.einsum("i,k->ki", dofs[:, 2 * k], cos)
        gamma_dash += np.einsum("i,k->ki", dofs[:, 2 * k - 1],
                                2.0 * np.pi * k * cos)
        gamma_dash += np.einsum("i,k->ki", dofs[:, 2 * k],
                                -2.0 * np.pi * k * sin)
    return gamma, gamma_dash


def _rotate_points(points, phi, flip):
    rotmat = np.array(
        [[np.cos(phi), -np.sin(phi), 0.0],
         [np.sin(phi),  np.cos(phi), 0.0],
         [0.0,          0.0,         1.0]]
    ).T
    if flip:
        rotmat = rotmat @ np.array(
            [[1.0,  0.0,  0.0],
             [0.0, -1.0,  0.0],
             [0.0,  0.0, -1.0]]
        )
    return np.asarray(points) @ rotmat


def _unwrap_near(phi, reference):
    while phi - reference > np.pi:
        phi -= 2.0*np.pi
    while phi - reference < -np.pi:
        phi += 2.0*np.pi
    return phi


def _nearest_plane(phi, target_phi, period):
    return target_phi + np.round((phi - target_phi) / period) * period


def _plane_crossings(phi0, phi1, target_phi, period):
    if phi0 == phi1:
        return []
    lo = min(phi0, phi1)
    hi = max(phi0, phi1)
    k0 = int(np.ceil((lo - target_phi) / period))
    k1 = int(np.floor((hi - target_phi) / period))
    crossings = []
    for k in range(k0, k1 + 1):
        plane = target_phi + k*period
        if lo <= plane <= hi:
            crossings.append(plane)
    if phi1 < phi0:
        crossings.reverse()
    return crossings
