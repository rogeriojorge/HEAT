#stellaratorTraceClass.py
#Description: Stellarator field-line hit and reduced heat-load utilities

import csv
from dataclasses import dataclass
import numpy as np


@dataclass
class CylindricalTarget:
    """Simple limiter/PFC proxy at fixed major radius."""

    radius: float
    z_min: float
    z_max: float
    name: str = "cylindrical_limiter"

    def signed_distance(self, xyz):
        xyz = np.asarray(xyz, dtype=float)
        return np.sqrt(xyz[..., 0]**2 + xyz[..., 1]**2) - self.radius

    def normal_at(self, xyz):
        xyz = np.asarray(xyz, dtype=float)
        R = np.sqrt(xyz[..., 0]**2 + xyz[..., 1]**2)
        if np.any(R == 0.0):
            raise ValueError("Cannot compute cylindrical normal on the axis")
        normal = np.zeros_like(xyz, dtype=float)
        normal[..., 0] = xyz[..., 0] / R
        normal[..., 1] = xyz[..., 1] / R
        return normal

    def phi_z(self, xyz):
        xyz = np.asarray(xyz, dtype=float)
        phi = np.mod(np.arctan2(xyz[..., 1], xyz[..., 0]), 2.0*np.pi)
        return phi, xyz[..., 2]

    def cell_areas(self, phi_edges, z_edges):
        dphi = np.diff(phi_edges)
        dz = np.diff(z_edges)
        return self.radius * dphi[:, None] * dz[None, :]

    def intersect_segment(self, p0, p1, tol=1.0e-12):
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        d0 = self.signed_distance(p0)
        d1 = self.signed_distance(p1)
        denom = d1 - d0
        if abs(denom) < tol:
            return None
        alpha = -d0 / denom
        if alpha <= tol or alpha > 1.0:
            return None
        hit = p0 + alpha*(p1 - p0)
        if hit[2] < self.z_min or hit[2] > self.z_max:
            return None
        return hit, float(alpha)


def trace_to_target(field, start_xyz, target, step_m=0.02, max_steps=5000,
                    direction=1.0, store_path=False):
    """
    Trace one field line until it intersects a target.

    Returns a dictionary with ``hit`` status, hit location, connection length,
    incidence, and optionally the traced path.
    """
    h = float(step_m) * np.sign(direction)
    if h == 0.0:
        raise ValueError("direction cannot be zero")
    y = np.asarray(start_xyz, dtype=float)
    length = 0.0
    path = [y.copy()] if store_path else None

    for step in range(int(max_steps)):
        y_next = _rk4_step(field, y, h)
        segment = y_next - y
        segment_length = float(np.linalg.norm(segment))
        intersection = target.intersect_segment(y, y_next)
        if intersection is not None:
            hit, alpha = intersection
            hit_length = length + alpha*segment_length
            B = np.asarray(field.B_xyz(hit), dtype=float)
            Bnorm = np.linalg.norm(B)
            if Bnorm == 0.0:
                raise ValueError("Cannot compute incidence at zero magnetic field")
            bhat = B / Bnorm
            normal = target.normal_at(hit)
            bdotn = float(np.dot(bhat, normal))
            phi, z = target.phi_z(hit)
            if store_path:
                path.append(hit.copy())
            return {
                "hit": True,
                "hit_xyz": hit,
                "connection_length_m": hit_length,
                "steps": step + 1,
                "bdotn": bdotn,
                "abs_bdotn": abs(bdotn),
                "strike_angle_deg": float(np.degrees(np.arcsin(np.clip(abs(bdotn), 0.0, 1.0)))),
                "phi": float(phi),
                "z": float(z),
                "path_xyz": np.asarray(path) if store_path else None,
            }
        y = y_next
        length += segment_length
        if store_path:
            path.append(y.copy())

    return {
        "hit": False,
        "hit_xyz": np.full(3, np.nan),
        "connection_length_m": length,
        "steps": int(max_steps),
        "bdotn": np.nan,
        "abs_bdotn": np.nan,
        "strike_angle_deg": np.nan,
        "phi": np.nan,
        "z": np.nan,
        "path_xyz": np.asarray(path) if store_path else None,
    }


def make_vmec_launch_points(vmec, n_theta=12, n_phi=12, offsets_m=None,
                            lambda_m=0.02, s=1.0):
    """
    Create deterministic launch markers near a VMEC surface.

    Markers start on surface ``s`` and are displaced along the local minor-radial
    vector by each offset.  The weight is ``exp(-offset/lambda_m)``.
    """
    if offsets_m is None:
        offsets_m = [0.01, 0.02, 0.03]
    launches = []
    theta_values = np.linspace(0.0, 2.0*np.pi, int(n_theta), endpoint=False)
    phi_values = np.linspace(0.0, 2.0*np.pi, int(n_phi), endpoint=False)
    for phi in phi_values:
        axis = vmec.surface_xyz(0.0, 0.0, phi).reshape(3)
        for theta in theta_values:
            base = vmec.surface_xyz(s, theta, phi).reshape(3)
            outward = base - axis
            norm = np.linalg.norm(outward)
            if norm == 0.0:
                continue
            outward /= norm
            for offset in offsets_m:
                launches.append({
                    "start_xyz": base + float(offset)*outward,
                    "base_xyz": base,
                    "theta": float(theta),
                    "phi": float(phi),
                    "offset_m": float(offset),
                    "weight": float(np.exp(-float(offset)/float(lambda_m))),
                    "s": float(s),
                })
    return launches


def run_limiter_scan(field, target, launches, directions=(-1.0, 1.0),
                     step_m=0.02, max_steps=5000, store_paths=12,
                     estimate_vmec=True, s_samples=51, theta_samples=181):
    """Trace launch markers in one or more directions to a limiter target."""
    rows = []
    paths = []
    path_budget = int(store_paths)
    vmec = getattr(field, "vmec", None)
    for marker_id, launch in enumerate(launches):
        for direction in directions:
            store_path = len(paths) < path_budget
            trace = trace_to_target(
                field,
                launch["start_xyz"],
                target,
                step_m=step_m,
                max_steps=max_steps,
                direction=direction,
                store_path=store_path,
            )
            row = {
                "marker_id": marker_id,
                "direction": float(direction),
                "theta": launch["theta"],
                "launch_phi": launch["phi"],
                "offset_m": launch["offset_m"],
                "weight": launch["weight"],
                "s_launch": launch["s"],
                "hit": bool(trace["hit"]),
                "connection_length_m": trace["connection_length_m"],
                "steps": trace["steps"],
                "bdotn": trace["bdotn"],
                "abs_bdotn": trace["abs_bdotn"],
                "strike_angle_deg": trace["strike_angle_deg"],
                "hit_phi": trace["phi"],
                "hit_z": trace["z"],
                "hit_x": trace["hit_xyz"][0],
                "hit_y": trace["hit_xyz"][1],
                "hit_z_cart": trace["hit_xyz"][2],
                "launch_x": launch["start_xyz"][0],
                "launch_y": launch["start_xyz"][1],
                "launch_z": launch["start_xyz"][2],
            }
            if trace["hit"] and vmec is not None and estimate_vmec:
                s_near, d_near = vmec.estimate_flux_label_xyz(
                    trace["hit_xyz"],
                    s_samples=s_samples,
                    theta_samples=theta_samples,
                )
                row["nearest_vmec_s"] = s_near
                row["distance_to_vmec_m"] = d_near
            else:
                row["nearest_vmec_s"] = np.nan
                row["distance_to_vmec_m"] = np.nan
            rows.append(row)
            if store_path:
                paths.append({
                    "marker_id": marker_id,
                    "direction": float(direction),
                    "hit": bool(trace["hit"]),
                    "path_xyz": trace["path_xyz"],
                })
    return rows, paths


def bin_limiter_heat_load(rows, target, total_power_w=1.0e6, n_phi=72,
                          n_z=80, z_range=None):
    """Bin successful limiter hits into perpendicular heat flux on a cylinder."""
    hits = [r for r in rows if r["hit"]]
    phi_edges = np.linspace(0.0, 2.0*np.pi, int(n_phi) + 1)
    if z_range is None:
        z_edges = np.linspace(target.z_min, target.z_max, int(n_z) + 1)
    else:
        z_edges = np.linspace(float(z_range[0]), float(z_range[1]), int(n_z) + 1)
    power = np.zeros((int(n_phi), int(n_z)), dtype=float)
    if not hits:
        return {
            "phi_edges": phi_edges,
            "z_edges": z_edges,
            "power_w": power,
            "q_perp_w_m2": power.copy(),
            "area_m2": target.cell_areas(phi_edges, z_edges),
            "target_radius_m": target.radius,
            "marker_power_w": np.array([]),
        }

    weights = np.asarray([h["weight"] for h in hits], dtype=float)
    marker_power = total_power_w * weights / weights.sum()
    for h, p in zip(hits, marker_power):
        i = np.searchsorted(phi_edges, h["hit_phi"], side="right") - 1
        j = np.searchsorted(z_edges, h["hit_z"], side="right") - 1
        i %= int(n_phi)
        if 0 <= j < int(n_z):
            power[i, j] += p
    area = target.cell_areas(phi_edges, z_edges)
    q = np.divide(power, area, out=np.zeros_like(power), where=area > 0.0)
    return {
        "phi_edges": phi_edges,
        "z_edges": z_edges,
        "power_w": power,
        "q_perp_w_m2": q,
        "area_m2": area,
        "target_radius_m": target.radius,
        "marker_power_w": marker_power,
    }


def summarize_limiter_result(rows, heat_grid, total_power_w=1.0e6,
                             wetted_fraction=0.01):
    hits = [r for r in rows if r["hit"]]
    q = heat_grid["q_perp_w_m2"]
    power = heat_grid["power_w"]
    if len(hits) == 0:
        return {
            "n_traces": len(rows),
            "n_hits": 0,
            "hit_fraction": 0.0,
            "total_power_w": total_power_w,
            "mapped_power_w": 0.0,
            "peak_q_mw_m2": 0.0,
            "wetted_area_m2": 0.0,
        }
    threshold = wetted_fraction * np.max(q)
    area = heat_grid["area_m2"]
    wetted_area = float(np.sum(area[q >= threshold]))
    lc = np.asarray([h["connection_length_m"] for h in hits])
    bdotn = np.asarray([h["abs_bdotn"] for h in hits])
    return {
        "n_traces": len(rows),
        "n_hits": len(hits),
        "hit_fraction": len(hits) / len(rows),
        "total_power_w": total_power_w,
        "mapped_power_w": float(np.sum(power)),
        "peak_q_mw_m2": float(np.max(q) / 1.0e6),
        "wetted_area_m2": wetted_area,
        "median_connection_length_m": float(np.median(lc)),
        "p95_connection_length_m": float(np.percentile(lc, 95)),
        "median_abs_bdotn": float(np.median(bdotn)),
        "median_strike_angle_deg": float(np.degrees(np.arcsin(np.clip(np.median(bdotn), 0.0, 1.0)))),
        "wetted_threshold_fraction": wetted_fraction,
        "target_radius_m": float(heat_grid["target_radius_m"]),
    }


def write_rows_csv(filename, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _rk4_step(field, y, h):
    k1 = _bhat(field, y)
    k2 = _bhat(field, y + 0.5*h*k1)
    k3 = _bhat(field, y + 0.5*h*k2)
    k4 = _bhat(field, y + h*k3)
    return y + h*(k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0


def _bhat(field, xyz):
    B = np.asarray(field.B_xyz(xyz), dtype=float)
    norm = np.linalg.norm(B)
    if norm == 0.0:
        raise ValueError("Cannot trace field line through zero magnetic field")
    return B / norm
