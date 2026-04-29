Stellarator Generalization
##########################

This page tracks the early design direction for generalizing HEAT from primarily
tokamak workflows to stellarator and other nonaxisymmetric configurations.

Motivation
**********

HEAT already connects CAD geometry, magnetic field-line tracing, heat-flux
models, thermal solvers, and visualization. The main stellarator challenge is
not the PFC geometry itself, but the assumptions used to define heat-flux
coordinates and source models. Existing tokamak workflows often rely on EFIT,
outboard-midplane mapping, normalized poloidal flux, and Eich-style scrape-off
layer scalings.

For stellarators, important quantities include connection length, island
topology, strike angle, 3D wetted area, cross-field transport, and imported
loads from tools such as EMC3-EIRENE, EMC3-Lite, M3D-C1, NIMROD, VMEC-based
workflows, or field-line diffusion codes.

Initial Design Goals
********************

* Preserve existing tokamak behavior and input files.
* Make equilibrium and field-line tracing access explicit interfaces.
* Support imported target heat loads before assuming a universal stellarator
  scaling.
* Store nonaxisymmetric diagnostics such as connection length, strike angle,
  component-integrated power, peak heat flux, and wetted area.
* Use W7-X steady-state cases and EMC3-EIRENE workflows as the first validation
  target.
* Include HSX as a possible second validation target when benchmark data are
  available.

Stellarator Work Packages
*************************

The first stellarator development path should focus on concrete, testable work
packages:

* import nonaxisymmetric equilibrium or field data from sources such as VMEC,
  DESC, M3D-C1, NIMROD, VMEC-EXTENDER, VMEC-BMW, or HINT;
* verify that the field-line tracing path works for nonaxisymmetric stellarator
  geometry;
* define explicit upstream/source locations for island-divertor heat mapping,
  rather than assuming a tokamak outboard-midplane mapping;
* compute loads on preliminary PFCs produced from plasma-boundary extension;
* compute loads on realistic stellarator PFCs from CAD or mesh models;
* compare HEAT heat loads with EMC3-EIRENE benchmark simulations for W7-X and
  other stellarators where data are available;
* preserve enough metadata to support reproducible external workflows.

Existing HEAT 3D Baseline
*************************

Recent HEAT work already added a nonaxisymmetric tokamak path using M3D-C1
fields, MAFOT field-line tracing, and a 3D layer heat-flux model. That path is a
useful starting point for stellarators because it separates field-line footprints
from target heat-flux assignment. The stellarator work should keep that lesson,
while replacing tokamak-specific layer definitions with explicit source regions,
connection length, incidence angle, and imported or reduced stellarator heat-load
models.

Vacuum Coil Fields
******************

As an early stellarator path, HEAT can read ESSOS and SIMSOPT Biot-Savart coil
JSON files directly through ``source/vacuumFieldClass.py``. These files define
Fourier coils and currents, so HEAT can evaluate the vacuum magnetic field in
Cartesian coordinates on either side of any user-defined LCFS without first
generating an EFIT, VMEC extension, or other external field file.

This mode currently provides:

* direct ``B(x,y,z)`` evaluation from coil geometry;
* cylindrical component conversion for diagnostics;
* a lightweight fixed-step RK4 field-line tracer for vacuum-field experiments;
* Poincare-plane hit detection for coil-field surface checks;
* ``MHDClass`` detection of these files as ``coiljson`` inputs.

The coil field can also be paired with a companion VMEC ``wout`` file through
``source/vmecEquilibriumClass.py``. In that mode the coil JSON remains the
magnetic-field provider everywhere, while VMEC supplies normalized toroidal flux
``s``, pressure and iota profiles, plasma current, and the physical plasma
boundary/LCFS up to the VMEC edge. This is the current path for the
Landreman-Paul QA vacuum case using the ESSOS/SIMSOPT coil JSON and
``wout_LandremanPaul2021_QA_lowres.nc``.

The VMEC reader also provides a nearest-surface flux-label diagnostic and a
coil-field surface-preservation check. For the local Landreman-Paul QA files,
eight Poincare hits on ``s = 0.25, 0.50, 0.75, 1.00`` stayed within
``max_abs_s_drift <= 0.015`` and ``max_surface_distance_m <= 0.0011`` in the
current fixed-step diagnostic. This validates that the vacuum coils reproduce
the VMEC magnetic surfaces well enough for the next HEAT integration step.

This is still not a complete stellarator heat-flux model. Heat-flux models that
require an OMP mapping, connection length, target strike points, or a validated
stellarator scrape-off-layer source still need the next provider and mapper
layers.

Candidate Interfaces
********************

``EquilibriumProvider``
    Supplies magnetic-field access, coordinate labels, time indexing, and
    optional flux-surface metadata.

``FieldLineTracer``
    Launches and parses field-line traces, including connection length and
    target hits.

``HeatFluxModel``
    Converts a physics source model into parallel or target heat flux. Existing
    Eich and RZQ models become implementations rather than assumptions baked
    into the whole workflow.

``SurfaceLoadMapper``
    Maps imported or computed loads to HEAT PFC mesh faces and computes power
    balance diagnostics.

First Implementation Path
*************************

The lowest-risk first feature is an imported target-load workflow for
nonaxisymmetric cases. This lets HEAT consume outputs from high-fidelity or
reduced stellarator tools while the native reduced model is developed and
validated.

After that, a reduced field-line diffusion model can be prototyped and compared
against W7-X EMC3-EIRENE or EMC3-Lite cases before being promoted into the main
HEAT workflow.

Near-Term Patch Candidates
**************************

* Add regression tests around current tokamak heat-flux profiles before
  refactoring.
* Add an imported surface-load reader for ``x,y,z,q`` or ``face_id,q`` data.
* Add output fields for connection length and incidence angle where field-line
  tracing already computes enough information.
* Add a stellarator example skeleton with placeholder VMEC/field-line products
  and documented expected inputs, keeping large benchmark data outside git.
