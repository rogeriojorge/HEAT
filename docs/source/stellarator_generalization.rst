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
