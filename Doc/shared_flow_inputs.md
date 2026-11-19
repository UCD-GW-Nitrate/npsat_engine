# Shared steady-flow inputs (migration step 1)

The preferred flow entry point is `npsat -c model.ini`. Boost.Program_options
parses typed, case-sensitive options and rejects unknown keys and duplicates.
`npsat --help-config` prints the complete schema and defaults. Use
`npsat -c model.ini --validate-only` to read and validate the configured data
without constructing a mesh or solving. `--print-config` prints a stable,
sorted option listing, resolved paths, raw selected source rates and sample
interpolated values before solving. Validation does not claim domain coverage,
mesh quality, well/cell intersection, output writability or solver convergence.

The old `-p model.prm` entry point is retained explicitly for regression and
existing tracing/gather workflows. The INI path does not convert values through
deal.II ParameterHandler: both readers use a common property initialization
adapter. See [the option mapping](flow_option_mapping.md) for all 75 legacy
fields. New `FEM.*` and `Tracing.*` options are v1-specific. Full configuration
files therefore differ between solvers; the physical data files are shared.

## Paths and configuration

`Paths.Main` is relative to the process working directory, matching v2.
`Paths.Input` and `Paths.Output` are relative to Main; absolute paths are also
accepted. All nested interpolation filenames are relative to the resolved
input directory, not their containing master file. Output directories must
already exist for a solve. The example README gives the `mkdir` command.

| Keys | Interpretation in v1 |
| --- | --- |
| `Geometry.Type`, `BoxDims`, `BoxLLP`, `MeshFileName`, `Top`, `Bottom` | BOX/FILE geometry; top/bottom reference interpolation masters |
| `Discretization.Nxyz` | Positive integer cell counts; the INI adapter generates Nz+1 vertical coordinates if Vertical is empty |
| `Discretization.Vertical` | Inline increasing list from 0 to 1; BOX requires Nz+1 entries. File-valued vertical distributions are not implemented in the INI adapter |
| `Refinement.Initial/Wells/Streams/Top` | Existing v1 refinement settings; other v1 refinement controls live under FEM |
| `Properties.Kxy`, `Properties.Kz` | Horizontal and vertical conductivity masters; absent Kz follows Kxy |
| `FEM.Ky`, `FEM.Porosity` | Optional distinct y conductivity; required porosity master. KY-only anisotropy uses Kxy for Kz |
| `Sources.Rch_Data`, `Rch_factor` | Recharge master and multiplier, applied once by existing assembly |
| `Sources.Well_Data/Well_Rates/Well_factor` | v2 well geometry, separate rates and multiplier |
| `Sources.Stream_Data/Stream_Rates/Stream_factor` | v2 stream geometry, separate rates and multiplier |
| `BC.Dirichlet`, `HalfWidth`, `MinOverlap` | v2 boundary grammar and lateral matching tolerances |
| `BC.Neumann` | Still v1's BOT/BOUNDARY_LINE grammar from `interpolation_v2.md`; this is not a completed v2 Neumann implementation |
| `BC.GHB` | Must be empty until milestone 2; a nonempty value fails explicitly |
| `Simulation.Steady_state` | Must be 1; no transient solve |
| `Simulation.Start_step` | Nonnegative zero-based stress column, applied consistently to loaded flow interpolation data and source rates |
| `Simulation.Confined` | 0/1; confined requires FEM.MeshIterations=1 to keep geometry fixed |
| `Simulation.Nsteps` | Positive metadata; only one selected stress is solved |
| `Simulation.Delta_time_file`, `Properties.Sy/Ss`, `IC.Head` | Accepted inactive steady metadata. Supplied files are read/validated; storage and initial-head evolution are not assembled |
| `FEM.MeshIterations` | Existing solve/elevation-update cycles, not transient steps; default 1 |
| `FEM.MaxRefinements` | Existing refinement-loop limit; default 0 |
| `FEM.WellModel` | Default mnw: geometry can be validated, but solving with wells is refused until milestone 3. Choose legacy explicitly to apply the existing simple-well assembly |
| `Solver.System_tol`, `System_iterations` | Positive relative tolerance and CG iteration limit. The latter now actually controls SolverControl |
| `Output.Prefix`, `Print_vtk`, `Print_initial_mesh` | Existing output controls; formats unchanged |
| `Tracing.*` | Existing tracing parameters mapped without changing its algorithm; disabled by default for INI flow runs |

Most numerical defaults follow v1, not v2. Use the printed schema and set
important choices explicitly when comparing solvers. Do not pass the full v2
solver/nonlinear/checkpoint configuration into v1: unknown keys are errors.
Likewise, v2 rejects the FEM/Tracing sections. This deliberate distinction
prevents silently accepting features that are not implemented.

## Time columns and interpolation

Matrices remain headerless, rectangular tables: spatial row IDs are zero-based,
columns are stress scenarios. A one-column table broadcasts its sole column to
any selected stress. A multi-column table must contain the requested column;
v1 INI does not clamp or cycle it. Every region is checked, including regions
that a center-point sample would not visit. Geometry, K, porosity, recharge,
Dirichlet and legacy Neumann all use this selection. In-memory moving-surface
interpolants continue to use their newly computed one-column data.

The spatial grammars remain those in [interpolation_v2.md](interpolation_v2.md).
CONST requires one value row. Grid/scatter readers validate their referenced
row IDs; lateral row-ID validation is now also done at load time. Empty/ragged
tables, invalid trailing tokens, nonfinite/out-of-range values and invalid
counts are errors. Comments on otherwise empty table lines are supported;
do not append comments to numeric rows.

The legacy `-p` path retains column-zero semantics. Public low-level time-series
queries retain their former extrapolation behavior; the strict selection is
enforced by the INI flow adapter. v2 may still clamp missing multi-column
scenarios, so use valid shared inputs and v1's validation rather than relying
on that permissive behavior.

## Wells and streams

Well geometry uses the simple numeric CSV form of the v2 grammar:

```text
Eid,x,y,top,bottom,q_row,rw,Rskin,Kskin
17,25,25,9,1,0,0.1,0.2,1
```

The header is checked. IDs must be unique/nonnegative, screen top must exceed
bottom, radius and skin conductivity must be positive, and skin radius must
be at least the well radius. Coordinates must be finite. `q_row` selects the
row of the separate pumping table. Negative rates mean extraction and positive
rates injection, matching both actual readers/assemblers. Unquoted numeric
fields are supported; this is not a general quoted-string CSV reader.

Legacy well mode keeps the old conductivity/screen-length source distribution.
It reads but does not apply the MNW radius/skin conductance model. External IDs
are preserved, including nonconsecutive IDs: KD-tree positions are explicitly
translated to stored IDs. This does not implement well-head unknowns or MNW
crossflow.

Stream geometry has a segment count followed by `npoints row_id [width]` and
XY vertices. Width is a **half-width** (both implementations offset the centerline
by that distance on each side), required and positive for two-point segments; three- and
four-point polygons are also accepted. Rates are signed recharge flux densities
in L/T, not total volume per time. Positive rates enter the aquifer. Geometry
is validated before conversion in memory to the existing stream constructor;
no intermediate legacy files are written. Invalid rows, duplicate well IDs,
degenerate stream geometry, and missing geometry/rate pairs are errors.

## MPI behavior and limits

INI, interpolation, well, stream and Neumann input text is read on rank 0 and
broadcast. Numeric-table parsing errors are broadcast before dimensions/data,
preventing workers from waiting forever when root cannot read a file. Existing
chunked transfers retain `int` counts for old MPI interfaces; matrix product
overflow is checked before allocation. The executable catches exceptions and
calls MPI_Abort so a rank-local failure cannot strand other ranks. Errors have
nonzero exit status. These are not large-scale MPI or allocation-failure tests.

No new C++14-only constructs were introduced. The build still requests C++14
because of deal.II 9.3; testing used GCC 9.3/Open MPI 4.0.3. Aqua/GCC 5 and its
MPI/Trilinos stack still require their own clean build and execution checks.

See [the portable example](../examples/shared_flow/README.md) and migration
section 1.1 for test commands and exact verification scope.
