# Steady-state FEM flow migration plan

Status: steps 0 and 1 implemented, 2026-11-19; local verification is recorded in
0.1 and 1.1. Aqua validation remains open.
The user approved C++14 instead of strict C++11 after the dependency audit below.
Flow assembly and tracing algorithms are unchanged.

## Baselines and scope

- Working branch: `codex/fem-flow-migration`.
- Base: latest fetched `origin/topo3`, `51888a1e75aa5f4ab5cb8b9a211936c6f3591493`.
- Reference: local `G:/UCDAVIS/ClionProjects/npsat_v2_9_3`, branch `trace_dbg`, commit `8e45fff` at inspection. Pin this reference for initial comparisons; its remote freshness was not checked.
- Target: continuous finite-element steady groundwater flow in `npsat.cc`, retaining v1 mesh movement and refinement. Reuse v2 data formats and suitable physics/helpers, not its HMFEM assembly.
- Required platform: deal.II 9.3, C++14, GCC 5, the installed Aqua MPI and Trilinos stack. C++14 was explicitly approved during step 0 because deal.II 9.3 cannot compile as strict C++11. Actual Aqua compiler/dependency versions still need to be recorded.
- Leave particle-tracing algorithms and their existing consumed formats unchanged. New flow exports must be separate or explicitly opt-in.
- No transient storage solve. Time-series inputs select a steady stress scenario.
- Preserve simple wells as an explicit legacy comparison mode; MNW is the intended production mode after validation.

## Findings from the current code

| Area | v1 location and current state | v2 reference and migration implication |
| --- | --- | --- |
| Flow | `myheaders/steady_state.h`: assembly, CG/AMG, output | `npsat_flow/main_class_impl/npsat_flow_assemble.impl.h` uses HMFEM condensation; derive FEM terms independently |
| Driver/mesh | `myheaders/npsat.h`: `solve_refine`, refinement, elevation updates | Retain v1 lifecycle; rebuild all geometric source/boundary mappings after each mesh change |
| Configuration | `myheaders/user_input.h` uses deal.II parameters | `npsat_flow/flow_input.h` uses Boost.Program_options; adapt option vocabulary and validation |
| Interpolation | `myheaders/interpolation/`, `TimeSeries.h`, `BC/dirichlet_bc.h` already migrated in topo3 | Existing migration predates inspected trace_dbg; test compatibility rather than recopy wholesale |
| Scenario | `Doc/interpolation_v2.md` documents time column zero | v2 has `Simulation.Steady_state` and `Simulation.Start_step`; use one selected column consistently |
| Neumann | `myheaders/neumann_boundary.h` and flow assembly already support legacy bottom/lateral fluxes | v2 exposes Neumann options, but inspected assembly and preparation contain placeholders; there is no complete general Neumann path to port |
| GHB | No equivalent wired into inspected v1 flow assembly | `npsat_flow/BC/ghb_bc.h` supplies geometry, external head and conductance-density functions |
| Wells | `myheaders/wells.h`: existing source treatment | `npsat_flow/mnwells.h`: screens, rates, conductance and MPI helpers; HMFEM coupling must be replaced by FEM coupling |
| Streams | `myheaders/streams.h` retains old source format | `npsat_flow/streams.h` and separate stream rates need a format/units audit |
| Budget | v1 assembly reports some prescribed-flow totals | v2 postprocessing and well residual output are useful references; FEM requires its own reaction accounting |
| Output | `steady_state.h`, `npsat.h`, `snapshot.h` | v2 output/checkpoint/Ichnos writers offer patterns, not an automatically safe shared serialization layer |
| Language | At initial inspection, main and test CMake files selected C++14 and copied headers used `std::make_unique` | Step 0 removed those calls, but confirmed deal.II 9.3 itself requires C++14; see the approved revision in 0.1 |

Existing interpolation tests live in `tests/interpolation_tests.cc`; the small input example is `examples/interpolation_v2`. Existing older examples are not a validated baseline for the new input contract.

## Intended numerical contract

Use Darcy flux `q = -K grad(h)`. In internal assembly and budget reporting define positive external flow as inflow to the aquifer. Convert file conventions explicitly at reader boundaries after documenting v2's actual conventions.

For a prescribed outward Neumann flux `g = q.n`, add `-integral(g v)` to the FEM right-hand side. For GHB `q.n = c(h-h_ext)`, add `integral(c u v)` to the matrix and `integral(c h_ext v)` to the right-hand side. The reference GHB code defines `c` as conductance per unit area; avoid applying boundary area twice. Reject negative/nonfinite conductance. Verify units for horizontal, lateral and projected recharge areas separately.

For an MNW connection, let `a_l^T h` be a documented FEM approximation to screen head, `C_l >= 0` its effective conductance, and `H_w` the shared well head. Aquifer inflow is `q_l = C_l(H_w-a_l^T h)`. For pumping `Q_w > 0`, require `sum C_l(a_l^T h-H_w) = Q_w`. This gives the symmetric contribution:

```
aquifer block:  +C_l a_l a_l^T
coupling:      -C_l a_l
well diagonal: +C_l
well RHS:      -Q_w
```

Use screen quadrature or an explicitly defined normalized screen average to construct `a_l`; compare alternatives for mesh sensitivity before freezing the contract. Adapt radius, skin and wet-screen conductance rules from v2 only after checking their meaning on FEM cells. Do not copy trace-DOF condensation coefficients. Apply hanging-node and inhomogeneous Dirichlet constraints to both coupling directions and RHS terms.

Initially solve the explicit coupled head/well system for transparency. Evaluate well-head elimination only after equivalence tests; elimination creates couplings between cells sharing a well and can increase sparsity costs. Assign each physical well a stable ID and one owner; count each connection exactly once across MPI ranks. Zero-conductance, fully dry and out-of-domain wells need explicit diagnostics and a documented reject/disable policy; never silently lose requested pumping or divide by zero. A zero-net-pumping well may still redistribute water between screens.

## Shared input contract

Adopt v2's INI sections and data-file grammar as the common contract where implemented. Map into v1 structures through an adapter so parsing does not dictate the FEM architecture. Maintain a compatibility table covering each key, units, sign, default, allowed values, path resolution and whether it is used by each solver.

- Common inputs: geometry, interpolation masters/spatial/value files, conductivity, recharge, Dirichlet, GHB, wells/screens/rates, streams/rates, and stress selection.
- Shared scenario: `Simulation.Steady_state=1`, `Simulation.Start_step=k`; validate the selected column in every relevant series. Define one-column constant broadcasting explicitly and test both solvers' behavior. No silent clamping of missing columns.
- Transient-only settings may be accepted as documented inactive fields in steady mode. Do not silently ignore unknown keys or accept a request for transient v1 flow.
- Some semantics are necessarily solver-specific: v1 moving mesh versus v2 unconfined constitutive treatment, mesh discretization, preconditioners and outputs. Use thin solver configuration overlays while referencing the same physical data files. Exact identical full configuration requires an agreed handling of these fields in both readers.
- Neumann needs a jointly documented outer grammar because v2's implementation is incomplete. First provide and test it in v1; a later narrowly scoped v2 reader/assembly change is needed before claiming both codes can run that boundary from the same file.
- Legacy wells should be selectable without editing the physical MNW dataset where a meaningful conversion exists; retain the original legacy-format reader only behind an explicit compatibility path if needed.
- Replace ParameterHandler for flow configuration. Translate retained tracing options into existing tracing structures without changing tracing algorithms. Provide a migration table for old `.prm` options; decide a one-time converter versus a temporary legacy parser during the parser milestone.

## Ordered implementation milestones

### 0. Reproducible baseline and C++11 compatibility (revised to C++14)

Record toolchain, MPI, Trilinos components and exact source revisions. Save a small fixed-mesh steady baseline and a moving-mesh baseline before behavior changes. Audit language compatibility, bundled headers and build flags, and link Boost.Program_options explicitly. Audit the existing solver convergence callback, which currently returns `SolverControl::success`, against deal.II 9.3 semantics before retaining it. The original strict C++11 requirement was revised by the user to C++14 during execution; retain GCC 5 as the deployment target.

Acceptance: existing interpolation tests and baseline flow cases build/run on the target stack; archive heads, applied sources, mesh and solver diagnostics. Confirm no tracing code changes. Do not infer GCC 5 support from a modern compiler build.

#### 0.1 Implementation and resumption record — 2026-11-19

**Completion boundary.** The code changes and local baseline verification are complete.
The target Aqua/GCC 5/old-MPI acceptance check is still outstanding: no Aqua session
or GCC 5 installation was available. This is not a claim that production deployment
has been validated. Changes are on `codex/fem-flow-migration`, based on
`51888a1e75aa5f4ab5cb8b9a211936c6f3591493`; they were not committed or pushed in this step.
The v2 reference repository was not modified.

**Language requirement and evidence.** An actual strict `-std=c++11` build was
attempted after replacing all five local `std::make_unique` calls. Both the flow
executable and interpolation test failed in deal.II 9.3 headers, including
`base/table_indices.h`, `base/tensor.h` and `base/std_cxx17/tuple.h`. These use
C++14 relaxed `constexpr` and `std::index_sequence`/`std::make_index_sequence`.
The installed `base/std_cxx14/memory.h` directly aliases `std::make_unique`.
This cannot be repaired by changing NPSAT alone. The user explicitly selected
"Use C++14; retain GCC 5 target." Accordingly, final targets require C++14;
there are no dependency-header patches, standard-library shims, or silent
fallback to a newer language. Failed strict-C++11 logs remain in
`build-step0-cxx11/build.log` and `build-step0-tests/build.log`.

**Code/build changes.**

- `CMakeLists.txt`: replaced the tutorial/autopilot setup with an explicit `npsat`
  target, preserved deal.II target setup, and set `CXX_STANDARD=14`,
  `CXX_STANDARD_REQUIRED=YES`, `CXX_EXTENSIONS=NO`. The effective compile command
  was checked for `-std=c++14`. The old global flag could be replaced by deal.II's
  cache initialization; target properties remove that ambiguity. CMake 3.1 is
  the declared minimum for this target-property approach. Existing autopilot
  convenience targets are no longer provided; use standard configure/build commands.
- Require MPI, Trilinos and p4est explicitly. The former PETSc-or-Trilinos check
  accepted PETSc-only configurations even though this application directly uses
  Trilinos wrappers. This is a dependency check, not a solver-backend migration.
- Discover and link Boost.Program_options explicitly through CMake. The parser
  itself still uses ParameterHandler; replacing it belongs to milestone 1.
  Set `BOOST_ROOT` on Aqua if necessary to match deal.II's Boost installation.
- `myheaders/BC/dirichlet_bc.h`: construct the interpolation function using
  `unique_ptr::reset(new ...)`. `myheaders/interpolation/interp_interface.h`:
  construct each of the four concrete interpolants directly into a base-class
  `unique_ptr`. Ownership and interpolation behavior are unchanged. These
  constructs are C++11-compatible individually, but the complete target is C++14.
- Audited bundled nanoflann 1.9 and interpolation headers for newer-language
  constructs. The bundled header was not changed; it compiles as part of the
  tested application/interpolation targets. This does not constitute GCC 5
  verification of every unused template instantiation.
- `myheaders/steady_state.h`: documented why the logging callback returns
  `SolverControl::success`; its behavior was retained. Installed deal.II 9.3
  `lac/solver.h` documents `failure > iterate > success` in `StateCombiner`,
  and `SolverBase` also registers the real SolverControl check. A logging slot
  returning success cannot override iteration or failure; returning iterate
  unconditionally would prevent convergence.
- `tests/solver_callback_tests.cc`: a three-eigenvalue SPD problem verifies that
  an always-success observer still takes three CG iterations and reaches a true
  residual below `1e-12`. A separate one-iteration limit verifies that the same
  observer does not suppress `NoConvergence`.
- `tests/CMakeLists.txt`: uses the same explicit C++14 requirements, CMake 3.1
  minimum, and registers the callback test alongside the interpolation test.
- `tests/flow_baseline.py`: Python-3-standard-library-only baseline harness.
  Generates and saves all input files, launches MPI with a timeout, refuses to
  overwrite a run directory, decodes compressed VTU data, checks analytic heads
  and mesh movement, archives source/solver diagnostics, and compares to references.
  This is optional test infrastructure, not a runtime Python dependency.
- `tests/baselines/step0/`: durable `serial.json`, `mpi2.json`, `toolchain.txt`
  and a README explaining the fixtures, tolerances, scope and reproduction.
  `Doc/interpolation_v2.md` now correctly attributes C++14 to deal.II rather
  than to the removed `make_unique` calls.

**Local toolchain.** Windows PowerShell launched the existing Docker image
`dealii/dealii:v9.3.0-focal`, digest
`sha256:3d9b4bca5a1e031705d7e774e6744fc871dfb8a62768690eef7fc7f50d8be016`,
with the repository mounted at `/work`. Versions: deal.II 9.3.0, GCC 9.3.0,
CMake 3.16.3, Open MPI 4.0.3 (MPI API 3.1), Boost 1.71.0, Trilinos 12.14.1,
and p4est 2.2. `HAVE_AMESOS_KLU` is defined, and the existing AMG/KLU path ran
successfully. The image lacked Boost.Program_options; package
`libboost-program-options-dev` and its 1.71 development/runtime dependencies
were installed only in the disposable validation container. This did not modify
host/Aqua libraries. No GCC 5 executable was present. Old MPI count limits and
large distributed I/O are not exercised by these small tests.

**Numerical baseline and results.** The original code was built before code
edits in `build-step0-baseline`, with explicit `-std=c++14` and Debug mode.
Its executable SHA-256 is stored in both reference JSON files. Existing
`build-topo3` binaries/output were not used as the reference.

The generated problem is a 100 x 100 horizontal box, bottom z=0, initial top
z=12, uniform K=1, bottom Dirichlet h=10, and uniform recharge 0.001. All other
sources and tracing are disabled. Exact fixed-domain head is
`h(z)=10+0.001*z`, with total applied recharge 10. The fixed case performs one
solve; the moving case performs three solves and two elevation updates. The
top moves from 12 to between 10 and 10.02. Adaptive refinement is disabled;
this tests elevation-update integration, not adaptive refinement or a converged
nonlinear solution. Every run verifies analytic heads within `3e-6` because
the existing VTU writer stores Float32 data.

| Check | Result |
| --- | --- |
| Original fixed and moving cases, 1 and 2 MPI ranks | Passed analytic heads, recharge and movement checks |
| Revised executable, Debug, explicit C++14 | Built successfully |
| Interpolation and callback tests via CTest | 2/2 passed |
| Interpolation integration test, 2 MPI ranks | Passed on both ranks |
| Revised fixed/moving outputs vs original, same stack and rank count | Maximum decoded-array difference 0 for all four comparisons |
| Prescribed recharge, CG iteration counts and logged residuals | Identical before/after; recharge 10, one CG iteration per flow solve |
| Strict C++11 build | Failed in deal.II headers; requirement revised with user approval |
| GCC 5 / Aqua / old MPI | Not run; deployment gate remains open |

All decoded arrays are retained: head, coordinates, cell connectivity, offsets,
cell types, conductivity and subdomain. Timestamped XML bytes are not compared.
Full prescribed-source balance and Dirichlet reactions are not available yet;
reporting recharge 10 is not proof of full global budget closure. These baselines
do not cover wells, streams, nonzero Neumann loads, adaptive refinement or large
MPI partitions. Existing compiler warnings (including nanoflann fallthrough and
unused helpers) remain; unrelated code was not changed to suppress them.

**Artifacts available locally.** Build/output directories are ignored by Git;
the compact reference files under `tests/baselines/step0` are intended for source
control. Paths below are repository-relative:

- `build-step0-baseline/{configure.log,build.log,npsat}`: original build.
- `build-step0-baseline/{serial,mpi2}/`: original copied inputs, case parameter
  files, run logs, raw VTU/PVTU and decoded `summary.json`.
- `build-step0-cxx11/`: dependency installation, configuration and failed strict
  C++11 flow build; `build-step0-tests/`: failed strict C++11 tests.
- `build-step0-final/`: final build and compile-command database;
  `serial-verified/` and `mpi2-verified/` hold successful comparison runs, with
  corresponding `serial-verified.log` and `mpi2-verified.log`.
- `build-step0-tests-final/`: test build, compile-command database and `ctest.log`.
- An initial `build-step0-final/serial/` run passed numerical comparisons but
  hit a harness-only diagnostic comparison bug (Python tuples vs JSON lists).
  That normalization was fixed and both rank-count runs passed afterward;
  use the `*-verified` artifacts as the final result.

**Reproduce on a Linux environment with the dependencies installed.** These
commands run from the repository root. Use fresh run-directory names because
the harness deliberately refuses overwrites. On older CMake installations use
`mkdir build; cd build; cmake ..` instead of the newer `-S/-B` CLI syntax.

```sh
cmake -S . -B build-flow -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=/path/to/dealii -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-flow -- -j2
cmake -S tests -B build-flow-tests -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=/path/to/dealii
cmake --build build-flow-tests -- -j2
(cd build-flow-tests && ctest --output-on-failure)
mpirun -np 2 build-flow-tests/interpolation_tests examples/interpolation_v2
python3 tests/flow_baseline.py --exe build-flow/npsat --output build-flow/serial --ranks 1 --compare tests/baselines/step0/serial.json
python3 tests/flow_baseline.py --exe build-flow/npsat --output build-flow/mpi2 --ranks 2 --compare tests/baselines/step0/mpi2.json
```

For a new Docker validation environment, mount this repository at `/work`, use
the image/digest above, install `libboost-program-options-dev` inside the
container, and run these commands there (deal.II is automatically found; omit
the placeholder `DEAL_II_DIR`). The baseline harness sets Open MPI's root-run
environment switches for containers; an independent root-run MPI test needs
`OMPI_ALLOW_RUN_AS_ROOT=1` and `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1` as well.
The local container `npsat-step0-validation` was stopped after verification and
retained with its installed dependency. Resume it with
`docker start npsat-step0-validation`, then use `docker exec` to run commands
in `/work`; stop it again when finished.

**Resume here.** Before declaring target-platform acceptance, record Aqua's exact
GCC 5 minor version, CMake, deal.II, Boost, Trilinos/KLU, p4est and MPI versions;
perform clean builds of the application and tests; inspect effective flags;
then run the same serial/two-rank checks. Start the baseline harness without
`--compare` on Aqua: exact logged residual equality is a same-stack regression
criterion and may not hold across compilers/MPI reductions. Keep its analytic
checks, compare heads/geometry with documented tolerances, and archive that
stack's own references. If Python 3 is absent on Aqua, generate the inputs
elsewhere and run the C++ executable with those `.prm` files, then analyze output
on the workstation. Do not install/replace cluster libraries without its normal
deployment process. After this gate, proceed to milestone 1 (typed Boost options
and the shared input contract). Solver tolerances, preconditioner tuning, full
budget accounting and adaptive-mesh safeguards remain in their later milestones.

### 1. Boost configuration and shared-data validation

Implement typed flow options, a v1 adapter, documented defaults, selected stress column and collective validation. Audit topo3 interpolation against pinned v2; add common fixtures rather than assuming compatibility. Port well/stream geometry and rate readers independently of their assembly. Establish a canonical parsed-input summary for comparing both executables.

Acceptance: both readers agree on common fixtures and selected values; bad paths, invalid dimensions, missing IDs, nonfinite numbers and unavailable columns fail clearly on all ranks. Old-to-new option mapping is documented. Preserve the baseline under the new parser.

#### 1.1 Implementation and resumption record — 2026-11-19

**Scope and entry points.** Added a preferred Boost.Program_options flow path:
`npsat -c model.ini`, with `--validate-only`, `--print-config`, and
`--help-config`. The existing `-p model.prm` path remains an explicit temporary
compatibility path for comparison, tracing and gather workflows. No particle
tracing algorithm was changed. New C++ code uses C++11 facilities; compilation
remains explicit C++14 for deal.II 9.3 as approved in step 0. GCC 5/Aqua validation
has not been performed.

**Architecture and files changed.**

- `myheaders/flow_options.h` contains typed options, range/consistency checks,
  path resolution and deterministic summaries. `flow_option_schema.h` lists
  all 75 old-label-to-INI mappings plus the new shared-data/scenario options.
  Unknown/duplicate options are rejected. The mapping exposes v2 names where
  their meanings match; v1-specific mesh, output and tracing settings live under
  FEM/Tracing instead of being disguised as v2 solver options.
- `myheaders/user_input.h` shares the existing AquiferProperties initialization
  through a templated adapter. The INI path directly supplies typed Boost values;
  it does not write a temporary `.prm` or parse values through ParameterHandler.
  Tracing parameters still populate the existing structures. INI tracing defaults
  off; mesh iterations default to 1 and adaptive refinement iterations to 0.
  Legacy `.prm` defaults are retained.
- New INI `Discretization.Nxyz` follows v2's cell counts. An empty Vertical option
  generates Nz+1 coordinates; an explicit BOX list must have Nz+1 entries and
  increase from 0 to 1. The adapter handles the older v1 vertical-level convention
  internally. Horizontal counts remain cell counts. The old `.prm` semantics
  are unchanged. File-valued vertical distributions are explicitly unsupported
  by this new adapter; inline lists are the tested common subset.
- `myheaders/shared_sources.h` reads the v2 nine-column well geometry CSV and
  separate rates, and stream geometry with row IDs and separate rates. It checks
  counts, IDs, screen/radius/skin parameters, finite coordinates, stream geometry,
  rate-row references and column availability. Negative well rates remain
  extraction, positive rates injection; stream rates are signed flux densities.
- `FEM.WellModel=legacy` is required explicitly to solve with the shared wells
  before milestone 3. The default mnw mode supports validation but refuses a
  well-containing solve. Radius/skin are read and validated, not used by legacy
  assembly. The adapter initializes legacy source containers from memory and
  writes no converted source files. `streams.h` now accepts a stream object as
  well as the old filename, reusing its existing geometry construction.
- `wells.h` now translates KD-tree positions into the stored external IDs. The
  source fixture intentionally uses Eid 17: this exposed the old assumption that
  an ID always equaled its position, which caused an invalid map lookup. Existing
  sequential-ID legacy data retains the same behavior.
- `InterpInterface::select_steady_step` pins the chosen column for input-backed
  functions. Every region is validated, even if it is never sampled in a test.
  One-column regions broadcast; multi-column regions must contain the selected
  column. Top/bottom, conductivity, porosity, recharge, Dirichlet and legacy
  Neumann use the same selection, including subsequent mesh updates. Existing
  explicit time-query and in-memory moving-surface behavior remains available.
- Dirichlet data is validated before mesh creation and its selected scenario and
  lateral tolerances are also applied when the driver loads its own boundary
  objects. CONST requires one value row; lateral interpolation now validates its
  referenced row IDs at load time, alongside the existing grid/scatter checks.
- `reader_helper_func.h` rejects empty/ragged matrices, invalid trailing tokens,
  nonfinite/out-of-range numbers and overflowing dimension products. Rank-0
  table-read errors are broadcast before the dimensions/data collectives, fixing
  the previous missing-file deadlock. Neumann outer files now use collective
  text reads and validate records/coordinates. Existing chunked MPI transfers
  continue using legacy-compatible integer counts.
- `npsat.cc` returns nonzero on input failure and aborts MPI on caught exceptions
  so peers cannot remain stuck in a collective after a rank-local failure.
  Normal help/validation exits remain successful. Solves require an existing
  output directory; this is not the complete safe-output implementation of step 6.
- `Solver.System_iterations` now actually controls SolverControl instead of the
  number of head DoFs. No algorithm/preconditioner change was made. Both property
  initialization and the solve driver now handle KY supplied without KZ by
  using KX for KZ; the new KY-only test exposed the second null-pointer path.

**Shared-input contract and deliberate limits.** See
`Doc/shared_flow_inputs.md` and `Doc/flow_option_mapping.md` for the option table,
path rules, units, signs, defaults and compatibility limits. Physical input files
can be prepared once; whole configuration files are not interchangeable because
v2 rejects v1's FEM/Tracing sections and v1 rejects unsupported v2 solver keys.
`examples/shared_flow/v2.ini` supplies a separate reference configuration pointing
to the same physical files. Full HMFEM flow execution is not claimed by step 1.

Only steady mode is accepted. Nsteps is metadata; Start_step selects one stress,
and MeshIterations controls v1 solve/move iterations. Confined mode requires a
fixed mesh. Supplied Sy/Ss/IC.Head masters and a delta-time file are read/validated
as inactive steady metadata, not assembled as transient storage. GHB must remain
empty until milestone 2. Neumann still uses the documented v1 BOT/BOUNDARY_LINE
outer grammar; v2 general Neumann support remains unfinished. The new readers are
stricter than v2 about malformed inputs and missing stress columns. Validation
does not prove spatial coverage, positive conductivity everywhere, mesh validity,
well intersection, full physical budget closure or output writability.

**Portable verification assets.**

- `examples/shared_flow/` is a self-contained directory with fixed, second-stress,
  moving-mesh, shared-source and reference-v2 INI files, plus all referenced input
  data. No machine-specific source/data paths appear in these fixtures. Its
  README includes build, run, validation and optional v2-comparison commands.
- The analytic cases use a 100x100 box with 2x2x1 cells. Column 0 has bottom head
  10, initial top 12, K=1 and recharge .001 (`h=10+.001*z`, total recharge 10).
  Column 1 has bottom head 20, top 24, K=2 and recharge .004
  (`h=20+.002*z`, total recharge 40). Bottom elevation and porosity use one-column
  tables to exercise constant broadcasting. The moving case performs three
  solves with two elevation updates and no adaptive refinement.
- `tests/shared_flow_tests.py` copies the fixture inputs to fresh run directories,
  runs the executable on one/two ranks, retains logs and generated inputs/outputs,
  checks analytic heads, compares decoded fixed/moving VTU data to step-0
  references, checks successful validation-only modes, and verifies malformed
  inputs fail with nonzero status under two-rank MPI. Every subprocess has a
  timeout so a collective hang fails the test rather than hanging indefinitely.
  The harness uses only Python 3's standard library and reuses `flow_baseline.py`.
- `tests/shared_reader_probe.cc` can be built against either v1 or the unmodified
  v2 `npsat_flow` headers. The optional `NPSAT_V2_REFERENCE` CMake path enables the
  second executable; it is not required for ordinary tests. The fixture's
  `expected_reader.txt` preserves the actual v2 result so another PC can check
  the same reference without having a v2 checkout.
- The reference was the read-only `trace_dbg` checkout at
  `8e45fff22713b5e9cfde40face27f58f04f1ac00`. Its pre-existing `.idea/` directory
  was untouched. Both readers produced identical output for two scenarios of
  constant/grid/scatter/lateral interpolation, Dirichlet values, well geometry
  and rates, and stream rates. This tests the data readers, not equality of FEM
  and HMFEM flow solutions.

**Build and verification environment.** Reused the step-0 deal.II 9.3 Docker
image and the `npsat-step0-validation` container with its installed
Boost.Program_options package: GCC 9.3, CMake 3.16.3, Open MPI 4.0.3, Boost 1.71,
Trilinos 12.14.1 and p4est 2.2. The reference-reader build mounted v2 read-only.
The full current result and final log locations are recorded below after the
verification run; use those rather than intermediate debugging artifacts.

**Reproduce and resume.** From the repository root:

```sh
cmake -S . -B build-flow -DCMAKE_BUILD_TYPE=Debug
cmake --build build-flow -- -j2
cmake -S tests -B build-flow-tests -DCMAKE_BUILD_TYPE=Debug
cmake --build build-flow-tests -- -j2
(cd build-flow-tests && ctest --output-on-failure)
python3 tests/shared_flow_tests.py --exe build-flow/npsat --output build-flow/verification --reader build-flow-tests/shared_reader_probe
python3 tests/flow_baseline.py --exe build-flow/npsat --output build-flow/legacy-serial --ranks 1 --compare tests/baselines/step0/serial.json
python3 tests/flow_baseline.py --exe build-flow/npsat --output build-flow/legacy-mpi2 --ranks 2 --compare tests/baselines/step0/mpi2.json
```

Use fresh run directories, supply deal.II/Boost paths for the local installation,
and use the older configure syntax described in the example README if necessary.
For a live reference comparison, configure the test build with
`-DNPSAT_V2_REFERENCE=/path/to/npsat_v2_9_3/npsat_flow` and add
`--reference-reader build-flow-tests/v2_reader_probe` to the Python invocation.
The scripts refuse to overwrite previous run directories. Build trees, generated
outputs and Python caches are ignored; source fixtures, tests and documentation
are intended for version control. Code-generation helpers under ignored
`build-step1/` were development conveniences only: the explicit schema header
and mapping document are the maintained sources and need no generator to build.

Before Aqua deployment, rebuild with its actual GCC 5/deal.II/MPI/Trilinos stack
and archive its versions/results as described in 0.1. Existing warnings and
large-scale I/O/MPI limitations remain outside this step. The next implementation
work is milestone 2: GHB/general Neumann assembly and reaction-based budget
accounting, using the now-testable INI/shared-data path. MNW assembly remains
milestone 3. Do not remove the legacy reader until its tracing/gather consumers
and the documented comparison cases have an agreed migration path.

### 2. Boundary assembly and initial budget accounting

Introduce explicit boundary classification and overlap rules. Retain recharge and streams; adapt GHB geometry/functions and generalize existing Neumann handling. Reject incompatible overlapping Dirichlet/GHB/Neumann assignments unless a supported additive combination is explicitly defined. Preserve geometric top/bottom identity independently of assigned boundary IDs.

Compute prescribed inflows, GHB exchange and Dirichlet reactions using the original discrete equations, before Dirichlet row replacement/condensation obscures reactions. Include hanging-node constraint transformations. This accounting becomes the MNW verification foundation.

Acceptance: constant-head zero-flow, linear-head Dirichlet, prescribed-flux balance and analytic one-dimensional GHB cases; signs, boundary area and MPI partition invariance checked. Explain pure-Neumann compatibility and nullspace handling; reject incompatible net flux instead of regularizing silently.

### 3. MNW FEM coupling and legacy comparison

Build screen/cell intersections and shape-function coupling, add stable well DOFs and distributed sparsity, and assemble the symmetric coupled equations above. Start with confined fully wet wells, then implement wet/dry conductance and pumping policy with explicit convergence guards. Keep legacy simple wells selectable and report the chosen model.

Acceptance: single-connection pumping balance, multiple screens with unequal conductivity, zero-pumping crossflow, injection, skin/radius variation, dry/disconnected screens, screens on partition faces and hanging nodes. Verify total screen exchange equals delivered pumping and expose requested versus delivered pumping. Compare MNW/legacy only where their physical assumptions coincide.

### 4. Mesh movement and refinement integration

Preserve `solve_refine` and existing elevation/refinement modes. After movement, refinement or repartition, rebuild DoFs/constraints, boundary matches, screen intersections, stream/recharge integration, material evaluations and matrix sparsity. Transfer heads as an initial guess where valid and preserve well identities rather than stale cell/DOF indices.

Add head-update, mesh-displacement and budget convergence criteria with maximum iterations. Evaluate damping and mesh-quality rejection/rollback; never accept inverted cells or nonpositive layer thickness. Separate mesh-refinement iterations from physical nonlinear convergence in reporting.

Acceptance: fixed and moving meshes, refinement and coarsening, changed screen intersections, partition changes, stable water table and budget. All published results correspond to a solved final mesh.

### 5. Solver and preconditioner improvements

Retain CG with AMG as the baseline only when the constrained operator and preconditioner are symmetric positive definite. Handle unanchored domains explicitly. Add an absolute tolerance floor alongside relative tolerance, configurable iteration limits, periodic logging and an independently recomputed final residual. Distinguish linear failure from nonlinear/mesh failure.

Benchmark the coupled system, block preconditioning and well elimination before choosing defaults. Use only components available in Aqua's installed Trilinos; the current hard-coded `Amesos-KLU` choice needs availability validation. Consider GMRES only for a justified nonsymmetric formulation/preconditioner. Reuse AMG only while its operator/mesh assumptions remain valid.

Acceptance: residual correctness including zero RHS, anisotropy/contrast tests, rank scaling, setup/solve time and memory measurements. Performance changes must preserve head and budget accuracy.

### 6. Complete budgets and safe ASCII/binary flow outputs

Report signed inflow/outflow by recharge, streams, Neumann, GHB, Dirichlet reactions and wells; include per-boundary/per-well totals, requested/delivered pumping, absolute residual and a documented normalized discrepancy with a zero-flow floor. Sum owned entities only. Keep algebraic residual, physical global budget and optional cell flux diagnostics distinct: continuous FEM does not automatically have HMFEM's cellwise local conservation.

Provide new flow-only ASCII and binary exports with equivalent content and explicit metadata: schema/version, units, scenario, mesh identity, entity IDs and counts. Use sufficient ASCII precision and fixed-width binary fields with declared byte order; never serialize padded native structs or process-local indices as portable identities. Keep existing tracer-consumed exports intact.

Validate sizes and arithmetic overflow before allocation; check opens, writes, flush and close. Bound/chunk MPI transfers to supported integer counts and throttle parallel file writers. Coordinate rank-local failures before subsequent collectives so other ranks cannot hang. Write temporary parts, verify global success, then publish a manifest as the completion marker; do not advertise incomplete multi-file output as valid. Document filesystem rename assumptions and overwrite policy. Reuse audited v2 writer/checkpoint patterns where applicable.

Acceptance: ASCII/binary round trips, truncated/corrupt/wrong-version files, unwritable destinations, one-rank failures and incomplete manifests. Read-back verifies entity counts, metadata and values; compare budgets across formats.

### 7. Cross-code comparison and release gate

Run both solvers using the same physical input files and steady stress selection. Start on fixed confined meshes; compare spatially sampled heads, integrated boundary/well flows and mesh-convergence trends rather than raw nodal/cell vectors. Moving-water-table results need separate interpretation because mesh movement and v2's constitutive wet/dry model can differ.

Run serial and at least 2/4 MPI ranks, including ranks without wells or sources. Include malformed-input and interrupted-output cases. Establish quantitative tolerances from analytic solutions and baseline convergence before accepting results. Publish example configurations, input compatibility table, numerical assumptions, benchmark results and remaining unsupported features.

## Decisions to resolve during implementation

1. Exact Aqua toolchain/library availability and representative production datasets.
2. Screen averaging/quadrature and effective-radius interpretation for FEM refinement.
3. Fully dry-well pumping policy and nonlinear convergence thresholds.
4. Common Neumann grammar and whether to implement matching support in v2 in a separate change.
5. Which flow fields need portable binary export, and whether restart serialization is a separate requirement.

These do not block branch setup or this plan. Implementation should begin with milestone 0, followed by the input contract; do not start by copying v2's assembled operator.

## Historical verification of the initial planning change

At initial plan creation, fetched v1 origin with TLS verification using the Windows certificate store and confirmed the topo3 base. Inspected both local working trees, flow readers, assembly/solver paths, mesh loop and existing migration documentation. No executable tests were run during that documentation-only pass. Step 0 implementation and subsequent test results are recorded in subsection 0.1 above.
