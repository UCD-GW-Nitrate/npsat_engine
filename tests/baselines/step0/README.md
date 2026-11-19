# Step 0 flow reference

Captured from unchanged topo3 commit `51888a1e75aa5f4ab5cb8b9a211936c6f3591493`,
before the step 0 ownership, CMake and callback-comment changes.
The capture script itself is new test infrastructure, not baseline solver code.

Image: `dealii/dealii:v9.3.0-focal`, digest
`sha256:3d9b4bca5a1e031705d7e774e6744fc871dfb8a62768690eef7fc7f50d8be016`.
See `toolchain.txt` for the compiler and dependency versions. Boost.Program_options
was installed in the disposable container after the old executable was built.
The original build did not explicitly require that component.

`serial.json` and `mpi2.json` contain every decoded VTU array (including heads,
coordinates and connectivity), applied recharge totals, logged CG residuals and
iteration counts, and the SHA-256 of the original executable. Rank partitioning
and Float32 VTU precision are part of this reference. Original raw files and logs
remain in ignored `build-step0-baseline/{serial,mpi2}` directories locally.

The test constructs a 100 x 100 box with bottom elevation 0, initial top elevation
12, K=1, bottom Dirichlet head 10, recharge 0.001, no other sources and no tracing.
The analytic head is `h(z)=10+0.001*z`; total applied recharge is 10.
The fixed case has one solve. The moving case has three solves and two elevation
updates, with adaptive refinement disabled. This is a baseline of mesh motion,
not a claim of nonlinear convergence or a refinement regression.

From the repository root, using a built executable:

```sh
python3 tests/flow_baseline.py --exe build-step0-final/npsat --output build-step0-final/serial --ranks 1 --compare tests/baselines/step0/serial.json
python3 tests/flow_baseline.py --exe build-step0-final/npsat --output build-step0-final/mpi2 --ranks 2 --compare tests/baselines/step0/mpi2.json
```

Use a fresh output directory for every run. The script refuses to overwrite
previous runs, saves the exact generated input files, checks analytic heads,
mesh movement, recharge and solve counts, then compares every decoded output
array. Connectivity is exact; numeric fields allow `2e-6*max(1,abs(reference))`
to accommodate Float32 output. Logged source/solver diagnostics must match exactly
when `--compare` is supplied; use the same stack and MPI rank count for that check.
On Aqua, first run without `--compare` to establish its own numerical reference.

The current code does not provide a complete reaction-based budget. These tests
check prescribed recharge and analytic heads, not global budget closure. Source
tests for wells, streams and nonzero Neumann conditions belong to later milestones.

Python 3 and `mpirun` are needed for this optional harness, not for the production
C++ executable. The VTU decoder intentionally targets deal.II 9.3's little-endian,
inline zlib-compressed output; it is not a general VTK reader.
