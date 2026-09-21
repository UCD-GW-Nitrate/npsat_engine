# Interpolation input on topo3

The interpolation and Dirichlet implementations come from
[giorgk/npsat_v2_9_3, trace_dbg](https://github.com/giorgk/npsat_v2_9_3/tree/7f6733787bf440ff6850fd4973090d3969aa3f9b/npsat_flow)
(commit `7f6733787bf440ff6850fd4973090d3969aa3f9b`).
The solver still performs steady-state calculations. All material properties,
recharge, boundary values and prescribed elevations use **time column 0**.
Additional columns are retained for future scenario selection; there is no
transient flow solve or scenario loop in this change.

## Master files

Top elevation, bottom elevation, KX/KY/KZ, porosity, recharge and boundary
values now reference interpolation master files. Bare scalar parameters and
legacy SCATTERED/GRIDDED/MULTIPOLY/BOUNDARY_LINE interpolation files are no
longer accepted. An omitted KY or KZ still follows the existing isotropy rules.
Old examples and MATLAB writers use the legacy format; use the new example
below as the format reference.

```
number_of_regions
number_of_vertices TYPE spatial_file values_file
x0 y0
x1 y1
...
```

Two vertices define opposite corners of an XY rectangle; three or more define
a polygon. The first region containing the query point wins. A point outside
all regions returns zero, so regions must cover the intended model domain.
Types are `CONST`, `SCATTER`, `GRIDDED`, and `LATERAL_POLYLINE` (`LATERAL` is an
alias). Relative paths, including nested spatial/value references, resolve
against the configured input directory, not the master file's directory.

Values are a headerless rectangular matrix with spatial IDs as zero-based row
indices and time steps as columns (dimensions are inferred by the reader):

```
value_row0_time0 value_row0_time1 ...
value_row1_time0 value_row1_time1 ...
```

For a constant of 10, the values file contains just `10`. For two scenarios
it contains just `10 20`. Do not put a dimensions header in the values file. CONST ignores its spatial filename, so `unused` is valid.
The master still supplies its region coordinates.

`SCATTER` spatial files contain point/triangle counts, XY/Z interpolation modes,
point rows `x y ID0 [interface_z ID1 ...]`, and zero-based triangle vertex IDs.
`GRIDDED` spatial files contain `x_origin ncols`, `y_origin nrows`, `dx dy`,
`nlayers`, then row-ID grids alternating with elevation-interface grids.
`LATERAL_POLYLINE` files contain `npoints nlayers`, horizontal/vertical modes,
XY polyline points, then row-ID vectors alternating with elevation-interface
vectors. Use `nlayers=0` for elevation-independent lateral values. Layer
interfaces run from top to bottom. See the copied headers for full definitions.

## Boundary files

Dirichlet records use the reference format:

```
number_of_parts
TYPE number_of_vertices interpolation_master_file
x0 y0
...
```

TYPE is TOP, BOT, EDGE or EDGETOP. Each record has explicit geometry: polygons
for TOP/BOT and polylines for EDGE/EDGETOP. The reference lateral matching
parameters are a half width of 20 and minimum overlap of 5, in model units.
The application rebuilds its top/bottom boundary-ID lists after assignment so
recharge and mesh updates continue to recognize those faces.

The Neumann outer file keeps this solver's `TYPE N multiplier master_file`
records. BOT takes N polygon vertices. BOUNDARY_LINE takes N=0 and a master
containing LATERAL_POLYLINE spatial data. Values use time column 0, with the
existing Neumann multiplier applied once. Well and stream source-file formats
are unchanged.

## Build and verification

Requires deal.II 9.3 or newer with the existing MPI/Trilinos/p4est dependencies.
C++14 is required by the copied implementation (`std::make_unique`). The
reference nanoflann 1.9 header is bundled, removing the external NANOFLAN and
GRID_INTERP include-path requirements.

```
cmake -S . -B build-topo3 -DCMAKE_BUILD_TYPE=Debug
cmake --build build-topo3 -j2
cmake -S tests -B build-topo3-tests -DCMAKE_BUILD_TYPE=Debug
cmake --build build-topo3-tests -j2
(cd build-topo3-tests && ctest --output-on-failure)
mpirun -np 2 build-topo3-tests/interpolation_tests examples/interpolation_v2
mkdir -p build-topo3/output
mpirun -np 2 build-topo3/npsat -p examples/interpolation_v2/smoke.prm
```

The integration test covers constants, gridded/scattered/lateral data, time
columns, shared ownership, diagonal conductivity, in-memory free surfaces,
and Dirichlet/Neumann integration. The small 3D smoke case performs two
nonlinear iterations, exercising the reconstructed surface path.

The upstream interpolation algorithms are preserved. Local changes provide
explicit includes and an in-memory `set_surface_data` input method for the
steady solver's triangulated free surface; it uses the same scattered
interpolation algorithm as the file reader.
