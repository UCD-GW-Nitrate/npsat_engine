# Legacy option mapping

Generated mapping used by the Boost adapter. See `shared_flow_inputs.md` for semantics and limits.

| Legacy label | INI key | Default | Type |
| --- | --- | --- | --- |
| `a Input directory` | `Paths.Input` | `.` | string |
| `b Output directory` | `Paths.Output` | `.` | string |
| `a Geometry Type` | `Geometry.Type` | `BOX` | string |
| `a XYZ dimensions` | `Geometry.BoxDims` | `5000,5000,300` | string |
| `b Left lower point` | `Geometry.BoxLLP` | `0,0,0` | string |
| `a Input mesh file` | `Geometry.MeshFileName` | `` | string |
| `b Relative elevation file` | `FEM.RelativeElevationFile` | `` | string |
| `a Top elevation function` | `Geometry.Top` | `` | string |
| `b Bottom elevation function` | `Geometry.Bottom` | `` | string |
| `c x-y threshold` | `FEM.XYThreshold` | `0.1` | double |
| `d z threshold` | `FEM.ZThreshold` | `0.01` | double |
| `e Power for top` | `FEM.TopPower` | `3` | double |
| `f Radius for top` | `FEM.TopRadius` | `1000` | double |
| `g Power for bottom` | `FEM.BottomPower` | `3` | double |
| `h Radius for bottom` | `FEM.BottomRadius` | `1000` | double |
| `a Nxyz` | `Discretization.Nxyz` | `10,10,3` | string |
| `b Vertical discretization` | `Discretization.Vertical` | `` | string |
| `c Initial Refinement` | `Refinement.Initial` | `0` | int |
| `d Well Refinement` | `Refinement.Wells` | `0` | int |
| `e Stream Refinement` | `Refinement.Streams` | `0` | int |
| `f Top Refinement` | `Refinement.Top` | `0` | int |
| `g Remaining Boundaries Refinement` | `FEM.BoundaryRefinement` | `0` | int |
| `a Dirichlet file name` | `BC.Dirichlet` | `` | string |
| `a Hydraulic Conductivity KX` | `Properties.Kxy` | `` | string |
| `b Hydraulic Conductivity KY` | `FEM.Ky` | `` | string |
| `c Hydraulic Conductivity KZ` | `Properties.Kz` | `` | string |
| `d Porosity` | `FEM.Porosity` | `` | string |
| `a Groundwater recharge` | `Sources.Rch_Data` | `` | string |
| `b Stream recharge` | `Sources.Stream_Data` | `` | string |
| `c Wells` | `Sources.Well_Data` | `` | string |
| `d Neumann Conditions` | `BC.Neumann` | `` | string |
| `e Recharge multiplier` | `Sources.Rch_factor` | `1` | double |
| `f Well multiplier` | `Sources.Well_factor` | `1` | double |
| `g Stream multiplier` | `Sources.Stream_factor` | `1` | double |
| `a Nonlinear iterations` | `FEM.MeshIterations` | `1` | int |
| `b Solver tolerance` | `Solver.System_tol` | `1e-8` | double |
| `c Max iterations` | `Solver.System_iterations` | `1000` | int |
| `d Output details` | `FEM.AMGDetails` | `0` | int |
| `a Max refinements iterations` | `FEM.MaxRefinements` | `0` | int |
| `b Top fraction` | `FEM.TopFraction` | `0.15` | double |
| `c Bottom fraction` | `FEM.BottomFraction` | `0.01` | double |
| `d Minimum element size` | `FEM.MinElementSize` | `1.0` | double |
| `a Enable particle tracking` | `Tracing.EnableParticleTracking` | `0` | int |
| `a Tracking method` | `Tracing.TrackingMethod` | `3` | int |
| `b Step size` | `Tracing.StepSize` | `6` | double |
| `c Stuck iterations` | `Tracing.StuckIterations` | `50` | int |
| `d Streamline iterations` | `Tracing.StreamlineIterations` | `1000` | int |
| `e Outer iterations` | `Tracing.OuterIterations` | `100` | int |
| `f Search iterations` | `Tracing.SearchIterations` | `3` | int |
| `g N Particles in parallel` | `Tracing.NParticlesInParallel` | `5000` | int |
| `a Particles from file` | `Tracing.ParticlesFromFile` | `` | string |
| `b Trace wells` | `Tracing.TraceWells` | `1` | int |
| `c Layers per well` | `Tracing.LayersPerWell` | `25` | int |
| `d Particles per layer(well)` | `Tracing.ParticlesPerLayerWell` | `4` | int |
| `e Distance from well` | `Tracing.DistanceFromWell` | `50.0` | double |
| `f Well Distribution type` | `Tracing.WellDistributionType` | `` | string |
| `a Particles Prefix` | `Tracing.ParticlesPrefix` | `` | string |
| `b Simplify threshold` | `Tracing.SimplifyThreshold` | `5.5` | double |
| `c Print entity frequency` | `Tracing.PrintEntityFrequency` | `1` | int |
| `d Print streamline frequency` | `Tracing.PrintStreamlineFrequency` | `1` | int |
| `a Prefix` | `Output.Prefix` | `` | string |
| `a Savix` | `FEM.Savix` | `sol` | string |
| `b Save Solution` | `FEM.SaveSolution` | `0` | int |
| `c Load Solution` | `FEM.LoadSolution` | `0` | int |
| `d Print solution vtk` | `Output.Print_vtk` | `0` | int |
| `e Print velocity field cloud` | `FEM.PrintVelocityFieldCloud` | `0` | int |
| `f Print dependency graph for cloud` | `FEM.PrintDependencyGraphForCloud` | `0` | int |
| `g Velocity multiplier` | `FEM.VelocityMultiplier` | `10000` | double |
| `h Print Boundary conditions` | `FEM.PrintBoundaryConditions` | `0` | int |
| `i Print Average Velocity` | `FEM.PrintAverageVelocity` | `0` | int |
| `j Print GW Recharge` | `FEM.PrintGwRecharge` | `0` | int |
| `a Print initial mesh` | `Output.Print_initial_mesh` | `0` | int |
| `b Print free surface point cloud` | `FEM.PrintFreeSurfacePointCloud` | `0` | int |
| `c Domain Scale X` | `FEM.DomainScaleX` | `1000` | double |
| `d Domain Scale Z` | `FEM.DomainScaleZ` | `500` | double |
