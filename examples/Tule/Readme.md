Tule River example
===
The Tule River example is one of the most frequently used study areas for prototyping our tools. </br>
More information about this model can be found [here](http://groundwater.ucdavis.edu/Research/gw_203/).

The goal of this document is to show an overview of the workflow to simulate diffuse pollution 
using the NPSAT approach. 
The input files are all provided and the preparation is not discussed here.

Simulation of flow
---
The first step is to simulate the steady state groundwater flow. The main input file is the `tule.prm`.
The most important option in the input file is to enable the calculation and printing of the velocity field. 
```
set e Print velocity field cloud = 1
```
Another important option is the velocity multiplier, 
which is used to increase the precision of the outputs.
```
set f Velocity multiplier        = 10000
```

To run the model using 6 processors use the following
```
mpirun -n 6 ../../npsat -p tule.prm
```
The above command assumes that the current folder is where the tule.prm file is and the npsat executable lives two directories above.

The following figure shows the velocity cloud on a simulation with 6 processors.
![Velocity field simulated on six processors](VelocityField_shot1.png)

Redist
---
As we can see the distribution of the cloud points is very irregular 
and at least one of the subdomains is split into two parts. 
So the next step is to split velocity field into convex regular domains.
To do so we use [redist](https://github.com/giorgk/redist).

After preparing the redist input file we executed using the desired number of processor that the ichnos will be executed
```
mpirun -n 6 path/to/redist redist_input_file.dat
```
This will distribute the point cloud into new subdomains as follows.

![Velocity field simulated on six processors](VelocityField_shot2.png)

Ichnos
---
Finally the point cloud can be used to trace particles using the [Ichnos](https://github.com/giorgk/ichnos).

