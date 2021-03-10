# Overview
This example is primarily used for debugging and illustrating the various boundary conditions that can be applied in a domain. The *Dirichlet* boundary conditions are all specified in a single file. The *Neumann* boundary conditions are split into two categories. Groundwater recharge and other Neumann boundaries. Although groundwater Recharge is a type of Neumann boundary condition it is treated separately as it is always applied on the top, while the category Neumann boundary conditions in the input file it is used to set flows in boundaries of the domain other than the top.  

## Domain 
The domain of the example is a box with dimensions 5x5 km and depth 300 m approximately. The bottom of the aquifer is uniform and equal to -270 m. The initial approximation of the free surface is set to 30 m above msl.

## Stresses
* Groundwater recharge unless it is modified, it is assumed uniform and equal to 0.0002 m/day, which results in a total incoming volume of water from recharge equal to  5000 m^3/day.
* There are 19 wells with pumping rates varying from 100 to 500 m^3/day. Their total pumping equals the recharge amount.

# Dirichlet Boundary conditions
The purpose of this example is to test all the available boundary condition options of NPSAT. 
The format of the boundary condition file is the following:
- Number of boundaries
- TYPE N Value 
- Repeat N times the coordinates that describe the boundary


## Top boundary (Test 1 : dir_bc01.npsat)
 This type of boundary is a polygonal area applied on top (e.g. lake) that has constant head. In the example this is set to 30 m, while the polygon consist of 7 vertices. The file has the following format
```
1
TOP 7 30
x y  #repeat 7 times
```
**It is very important that the orientation of the polygon is counter clockwise**

The value does not have to be a constant value. On can pass a file that describe an interpolation function. 

We can see that that boundary polygon has been refined several times so that it is clrearly identified
<img src="Box3D_dirich1.png" alt="Dirichlet test 01" width="700"/>

# Side Boundaries (Test 2 : dir_bc02.npsat) 
This type of boundary is used when the side of the domain is associated with a constant head boundary.
In the 2nd example we set two boundaries 
* The left side of the aquifer (x = 0) is set equal to 25 m. This value will be assigned to the entire depth.
* The right side of the aquifer (x =5000) is set equal to 40 m. However this value will be assigned only on the top layer of that side.

The file has the following format:
```
2
EDGE 2 25
x1 y1
x2 y2
EDGETOP 2 40
x1 y1
x2 y2
``` 
<img src="Box3D_dirich2.png" alt="Dirichlet test 01" width="700"/>

We can see that in the right side of the aquifer only the top row of elements has constant head equal to 40 m. 

# Side Boundaries variable (Test 3 : dir_bc03.npsat)
For the third example we will use the same boundaries as in the second example and change their values. Instead of using a uniform value along the boundary we will use variable interpolation functions.
* For the left boundary the head will vary also with depth.
* The the right boundary since only the top face is affected we will use a variable interpolation function along the x-y plane. All z nodes will have the same constant head value for the same x,y location
 The input file is almost identical to the previous example. This time the values are replaced with the files that contain the interpolation functions:
 
```
2
EDGE 2 box3d_leftv1.npsat
x1 y1
x2 y2
EDGETOP 2 box3d_rightv1.npsat
x1 y1
x2 y2
```

## Test 1 with multipolygon recharge
Using the same boundary conditions we will assign zone recharge. This is implemented using the `MULTIPOLY` keyword as follows
```
MULTIPOLY
Npoly 
Nverts CONST value
x y
.
.
.
```
Repeat `Npoly` times from Nverts. An example of the input file is the mult_const_rch.npsat




# Test 4 (dir_bc04.npsat)
A common case in groundwater hydrology is to assign specific values on vertices that correspond to domain outline and assume linear interpolation between the vertices. This can be handled using interpolation functions, however one would have to provide a different file for each segment.
In this example we assume that the constant hydraulic head is defined for the segments (2500, 0)-(5000, 0)-(5000 2500)
In this case instead creating two interpolation functions for each segment the file will have the following format:
```
1
EDGETOP 0 box3d_bnd_lines.npsat
```
Note that for this case we do not provide and points to describe the boundary. The input file contains the definition of the segments along with the values.




