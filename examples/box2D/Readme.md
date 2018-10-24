# 2D Examples
---
## 1st Example (BOX2D_01)
### Domain
The domain of the 2D problem is a vertical cross section of a hypothetical aquifer 5 km long. 
The depth of the aquifer is approximately 300 m. The initial estimation of the top elevation is at 30 m elevation, and the bottom elevation is set to -270 m.
Initialy the domain was discretized into 10 columns and 4 layers.
<img src="box2d_01_init_mesh.png" alt="Initial Mesh" width="700"/>


### Boundary conditions

#### Constant head boundaries
* At x = 0 m and z = -270 m up to the free surface set H = 30 m.
* The free surface equal to H = 40 m for x = [3600 3700].
#### Stresses
* Groundwater recharge is considered constant and equal to 0.0004 m/day, 

* Pumping:
  	* Well 1, x = 2300 m, top = -30m, bottom = -200m, Q = 2.5 m^3/day
  	* Well 2, x = 4200 m, top = -10m, bottom = -50m, Q = 1.5 m^3/day

#### Aquifer properties
For this example we assume that the aquifer is isotropic with K =  10 m/day.

#### Solution
This is a trivial problem that can be solved rapidly even using one processor. and it is used as the firsth debugging test. The discretization and the sloution after 14 iterations is shown below:
<img src="box2d_01_ref_14.png" alt="After 14 refinements" width="700"/>

---

## 2nd Example (BOX2D_02)
### Domain
The domain of the second example is also a vertical cross section. The length of the aquifer is 9 km. The depth of the aquifer is variable. The bottom of the aquifer is defined as shown in the figure below. However what is shown as top is just an initial estimation of the free surface.
<img src="box2d_02_init_mesh.png" alt="Initial Mesh" width="800"/>

### Boundary conditions
#### Constant head boundaries
Two constant head boundaries are considered in this example. The hydraulic head on left side is set equal to 50 m and on the right side the hydraulic head is set equal to 30 m.

#### Stresses
* Groundwater recharge is uniform and equal to 0.12 m/day
* Pumping: For pumping we defined one well at 5950 m distance from the left side. The top of the screen is set to -250 m and the bottom of the screen at -350 m. The pumping rate is Q=1080/m^3/day

#### Aquifer properties
The hydraulic condactivity is uniform and isotrpoic equal to 13 m/day.

#### Solution 
The following figures show the free surface and the discretization after 2 and 7 iterations
<img src="box2d_02_ref_02.png" alt="After 14 refinements" width="700"/>
<img src="box2d_02_ref_07.png" alt="After 14 refinements" width="700"/>

---

## 3rd Example (BOX2D_03)
### Domain
Similar to the previous examples the domain is a vertical cross section. The length of the domain is 48 km and the depth is variable. The figure below shows the bottom of the aquifer and the initial estimation of the free surface. 

<img src="box2d_03_init_mesh.png" alt="Initial Mesh" width="800"/>

### Boundary conditions
#### Constant head boundaries
* Hydraulic head at the left side is set equal to 27 m
* Hydraulic head at the right side is set equal to 52 m

#### Stresses
* The groundwater recharge is variable as show in the figure
 <img src="rch_example3.png" alt="After 14 refinements" width="700"/>
 
* Pumping:
There are 30 wells in the aquifer with total pumping equal to 225 m^3/day.

#### Aquifer properties
In this example the aquifer is considered isotropic along x and y. Both horizontal and vertical hydraulic conductivity are defined as a stratified  interpolation functions. Over the hydraulic conductivities on the left side of the domain are higher compared to the values of the right side.

#### Solution 
The following figures show the free surface and the discretization after 2 and 8 refinement iterations. Due to the variability in the recharge alnog the the relatively low hydraulic conductivities in the right side the free surface creates many dips arounf the wells. Note however that these figures are exaggerated 30 times in the vertical direction
<img src="box2d_03_ref_02.png" alt="After 14 refinements" width="700"/>
<img src="box2d_03_ref_08.png" alt="After 14 refinements" width="700"/>
