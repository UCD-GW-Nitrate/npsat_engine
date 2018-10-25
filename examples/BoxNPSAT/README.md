# BOX NPSAT
This example shows how to build a hypothetical diffuse pollution example and all the steps to solve the flow run particle tracking and compute the Unit Response Functions.

## Aquifer description
- The domain is a rectangular area with 5 km lenth. 
- The boundaries that surround the aquifer are treated as no flow boundaries.
- There is a small rectangle highlighted with orange line that is constant head equal to 30 m. 
- The groundwater recharge is uniform and equal to 0.00018 m/day.

- There are also two rivers that recharge the area with 894 m^3/day.
- There are 18 wells that pump 5365 m^3/day approximately. This is almost eqyal to the volume of incoming water from the groundwater recharge and streams, therefore the groundwater budget is approximately 0.

<img src="aquifer_domain.png" alt="Initial Mesh" width="900"/>

## Simulate Flow only
While it is possible to solve the calculate the flow field and do particle tracking at one run, we will highlight the options here to do this separately. 
To deactivate particle tracking set the following option in the parameters file to 0
```
set a Enable particle tracking = 0

```
This option can be found under **subsection I. Particle tracking**


