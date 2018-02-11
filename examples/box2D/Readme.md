# 2D Example
## Aquifer description
The domain of the 2D problem is a rectangular aquifer with x dimension equal to 5000 m.
The initial top elevation is constant at 30 m, while the bottom elevation is set to -270 m.
Using the same aquifer geometry we tested several cases.

## BOX 01
The case of Box01 has the following properties:
* Constant head boundaries: 
  * At x = 0 m -> H = 30 m.
  * At x = [3600 3700] -> H = 40 m.
* Groundwater recharge is considered constant and equal to 0.0004 m/day, 
which means that the aquifer receives daily 5000(m)*0.0004 (m/day) = 2 m^2/day. 
If we assume that the aquifer thickness along the y is 1 m then we can convert the amount to 3m^3/day
* Last we assume that two wells exist with the following characteristics:
  * Well 1, x = 2300 m, top = -30m, bottom = -200m, Q = 1.5 m^3/day
  * Well 2, x = 4200 m, top = -10m, bottom = -50m, Q = 0.5 m^3/day
