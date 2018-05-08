import matplotlib.pyplot as plt
import random
import math
import numpy

# Domain outline
Dx = [0, 5000, 5000, 0, 0]
Dy = [0, 0, 5000, 5000, 0]
plt.plot(Dx, Dy, label='Outline')

# river segments
r1x = [1178, 2427, 1912, 2244, 3032]
r1y = [5000, 3568, 2877, 1808, 1200]
plt.plot(r1x, r1y, label='River 1')

r2x = [5000, 4052, 4329, 3200]
r2y = [3442, 2630, 1463, 1032]
plt.plot(r2x, r2y, label='River 2')

# Constant head boundary
Cdelta = 200
Cp = [3000, 1000]

cx = [Cp[0] - Cdelta, Cp[0] + Cdelta, Cp[0] + Cdelta, Cp[0] - Cdelta, Cp[0] - Cdelta]
cy = [Cp[1] - Cdelta, Cp[1] - Cdelta, Cp[1] + Cdelta, Cp[1] + Cdelta, Cp[1] - Cdelta]
plt.plot(cx, cy, label='Constant head')

# Recharge
N = 0.00018
Qrch = 5000*5000*N
print("Q Recharge = " + str(Qrch))

# Total stream recharge is 20% of the areal recharge
Qstream = Qrch * 0.2
print("Q Stream = " + str(Qstream))

Riv_len1 = 0.0
R_seg_len1 = []
for ii in range(0, len(r1x)-1):
    R_seg_len1.append(math.sqrt((r1x[ii+1] - r1x[ii])**2 + (r1y[ii+1] - r1y[ii])**2))
    Riv_len1 = Riv_len1 + R_seg_len1[-1]
print(Riv_len1)

Riv_len2 = 0.0
R_seg_len2 = []
for ii in range(0, len(r2x)-1):
    R_seg_len2.append(math.sqrt((r2x[ii+1] - r2x[ii])**2 + (r2y[ii+1] - r2y[ii])**2))
    Riv_len2 = Riv_len2 + R_seg_len2[-1]
print(Riv_len2)

# distribute according to river length
# Qstrm1 = Qstream * (Riv_len1 / (Riv_len1 + Riv_len2))
# Qstrm2 = Qstream * (Riv_len2 / (Riv_len1 + Riv_len2))\

# arbitrarily set 70% of the river to come from the 1st river
Qstrm1 = Qstream * 0.7
Qstrm2 = Qstream * 0.3

print("Qriver 1 = " + str(Qstrm1) + ' Qriver 2 = ' + str(Qstrm2))

Total_pumping = Qstrm1 + Qstrm2 + Qrch
print("Total pumping required " + str(Total_pumping))

# Generate as many wells as needed to equalize the pumping
Q_so_far = 0

wx = []
wy = []
Q = []
SL = []
Top = []
for ii in range(0, 200):
    x = random.uniform(300.0, 4700.0)
    y = random.uniform(300.0, 4700.0)

    if x > cx[0] and x < cx[1] and y > cy[0] and y < cy[2]:
        continue

    q = numpy.random.normal(150, 50)
    q = numpy.clip(q, 10, 250)

    sl = random.uniform(0.5, 1) * q
    t = random.uniform(0.1, 0.7)* sl

    well_exist = False
    if ii == 0:
        wx.append(x)
        wy.append(y)
    else:
        for iw in range(0, len(wx)):
            wdst = math.sqrt((x - wx[iw])**2 + (y - wy[iw])**2)
            # print(wdst)
            if wdst < 200:
                well_exist = True
                break

    if not well_exist:
        wx.append(x)
        wy.append(y)
        Q.append(q)
        SL.append(sl)
        Top.append(t)
        # print('Q = ' + str(q) + ', L = ' + str(sl) + ', T = ' + str(t) + ', B = ' +  str(sl + t))
        Q_so_far = Q_so_far + q

    if Q_so_far > Total_pumping:
        break

print(Q_so_far)
plt.plot(wx,wy, 'ro', label='Wells')

plt.show()

# -------------------PRINT FILES--------------------
# WELL file
f = open('box01_wells.npsat', 'w+')
f.write('%d\r\n' % len(wx))
Gelev = 50

for ii in range(0, len(wx)-1):
    f.write('%.2f\t%.2f\t%.2f\t%.2f\t%.2f\r\n' % (wx[ii], wy[ii], Gelev - Top[ii],  Gelev - Top[ii] -SL[ii], -Q[ii]))
f.close()

# RIVER file
f = open('box01_rivers.npsat', 'w+')
N_riv_seg = len(r1x) - 1 + len(r2x) - 1
f.write('%d\r\n' % N_riv_seg)

for ii in range(0, len(r1x)-1):
    Q_riv_seg = Qstrm1*(R_seg_len1[ii]/Riv_len1)/(Riv_len1*50.0)
    print(Qstrm1*(R_seg_len1[ii]/Riv_len1)/(Riv_len1*50.0))
    f.write('%.2f\t%.2f\t%.2f\t%.2f\t%.6f\t%.2f\r\n' %(r1x[ii], r1y[ii], r1x[ii+1], r1y[ii+1], Q_riv_seg, 50))

for ii in range(0, len(r2x)-1):
    Q_riv_seg = Qstrm2*(R_seg_len2[ii]/Riv_len2)/(Riv_len2*30.0)
    f.write('%.2f\t%.2f\t%.2f\t%.2f\t%.6f\t%.2f\r\n' %(r2x[ii], r2y[ii], r2x[ii+1], r2y[ii+1], Q_riv_seg, 30))

f.close()

