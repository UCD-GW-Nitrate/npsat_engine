node = hou.pwd()
geo = node.geometry()

import csv
import os

# +++++++++ INSTRUCTIONS++++++++++++
# Copy the content of this scipt into
# a python houdini node
#++++++++++++++++++++++++++++++++++++
# Add code to modify contents of geo.
# Use drop down menu to select examples.


geo.addAttrib(hou.attribType.Prim, "Proc", 0)

iter = node.parm('Iterparm').eval()
dim = node.parm('Dimparm').eval()
nproc = node.parm('nprocparm').eval()
frst_proc = node.parm('from_proc_parm').eval()


def main():
    
    for i in range(frst_proc, nproc+1  ,1):
        filename = '/home/giorgk/CODES/MoveMeshTest/build/mesh_PrintanimBefore_' + str(iter) + '_000' + str(i) + '.dat'
    
        with open(filename, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                if dim == 2:
                    poly = geo.createPolygon()
                    poly.setIsClosed(1)
                    poly.setAttribValue("Proc", i)
                    
                    point1=geo.createPoint()
        
                    point1.setPosition(hou.Vector3(float(row[0]), float(row[1]), float(row[2])))
                    poly.addVertex(point1)
                    
                    point2=geo.createPoint()
                    point2.setPosition(hou.Vector3(float(row[3]), float(row[4]), float(row[5])))
                    poly.addVertex(point2)
                    
                    point4=geo.createPoint()
                    point4.setPosition(hou.Vector3(float(row[9]),float(row[10]),float(row[11])))
                    poly.addVertex(point4)
                    
                    point3=geo.createPoint()
                    point3.setPosition(hou.Vector3(float(row[6]),float(row[7]),float(row[8])))
                    poly.addVertex(point3)
                    
                if dim == 3:
                    pnt_list = [ [0, 1, 3, 2], [4, 5, 7, 6], [1, 3, 7, 5], [0, 2, 6, 4], [0, 1, 5, 4], [2, 3, 7, 6] ]
                    pnts = list()
                    
                    for ii in range(0,22,3):
                        point=geo.createPoint()
                        point.setPosition(hou.Vector3(float(row[ii]), float(row[ii+1]), float(row[ii+2])))
                        pnts.append(point)
                        
                    for lst in pnt_list:
                        poly = geo.createPolygon()
                        poly.setIsClosed(1)
                        poly.setAttribValue("Proc", i)
                        for jj in lst:
                            poly.addVertex(pnts[jj])
                            
             
main()



#poly = geo.createPolygon()
#poly.setIsClosed(0) 
#for position in (0,0,0), (1,0,0):
#    point = geo.createPoint()
#    point.setPosition(position)
#    poly.addVertex(point)
#
#curve = geo.createNURBSCurve(4)
#i = 0
#for vertex in curve.vertices():
#        vertex.point().setPosition((i, i % 3, 0))
#        i = i + 1 

