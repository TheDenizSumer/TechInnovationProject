##Masking to seperate the place makers from the backrouin#
##centroid tracking to find the placemakers withing the pixel coordinates of the image. Placemakers must not be too large in order to not interfere with the digital image tracking of the overall softwar#
## list of coordinates of the centroids to see if the program can track relaibaly throught the entirity of a vide#
## start the process of appliying concepts of strain and other things to make sure that the overall concept isn't flawed if it is make sure that there sia  developed plan ready to deploy asa#
## once we can determine that the square tracking works, apply square tracking to the list of coordinates of the centroid#
##!/usr/bin/env python
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.##
## Copyright 2022 Jorgen Bergstrom
## See also code from http://compmech.lab.asu.edu/codes.php
#import numpy as np
#import math
#from matplotlib import pyplot as plt
#def shape(xi)#	x,y = tuple(xi#	N = [(1.0-x)*(1.0-y), (1.0+x)*(1.0-y), (1.0+x)*(1.0+y), (1.0-x)*(1.0+y)#	return 0.25*np.array(N)
#def gradshape(xi)#	x,y = tuple(xi#	dN = [[-(1.0-y),  (1.0-y), (1.0+y), -(1.0+y)]#		  [-(1.0-x), -(1.0+x), (1.0+x),  (1.0-x)]#	return 0.25*np.array(dN)
################################
#print('create mesh')
## input
#mesh_ex = 9
#mesh_ey = 49
#mesh_lx = 10.0
#mesh_ly = 50.0
## derived
#mesh_nx      = mesh_ex + 1
#mesh_ny      = mesh_ey + 1
#num_nodes    = mesh_nx * mesh_ny
#num_elements = mesh_ex * mesh_ey
#mesh_hx      = mesh_lx / mesh_ex
#mesh_hy      = mesh_ly / mesh_ey
#nodes = []
#for y in np.linspace(0.0, mesh_ly, mesh_ny)#	for x in np.linspace(0.0, mesh_lx, mesh_nx)#		nodes.append([x,y])
#nodes = np.array(nodes)
#conn = []
#for j in range(mesh_ey)#	for i in range(mesh_ex)#		n0 = i + j*mesh_n#		conn.append([n0, n0 + 1, n0 + 1 + mesh_nx, n0 + mesh_nx])
################################
#print ('material model - plane strain')
#E = 100.0
#v = 0.48
#C = E/(1.0+v)/(1.0-2.0*v) * np.array([[1.0-v,     v,     0.0]#								      [    v, 1.0-v,     0.0]#								      [  0.0,   0.0,   0.5-v]])
################################
#print('create global stiffness matrix')
#K = np.zeros((2*num_nodes, 2*num_nodes))
#q4 = [[x/math.sqrt(3.0),y/math.sqrt(3.0)] for y in [-1.0,1.0] for x in [-1.0,1.0]]
#B = np.zeros((3,8))
#for c in conn#	xIe = nodes[c,:#	Ke = np.zeros((8,8)#	for q in q4#		dN = gradshape(q#		J  = np.dot(dN, xIe).#		dN = np.dot(np.linalg.inv(J), dN#		B[0,0::2] = dN[0,:#		B[1,1::2] = dN[1,:#		B[2,0::2] = dN[1,:#		B[2,1::2] = dN[0,:#		Ke += np.dot(np.dot(B.T,C),B) * np.linalg.det(J#	for i,I in enumerate(c)#		for j,J in enumerate(c)#			K[2*I,2*J]     += Ke[2*i,2*j#			K[2*I+1,2*J]   += Ke[2*i+1,2*j#			K[2*I+1,2*J+1] += Ke[2*i+1,2*j+1#			K[2*I,2*J+1]   += Ke[2*i,2*j+1]
################################
#print('assign nodal forces and boundary conditions')
#f = np.zeros((2*num_nodes))
#for i in range(num_nodes)#	if nodes[i,1] == 0.0#		K[2*i,:]     = 0.#		K[2*i+1,:]   = 0.#		K[2*i,2*i]   = 1.#		K[2*i+1,2*i+1] = 1.#	if nodes[i,1] == mesh_ly#		x = nodes[i,0#		f[2*i+1] = 20.#		if x == 0.0 or x == mesh_lx#			f[2*i+1] *= 0.5
################################
#print('solving linear system')
#u = np.linalg.solve(K, f)
#print('max u=', max(u))
################################
#print('plotting displacement')
#ux = np.reshape(u[0::2], (mesh_ny,mesh_nx))
#uy = np.reshape(u[1::2], (mesh_ny,mesh_nx))
#xvec = []
#yvec = []
#res  = []
#for i in range(mesh_nx):
#    for j in range(mesh_ny):
#        xvec.append(i*mesh_hx + ux[j,i])
#        yvec.append(j*mesh_hy + uy[j,i])
#        res.append(uy[j,i])
#t = plt.tricontourf(xvec, yvec, res, levels=14, cmap=plt.cm.jet)
#plt.scatter(xvec, yvec, marker='o', c='b', s=2)
#plt.grid()
#plt.colorbar(t)
#plt.axis('equal')
#plt.show()
import numpy as np



def positions_at_t(global_coords, t):
    gc = np.array(global_coords)
    new_coords = gc[0] + t*(gc[1] - gc[0])
    return list(new_coords)




#x1 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0], [0.0, 3.0, 0.0], [1.0, 3.0, 0.0], [2.0, 3.0, 0.0]]
##
#x2 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0], [0.0, 4.0, 0.0], [1.0, 4.0, 0.0], [2.0, 4.0, 0.0], [0.0, 6.0, 0.0], [1.0, 6.0, 0.0], [2.0, 6.0, 0.0]]

#x1 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
#x2 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.5, 0.0, 1.5],[0.0, 1.5, 1.5], [2.0, 2.0, 2.0]]


#global_coords[0] = 
#global_coords[1] = x2
'''
x1 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

x2 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.2], [1.0, 0.0, 1.2],[0.0, 1.0, 1.2], [1.0, 1.0, 1.2]]




#x1 = [[0.0, 0.0], [1.0, 1.0]]
#x2 = [[0.0, 0.0], [2.0, 2.0]]

dx = x1
dX = x2

F = np.linalg.solve(dX,dx)


print(F)

C = F.T @ F

E = .5*(C-np.identity(3))

print(E)

from sympy import *
import sympy as sp
theta = sp.symbols("\u03B8")
X1,X2,X3=sp.symbols("X1 X2 X3")
x1,x2,x3=sp.symbols("x1 x2 x3")
Qx = Matrix([[1, 0, 0], 
              [0, sp.cos(theta),-sp.sin(theta)], 
              [0, sp.sin(theta),sp.cos(theta)]])
# y rotation matrix
Qy = Matrix([[sp.cos(theta), 0, sp.sin(theta)], 
              [0, 1, 0], 
              [-sp.sin(theta), 0, sp.cos(theta)]])
# z rotation matrix
Qz = Matrix([[sp.cos(theta), -sp.sin(theta), 0],
              [sp.sin(theta), sp.cos(theta), 0], 
              [0, 0, 1]])
Qx = Qx.subs({theta:30*sp.pi/180})
Qy = Qy.subs({theta:20*sp.pi/180})
Qz = Qz.subs({theta:20*sp.pi/180})
F = Qx*Qy*Qz
display("F =",F)
F = simplify(F).evalf()
display("F =",F)
X=Matrix([X1,X2,X3])
x=Matrix([x1,x2,x3])
display("X =",X)
display("x =",x)
display("x =F.X =",F*x)
x = F*X
u = x-X
display("u = x-X =",u)
gradu = Matrix([[diff(i,j) for j in X] for i in u])
display("\u2207u =",gradu)
small_strain = (gradu+gradu.T)/2
display("\u03B5_small =",small_strain)
green_strain = (gradu+gradu.T+gradu.T*gradu)/2
display("\u03B5_Green =",green_strain)'''