import math
from sympy import symbols, diff
import numpy as np


#a = 1/2 of square starting length x
#b = 1/2 of square starting length y
#
# 4  ______________  3
#   |              |
#   |              |
#   |              |
#   |              |
# 1 |______________| 2
#




# ordered in [x1, y1], [x2, y2], [x3, y3], [x4, y4]
square_undeformed = [
[1, 1],
[5, 1],
[5, 5],
[1, 5]
]


#square_deformed = [
#[6, 1],
#[10, 1],
#[10, 5],
#[6, 5]
#] #translated 5 to the right

#square_deformed = [
#[10 + math.sqrt(8), 1],
#[10 + 2*math.sqrt(8), 1 + math.sqrt(8)],
#[10 + math.sqrt(8), 1 + 2*math.sqrt(8)],
#[10, 1 + math.sqrt(8)]
#]

#square_deformed = [
#[1, 1],
#[9, 1],
#[9, 9],
#[1, 9]
#] #Simple Deformation

square_deformed = [
[1, 1],
[5, 1],
[7, 5],
[3, 5]
] #shear

#square_deformed = [
#[8, 1],
#[10, 3],
#[8, 5],
#[6, 3]
#]

#square_deformed = [
#[8, 6],
#[10, 8],
#[8, 10],
#[6, 8]
#]



def square_side_calc(coords):
    a1_2 = math.sqrt(((coords[1][0] - coords[0][0]) ** 2) + ((coords[1][1] - coords[0][1]) ** 2)) / 2
    a4_3 = math.sqrt(((coords[2][0] - coords[3][0]) ** 2) + ((coords[2][1] - coords[3][1]) ** 2)) / 2
    b4_1 = math.sqrt(((coords[3][1] - coords[0][1]) ** 2) + ((coords[3][0] - coords[0][0]) ** 2)) / 2
    b2_3 = math.sqrt(((coords[2][1] - coords[1][1]) ** 2) + ((coords[2][0] - coords[1][0]) ** 2)) / 2
    return a1_2, a4_3, b4_1, b2_3


origin = square_undeformed
transposed = square_deformed

a1_2, a4_3, b4_1, b2_3 = square_side_calc(origin)


#def u(x,y):
#    return (transposed[0][0]-origin[0][0])*(1/(4*a1_2*b4_1))*(x-origin[1][0])*(y-origin[3][1]) + (transposed[1][0]-origin[1][0])(-1*(1/(4*a1_2*b2_3))*(x-origin[0][0])*(y-origin[2][1])) + (transposed[2][0]-origin[2][0])(1/(4*a4_3*b2_3))*(x-origin[3][0])*(y-origin[1][1]) + (transposed[3][0]-origin[3][0])(-1*(1/(4*a4_3*b4_1))*(x-origin[2][0])*(y-origin[0][1]))
#def v(x,y):
#    return (transposed[0][1]-origin[0][1])*(1/(4*a1_2*b4_1))*(x-origin[1][0])*(y-origin[3][1]) + (transposed[1][1]-origin[1][1])(-1*(1/(4*a1_2*b2_3))*(x-origin[0][0])*(y-origin[2][1])) + (transposed[2][1]-origin[2][1])(1/(4*a4_3*b2_3))*(x-origin[3][0])*(y-origin[1][1]) + (transposed[3][1]-origin[3][1])(-1*(1/(4*a4_3*b4_1))*(x-origin[2][0])*(y-origin[0][1]))   


# + (transposed[1][0]-origin[1][0])(-1*(1/(4*a1_2*b2_3))*(x-origin[0][0])*(y-origin[2][1])) + (transposed[2][0]-origin[2][0])(1/(4*a4_3*b2_3))*(x-origin[3][0])*(y-origin[1][1]) + (transposed[3][0]-origin[3][0])(-1*(1/(4*a4_3*b4_1))*(x-origin[2][0])*(y-origin[0][1]))


x, y = symbols('x y', real=True)
    
u = (transposed[0][0]-origin[0][0])*(1/(4*a1_2*b4_1))*(x-origin[1][0])*(y-origin[3][1]) + (transposed[1][0]-origin[1][0])*(-1*(1/(4*a1_2*b2_3))*(x-origin[0][0])*(y-origin[2][1])) + (transposed[2][0]-origin[2][0])*(1/(4*a4_3*b2_3))*(x-origin[3][0])*(y-origin[1][1]) + (transposed[3][0]-origin[3][0])*(-1*(1/(4*a4_3*b4_1))*(x-origin[2][0])*(y-origin[0][1]))

v = (transposed[0][1]-origin[0][1])*(1/(4*a1_2*b4_1))*(x-origin[1][0])*(y-origin[3][1]) + (transposed[1][1]-origin[1][1])*(-1*(1/(4*a1_2*b2_3))*(x-origin[0][0])*(y-origin[2][1])) + (transposed[2][1]-origin[2][1])*(1/(4*a4_3*b2_3))*(x-origin[3][0])*(y-origin[1][1]) + (transposed[3][1]-origin[3][1])*(-1*(1/(4*a4_3*b4_1))*(x-origin[2][0])*(y-origin[0][1]))

#print(diff(u, x), diff(u, y))
#print(diff(v, x), diff(v, y))



xx = float(diff(u, x))
xy = float(diff(u, y))
yx = float(diff(v, x))
yy = float(diff(v, y))

#print(f'xx: {xx} xy: {xy} yx: {yx} yy: {yy}')

#F = np.matrix([[1, 0.495, 0.5], [-0.333, 1, -0.247], [0.959, 0, 1.5]])
F = np.matrix([[xx, xy], [yx, yy]])
#F = np.matrix([[xx, xy], [xy, yy]])


print(F)


'''C = np.matrix.transpose(F)*F # aka C = F.T @ F
print(C)
E = .5*(C-np.identity(3))
print(E)'''