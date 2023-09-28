import math
from sympy import symbols, diff
import numpy as np

def centroid(p1, p2, p3, p4):
    x = (p1[0]+p2[0]+p3[0]+p4[0])/4
    y = (p1[1]+p2[1]+p3[1]+p4[1])/4
    return x, y

def F(transposed, origin):
    a1_2 = math.sqrt(((origin[1][0] - origin[0][0]) ** 2) + ((origin[1][1] - origin[0][1]) ** 2)) / 2
    a4_3 = math.sqrt(((origin[2][0] - origin[3][0]) ** 2) + ((origin[2][1] - origin[3][1]) ** 2)) / 2
    b4_1 = math.sqrt(((origin[3][1] - origin[0][1]) ** 2) + ((origin[3][0] - origin[0][0]) ** 2)) / 2
    b2_3 = math.sqrt(((origin[2][1] - origin[1][1]) ** 2) + ((origin[2][0] - origin[1][0]) ** 2)) / 2

    x, y = symbols('x y', real=True)

    u = (transposed[0][0]-origin[0][0])*(1/(4*a1_2*b4_1))*(x-origin[1][0])*(y-origin[3][1]) + (transposed[1][0]-origin[1][0])*(-1*(1/(4*a1_2*b2_3))*(x-origin[0][0])*(y-origin[2][1])) + (transposed[2][0]-origin[2][0])*(1/(4*a4_3*b2_3))*(x-origin[3][0])*(y-origin[1][1]) + (transposed[3][0]-origin[3][0])*(-1*(1/(4*a4_3*b4_1))*(x-origin[2][0])*(y-origin[0][1]))

    v = (transposed[0][1]-origin[0][1])*(1/(4*a1_2*b4_1))*(x-origin[1][0])*(y-origin[3][1]) + (transposed[1][1]-origin[1][1])*(-1*(1/(4*a1_2*b2_3))*(x-origin[0][0])*(y-origin[2][1])) + (transposed[2][1]-origin[2][1])*(1/(4*a4_3*b2_3))*(x-origin[3][0])*(y-origin[1][1]) + (transposed[3][1]-origin[3][1])*(-1*(1/(4*a4_3*b4_1))*(x-origin[2][0])*(y-origin[0][1]))

    #u(x, y) = function of x & y returns relative x displacment in element
    #v(x, y) = function of x & y returns relative y displacment in element

    tx, ty = centroid(origin[0], origin[1], origin[2], origin[3])

    xx = float(diff(u, x).replace(y, ty))
    xy = float(diff(u, y).replace(x, tx))
    yx = float(diff(v, x).replace(y, ty))
    yy = float(diff(v, y).replace(x, tx))

    #F = np.matrix([[xx, xy], [yx, yy]])
    F = [[xx, xy], [yx, yy]]
    return F, [tx, ty]