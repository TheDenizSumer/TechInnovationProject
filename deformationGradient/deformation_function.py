'''import math


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
#[1, 1],
#[9, 1],
#[9, 9],
#[1, 9]
#]

#square_deformed = [
#[8, 1],
#[10, 3],
#[8, 5],
#[6, 3]
#]

square_deformed = [
[8, 6],
[10, 8],
[8, 10],
[6, 8]
]



def deformation(deformed, not_deformed):
    deformation = []
    for i in range(4):
        deformation.append([deformed[i][0] - not_deformed[i][0], deformed[i][1] - not_deformed[i][1]])
    return deformation



def square_side_calc(coords):
    a1_2 = math.sqrt(((coords[1][0] - coords[0][0]) ** 2) + ((coords[1][1] - coords[0][1]) ** 2)) / 2
    a4_3 = math.sqrt(((coords[2][0] - coords[3][0]) ** 2) + ((coords[2][1] - coords[3][1]) ** 2)) / 2
    b4_1 = math.sqrt(((coords[3][1] - coords[0][1]) ** 2) + ((coords[3][0] - coords[0][0]) ** 2)) / 2
    b2_3 = math.sqrt(((coords[2][1] - coords[1][1]) ** 2) + ((coords[2][0] - coords[1][0]) ** 2)) / 2
    return a1_2, a4_3, b4_1, b2_3

print(square_side_calc(square_undeformed))

a = 2
b = 2


def N1(a, b, x, y, x2, y4):
    return (1/(4*a*b))*(x-x2)*(y-y4)


def N2(a, b, x, y, x1, y3):
    return -1*(1/(4*a*b))*(x-x1)*(y-y3)


def N3(a, b, x, y, x4, y2):
    return (1/(4*a*b))*(x-x4)*(y-y2)


def N4(a, b, x, y, x3, y1):
    return -1*(1/(4*a*b))*(x-x3)*(y-y1)




def u(x, y, a, b, undeformed, deformed):
    Nodes = [
       
        N1(a, b, x, y, undeformed[1][0], undeformed[3][1]),
        N2(a, b, x, y, undeformed[0][0], undeformed[2][1]),
        N3(a, b, x, y, undeformed[3][0], undeformed[1][1]),
        N4(a, b, x, y, undeformed[2][0], undeformed[0][1])]
    delta = deformation(deformed, undeformed)
    answer = 0
    for i in range(4):
        answer += delta[i][0]*Nodes[i]
    return answer




def v(x, y, a, b, undeformed, deformed):
    Nodes = [
        N1(a, b, x, y, undeformed[1][0], undeformed[3][1]),
        N2(a, b, x, y, undeformed[0][0], undeformed[2][1]),
        N3(a, b, x, y, undeformed[3][0], undeformed[1][1]),
        N4(a, b, x, y, undeformed[2][0], undeformed[0][1])]
    delta = deformation(deformed, undeformed)
    answer = 0
    for i in range(4):
        answer += delta[i][1]*Nodes[i]
    return answer


print(u(5, 3, a, b, square_undeformed, square_deformed))
print(v(5, 3, a, b, square_undeformed, square_deformed))


'''

#grouped:



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

    #tx, ty = centroid(transposed[0], transposed[1], transposed[2], transposed[3])
    tx, ty = centroid(origin[0], origin[1], origin[2], origin[3])
    #ty = ty * .25
    print(u)
    print(v)
    print(diff(u, x))
    print(diff(u, y))
    print(diff(v, x))
    print(diff(v, y))

    xx = float(diff(u, x).replace(y, ty))
    xy = float(diff(u, y).replace(x, tx))
    yx = float(diff(v, x).replace(y, ty))
    yy = float(diff(v, y).replace(x, tx))

    F = np.matrix([[xx, xy], [yx, yy]])
    
    #xx = float(diff(u, x).replace(y, Ty))
    #xy = float(diff(u, y).replace(x, Tx))
    #yx = float(diff(v, x).replace(y, Ty))
    #yy = float(diff(v, y).replace(x, Tx))
    #FT = np.matrix([[xx, xy], [yx, yy]])
    return F, [tx, ty]


X = [
    [896, 386], 
    [947, 383], 
    [947, 330], 
    [897, 330]
    ]
x = [
    [896, 387], 
    [947, 384], 
    [946, 331], 
    [897, 332]
    ]



square_undeformed_green = [
[1, 1],
[5, 1],
[5, 5],
[1, 5]
]

square_undeformed_greennnnnnn = [
[1, 1],
[5, 1],
[5, 5],
[2, 2]
]
square_deformed_red = [
[1, 1],
[9, 1],
[9, 9],
[1, 9]
]

square_deformed_orange = [
[1, 1],
[5, 1],
[7, 5],
[3, 5]
]

square_deformed_blue = [
[6, 1],
[10, 1],
[10, 5],
[6, 5]
]

square_deformed_purple = [
[10 + math.sqrt(8), 1],
[10 + 2*math.sqrt(8), 1 + math.sqrt(8)],
[10 + math.sqrt(8), 1 + 2*math.sqrt(8)],
[10, 1 + math.sqrt(8)]
]

square_deformed = [
[8, 6],
[10, 8],
[8, 10],
[6, 8]
]




print(f'Blue square : \n{F(square_deformed_blue, square_undeformed_green)}')
print()
print(f'Red square : \n{F(square_deformed_red, square_undeformed_green)}')
print()

print(f'Orange square : \n{F(square_deformed_orange, square_undeformed_green)}')

print()
print(f'Purple square : \n{F(square_deformed_purple, square_undeformed_green)}')

print()
print(f'sdjdnkajs square : \n{F(square_undeformed_green, square_deformed_red)}')

'''X = [
    [896, 386], 
    [947, 383], 
    [947, 330], 
    [897, 330]
    ]#transposed
x = [
    [896, 387], 
    [947, 384], 
    [946, 331], 
    [897, 332]
    ]#transposed'''

'''X = [
    [896, 386], 
    [947, 383], 
    [947, 330], 
    [897, 330]
    ]#transposed
x = [
    [896, 387], 
    [947, 384], 
    [947, 321], 
    [897, 321]
    ]#transposed'''

X = [
    [896, 386], 
    [947, 383], 
    [947, 330], 
    [897, 330]
    ]#non-deformed
x = [
    [896, 386], 
    [937, 383], 
    [937, 330], 
    [897, 330]
    ]#deformed

print(F(x, X))

#xx: -
#xy: +
#yx: -
#yy: -