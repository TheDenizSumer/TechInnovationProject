import math
from sympy import symbols, diff
import numpy as np


def F(transposed, origin):

    a1_2 = math.sqrt(((origin[1][0] - origin[0][0]) ** 2) + ((origin[1][1] - origin[0][1]) ** 2)) / 2
    a4_3 = math.sqrt(((origin[2][0] - origin[3][0]) ** 2) + ((origin[2][1] - origin[3][1]) ** 2)) / 2
    b4_1 = math.sqrt(((origin[3][1] - origin[0][1]) ** 2) + ((origin[3][0] - origin[0][0]) ** 2)) / 2
    b2_3 = math.sqrt(((origin[2][1] - origin[1][1]) ** 2) + ((origin[2][0] - origin[1][0]) ** 2)) / 2

    x, y = symbols('x y', real=True)

    u = (transposed[0][0]-origin[0][0])*(1/(4*a1_2*b4_1))*(x-origin[1][0])*(y-origin[3][1]) + (transposed[1][0]-origin[1][0])*(-1*(1/(4*a1_2*b2_3))*(x-origin[0][0])*(y-origin[2][1])) + (transposed[2][0]-origin[2][0])*(1/(4*a4_3*b2_3))*(x-origin[3][0])*(y-origin[1][1]) + (transposed[3][0]-origin[3][0])*(-1*(1/(4*a4_3*b4_1))*(x-origin[2][0])*(y-origin[0][1]))

    v = (transposed[0][1]-origin[0][1])*(1/(4*a1_2*b4_1))*(x-origin[1][0])*(y-origin[3][1]) + (transposed[1][1]-origin[1][1])*(-1*(1/(4*a1_2*b2_3))*(x-origin[0][0])*(y-origin[2][1])) + (transposed[2][1]-origin[2][1])*(1/(4*a4_3*b2_3))*(x-origin[3][0])*(y-origin[1][1]) + (transposed[3][1]-origin[3][1])*(-1*(1/(4*a4_3*b4_1))*(x-origin[2][0])*(y-origin[0][1]))

    #u(x, y) = function of x & y returns zrelative x displacment in element
    #v(x, y) = function of x & y returns relative x displacment in element

    xx = float(diff(u, x))
    xy = float(diff(u, y))
    yx = float(diff(v, x))
    yy = float(diff(v, y))

    F = np.matrix([[xx, xy], [yx, yy]])
    
    return F


#
#square_undeformed_green = [
#[1, 1],
#[5, 1],
#[5, 5],
#[1, 5]
#]
#
#
#square_deformed_red = [
#[1, 1],
#[9, 1],
#[9, 9],
#[1, 9]
#]
#
#square_deformed_orange = [
#[1, 1],
#[5, 1],
#[7, 5],
#[3, 5]
#]
#
#square_deformed_blue = [
#[6, 1],
#[10, 1],
#[10, 5],
#[6, 5]
#]
#
#square_deformed_purple = [
#[10 + math.sqrt(8), 1],
#[10 + 2*math.sqrt(8), 1 + math.sqrt(8)],
#[10 + math.sqrt(8), 1 + 2*math.sqrt(8)],
#[10, 1 + math.sqrt(8)]
#]
#
#square_deformed = [
#[8, 6],
#[10, 8],
#[8, 10],
#[6, 8]
#]
#
#
#
#
#print(f'Blue square : \n{F(square_deformed_blue, square_undeformed_green)}')
#print()
#print(f'Red square : \n{F(square_deformed_red, square_undeformed_green)}')
#print()
#
#print(f'Orange square : \n{F(square_deformed_orange, square_undeformed_green)}')
#
#print()
#print(f'Purple square : \n{F(square_deformed_purple, square_undeformed_green)}')
#