import math


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