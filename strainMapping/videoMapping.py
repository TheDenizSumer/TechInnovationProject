#contour
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
import matplotlib

#DefGrad
import math
from sympy import symbols, diff
import numpy as np

#main
import math
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import math
from statistics import mode

def Calc_centroid(p1, p2, p3, p4):
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
    #tx, ty = Calc_centroid(transposed[0], transposed[1], transposed[2], transposed[3])
    tx, ty = Calc_centroid(origin[0], origin[1], origin[2], origin[3])

    xx = float(diff(u, x).replace(y, ty))
    xy = float(diff(u, y).replace(x, tx))
    yx = float(diff(v, x).replace(y, ty))
    yy = float(diff(v, y).replace(x, tx))

    #F = np.matrix([[xx, xy], [yx, yy]])
    F = [[xx, xy], [yx, yy]]
    return F, [tx, ty]

def getColorMask(img):
        lower = np.array([ 9, 161, 38] )
        upper = np.array([ 62, 255,  83])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        out  = cv2.inRange(hsv, lower, upper)
        return out

def dilate(out):
    kernel=np.ones((3,3),np.uint8)
    dilated=cv2.dilate(out,kernel,iterations=3)
    return dilated

def get_contours(dilated_image):
    contours,_ = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours=[cv2.boundingRect(cnt) for cnt in contours]
    return contours

def distance(x, y, px, py):
    xdiff = px-x
    ydiff = py-y
    csq = xdiff**2 + ydiff**2
    return math.sqrt(csq)

def calc_coordinates(video, frame_cap=0, remove_frames=True):
    cap = cv2.VideoCapture(video)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)

    if (cap.isOpened()== False):
        print("Error opening video stream or file")
        raise "Error opening video stream or file"

    #adding frames of the video to an array called images
    images = []
    frames = 0
    countnum = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            images.append(frame)
            frames += 1
            countnum.append(len(get_contours(dilate(getColorMask(frame)))))
            if frame_cap != 0 and frames == frame_cap:
                break
        else:
            break
    if len(images) == frames:
        print('Images_Processed')
    else:
        print(f'Images: {len(images)}')
        print(f'Frames: {frames}')
        raise f"Frames returned and frames displayed aren't the same. (Images Captured{len(images)} Frames Displayed{frames})"

    #processing images to extract coordinates and countour information
    information = []
    dot_num = mode(countnum)
    print(dot_num)
    for image in images:
        contours = get_contours(dilate(getColorMask(image)))
        if len(contours) != dot_num and remove_frames:
            continue
        info = []
        cont = []
        init_points = 0
        for cnt in contours:
            x,y,w,h=cnt
            cX, cY = x+int(w/2), y+int(h/2)
            cont.append([init_points, cX, cY])
            init_points += 1

        if information != []:
            i = -1
            try:
                while len(information[i]) != 27:
                    i -= 1
            except:
                i = -1
            prev_cont = information[i]
            for point in cont:
                contX = point[1]
                contY = point[2]
                smallest_n, smallest_dist = prev_cont[0][0], distance(contX, contY, prev_cont[0][1], prev_cont[0][2])
                for prev_point in prev_cont:
                    dist = distance(contX, contY, prev_point[1], prev_point[2])
                    if dist < smallest_dist:
                        smallest_n, smallest_dist = prev_point[0], dist #distance calculated here
                info.append([smallest_n, contX, contY])
                info.sort()
            information.append(info)
        else:
            information.append(cont)


    return information, images

def elements(T, NT, squares, direction=None):
    def_element = []
    centroids = []
    for element in squares:
        #et = [T[element[0]], T[element[1]], T[element[2]], T[element[3]]]
        #e = [NT[element[0]], NT[element[1]], NT[element[2]], NT[element[3]]]
        et = [T[element[0]][1:], T[element[1]][1:], T[element[2]][1:], T[element[3]][1:]]

        e = [NT[element[0]][1:], NT[element[1]][1:], NT[element[2]][1:], NT[element[3]][1:]]

        x, y = F(et, e)
        def_element.append(x)
        centroids.append(y)
    return def_element, centroids

coordinates = [
    [0, 1, 1],
    [1, 3, 1],
    [2, 5, 1],
    [3, 7, 1],
    [4, 1, 3],
    [5, 3, 3],
    [6, 5, 3],
    [7, 7, 3],
    [8, 1, 5],
    [9, 3, 5],
    [10, 5, 5],
    [11, 7, 5],
    [12, 1, 7],
    [13, 3, 7],
    [14, 5, 7],
    [15, 7, 7]
]

#top left corner stretched
coordinates_TL = [
    [0, 1, 1],
    [1, 3, 1],
    [2, 5, 1],
    [3, 7, 1],
    [4, 1, 3],
    [5, 3, 3],
    [6, 5, 3],
    [7, 7, 3],
    [8, 1, 5],
    [9, 3, 5],
    [10, 5, 5],
    [11, 7, 5],
    [12, 1, 8],
    [13, 3, 7.5],
    [14, 5, 7],
    [15, 7, 7]
]

coordinates_10 = [
    [0, 1, 1],
    [1, 3, 1],
    [2, 5, 1],
    [3, 7, 1],
    [4, 1, 3.2],
    [5, 3, 3.2],
    [6, 5, 3.2],
    [7, 7, 3.2],
    [8, 1, 5.4],
    [9, 3, 5.4],
    [10, 5, 5.4],
    [11, 7, 5.4],
    [12, 1, 7.6],
    [13, 3, 7.6],
    [14, 5, 7.6],
    [15, 7, 7.6]
]

coordinates_20 = [
    [0, 1, 1],
    [1, 3, 1],
    [2, 5, 1],
    [3, 7, 1],
    [4, 1, 3.4],
    [5, 3, 3.4],
    [6, 5, 3.4],
    [7, 7, 3.4],
    [8, 1, 5.8],
    [9, 3, 5.8],
    [10, 5, 5.8],
    [11, 7, 5.8],
    [12, 1, 8.2],
    [13, 3, 8.2],
    [14, 5, 8.2],
    [15, 7, 8.2]
]

coordinates_exp = [
    [0, 1, 1],
    [1, 3, 0.7],
    [2, 5, 0.75],
    [3, 7.3, 0.75],
    [4, .3, 2.98],
    [5, 2.7, 2.7],
    [6, 5.2, 2.62],
    [7, 7.43, 3],
    [8, 0, 5],
    [9, 2.75, 5.21],
    [10, 5.2, 5.31],
    [11, 7.54, 5],
    [12, 1.16, 6.72],
    [13, 3, 7.5],
    [14, 5, 7.5],
    [15, 7, 7]
]


#order -----> acending ^

squares = [
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [4, 5, 9, 8],
    [5, 6, 10, 9],
    [6, 7, 11, 10],
    [8, 9, 13, 12],
    [9, 10, 14, 13],
    [10, 11, 15, 14],
]
'''
for i in coordinates_exp:
    print(f'[{i[1]}, {i[2]}],')'''

coordinates = [coordinates, coordinates_10, coordinates_20]



'''#NonAcculmulative
Elements = [] # frames, e1 e2 e3 e4 e5... e27, [xx xy], [yx yy]
Centroids = [] # frames, e1 e2 e3 e4 e5... e27, x y

for frame in range(1, len(coordinates)):
    print(frame)
    element, centroid = elements(coordinates[frame], coordinates[frame-1], squares)
    Elements.append(element)
    Centroids.append(centroid)'''


#Acculmulative

Elements = [] # frames, e1 e2 e3 e4 e5... e27, [xx xy], [yx yy]
Centroids = [] # frames, e1 e2 e3 e4 e5... e27, x y

for frame in range(1, len(coordinates)):
#for frame in range(3):
    print(frame)
    element, centroid = elements(coordinates[frame], coordinates[frame-1], squares)
    #element = elementS[frame]
    if frame != 1:
      for i in range(len(element)):
        for x in range(2):
          for y in range(2):
            element[i][x][y] = element[i][x][y] + Elements[-1][i][x][y]
    Elements.append(element)
    Centroids.append(centroid)



matrix_dimentions = 2, 8
Z = []

frame = 0
for frame in range(len(Elements)):
  g = []
  print(frame)
  for i in range(matrix_dimentions[0]-1):
      for q in range(matrix_dimentions[1]):
          g.append(Elements[frame][q+i*(matrix_dimentions[1])][1][1])
          #general strain
          '''yy = Elements[frame][q+i*(matrix_dimentions[1])][1][1]
          xx = Elements[frame][q+i*(matrix_dimentions[1])][0][0]
          gs = math.sqrt(xx**2 + yy**2)
          Z.append(gs)'''
  g = np.array(g, dtype='float')
  Z.append(g)

master = []
for i in Elements:
  for x in i:
    master.append(round(x[1][1], 6))

smallest_strain = min(master)
largest_strain = max(master)
#smallest_strain = -0.028964
#largest_strain = 0.027823
print(smallest_strain, largest_strain)

#Z = np.array([1, 2, 3, 4, 8, 5, 6, 7, 1])

#numpy.arange([start, ]stop, [step, ]dtype=None, *, like=None)

#print(Z)
#print(y)
'''cs = plt.contour(y)
cs.cmap.set_over('red')
cs.cmap.set_under('blue')
cs.changed()'''
frame=0
#cmap='RdGy'
#[-0.01   -0.0075 -0.005  -0.0025  0.      0.0025  0.005   0.0075  0.01  ]


for element in Z:
    print(element)
    print()

levels = np.arange(smallest_strain, largest_strain, 0.0025)

frame=0
for element in Z:
  y = element.reshape(2, 4)
  fig, ax = plt.subplots()
  CS = ax.contourf(y, levels, cmap='CMRmap')
  CB = fig.colorbar(CS, shrink=0.8)
  List = ax.clabel(CS, colors='black', fontsize=10, inline=False)
  ax.set_title('Deformation')
  fig.savefig(f"eachFrame/frame{frame}.png",bbox_inches='tight')
  #plt.clabel(CS, *args, **kwargs)
  frame += 1


import cv2
import numpy as np
import glob
img_array = []
for x in range(len(Z)-1, 0, -1):
   filename = f'eachFrame/frame{x}.png'
   img = cv2.imread(filename)
   height, width, layers = img.shape
   size = (width,height)
   img_array.append(img)

out = cv2.VideoWriter('video5.avi',cv2.VideoWriter_fourcc(*'DIVX'), 14, size)

for i in range(len(img_array)):
   out.write(img_array[i])
out.release()