from deformationGradientFunction import F
import math
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import math 
from statistics import mode

'''



import matplotlib.pyplot as plt
import numpy as np
  
feature_x = np.linspace(-5.0, 3.0, 700)
feature_y = np.linspace(-5.0, 3.0, 700)
  
# Creating 2-D grid of features
[X, Y] = np.meshgrid(feature_x, feature_y)
  
fig, ax = plt.subplots(1, 1)
  
Z = X ** 2 + Y ** 2

print(Z)
# plots filled contour plot
ax.contourf(X, Y, Z)
  
ax.set_title('Filled Contour Plot')
ax.set_xlabel('feature_x')
ax.set_ylabel('feature_y')
  

plt.show()
'''

def calc_coordinates(video, frame_cap=0, remove_frames=True):
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


#frame cap 135

def Calc_centroid(p1, p2, p3, p4):
    x = (p1[0]+p2[0]+p3[0]+p4[0])/4
    y = (p1[1]+p2[1]+p3[1]+p4[1])/4
    return x, y


squares = [
    [21, 22, 24, 25],
    [22, 23, 26, 24],
    [18, 19, 22, 21],
    [19, 20, 23, 22],
    [15, 16, 19, 18],
    [16, 17, 20, 19],
    [12, 13, 16, 15],
    [13, 14, 17, 16],
    [10, 9, 13, 12],
    [9, 11, 14, 13],
    [6, 7, 9, 10],
    [7, 8, 11, 9],
    [3, 4, 7, 6],
    [4, 5, 8, 7],
    [1, 0, 4, 3],
    [0, 2, 5, 4]
    ]


coordinates, images = calc_coordinates('deform_purple.mov', 135)



def elements(T, NT, squares, direction=None):
    def_element = []
    centroids = []
    for element in squares:
        et = [T[element[0]], T[element[1]], T[element[2]], T[element[3]]]
        e = [NT[element[0]], NT[element[1]], NT[element[2]], NT[element[3]]]
        et = [T[element[0]][1:], T[element[1]][1:], T[element[2]][1:], T[element[3]][1:]]
        e = [NT[element[0]][1:], NT[element[1]][1:], NT[element[2]][1:], NT[element[3]][1:]]
        def_element.append(F(et, e)[0])
        centroids.append(F(et, e)[1])
    return def_element, centroids


Elements = [] # frames, e1 e2 e3 e4 e5... e27, [xx xy], [yx yy]
Centroids = [] # frames, e1 e2 e3 e4 e5... e27, x y 

'''for frame in range(1, 90):#len(coordinates)-132):
    print(frame)
    element, centroid = elements(coordinates[frame], coordinates[frame-1], squares)
    Elements.append(element)
    Centroids.append(centroid)'''

print('done')

matrix_dimentions = 2, 8
feature_x = np.linspace(0, matrix_dimentions[0], 8)
feature_y = np.linspace(0, matrix_dimentions[1], 2)

# Creating 2-D grid of features
[X, Y] = np.meshgrid(feature_x, feature_y)


fig, ax = plt.subplots(1, 1)

print(ax)

#Z = X ** 2 + Y ** 2


'''Z = [
    [9.71698113e-03  7.18415766e-06  0.00000000e+00  9.41786607e-03 1.00722847e-02 -9.90617203e-03  1.98168613e-02 -1.03999967e-02]
    [9.71698113e-03  7.18415766e-06  0.00000000e+00  9.41786607e-03 1.00722847e-02 -9.90617203e-03  1.98168613e-02 -1.03999967e-02]]
Z = np.array(Z)
'''


asdf, baba = elements(coordinates[124], coordinates[0], squares)
print(asdf)


Z = []
frame = 0
for i in range(matrix_dimentions[0]):
    Z.append([])
    for q in range(matrix_dimentions[1]):
        #Z[-1].append(Elements[frame][q+i*(matrix_dimentions[1])][1][1])
        Z[-1].append(asdf[q+i*(matrix_dimentions[1])][1][1])
Z = np.array(Z)

print(Z)


# plots filled contour plot
ax.contourf(X, Y, Z)

ax.set_title('Filled Contour Plot')
ax.set_xlabel('feature_x')
ax.set_ylabel('feature_y')

plt.show()
