import numpy as np
import cv2
import os
import math

def getColorMask(img):
    #beam
    #lower = np.array([34, 174, 136])
    #upper = np.array([54, 194, 216])
    #DICex
    lower = np.array([12, 105, 0] )
    upper = np.array([32, 125, 80])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out  = cv2.inRange(hsv, lower, upper)
    #cv2.imshow("out", out)
    #cv2.imshow("Input", img)
    return out

def dilate(out):
    kernel=np.ones((3,3),np.uint8)
    dilated=cv2.dilate(out,kernel,iterations=1)
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



image_names = ['beam_with_stickers.png', 'beam_with_stickers_2.png']

images = []

for name in image_names:
    image=cv2.imread(name)
    images.append(image)


information = []

for image in images:
    contours = get_contours(dilate(getColorMask(image)))

    

    #new sticker coords
    info = []
    cont = []
    init_points = 0
    for cnt in contours:
        x,y,w,h=cnt
        cX, cY = x+int(w/2), y+int(h/2)
        #cv2.circle(image, (cX, cY), 2, (0, 0, 255), -1)
        #cv2.putText(image, str(X), (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        cont.append([init_points, cX, cY])
        init_points += 1
    
    if information != []:
        prev_cont = information[-1]
        for point in cont:
            contX = point[1]
            contY = point[2]
            smallest_n, smallest_dist = prev_cont[0][0], distance(contX, contY, prev_cont[0][1], prev_cont[0][2])
            for prev_point in prev_cont:
                dist = distance(contX, contY, prev_point[1], prev_point[2])
                if dist < smallest_dist:
                    smallest_n, smallest_dist = prev_point[0], dist #distance calculated here
            info.append([smallest_n, contX, contY])
        information.append(info)
    else:
        information.append(cont)
    

xx= 0
for info in information:
    for point in info:
        cv2.circle(images[xx], (point[1], point[2]), 2, (0, 0, 255), -1)
        cv2.putText(images[xx], str(point[0]), (point[1]-9, point[2]-9),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow("hope this works",images[xx])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    xx += 1

