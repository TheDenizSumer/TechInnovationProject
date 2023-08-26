import numpy as np
import cv2
import os
import math 

def getColorMask(img):
    lower = np.array([ 12, 161, 38] ) 
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



cap = cv2.VideoCapture('deform_purple.mov')


frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
   
size = (frame_width, frame_height)

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

images = []
frames = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True:
        #cv2.imshow('Frame',frame)
        #frame = cv2.resize(frame, (960, 540))
        images.append(frame)
        frames += 1
        
    else: 
        break
if len(images) == frames:
    print('Good to go')
else:
    print(f'Images: {len(images)}')
    print(f'Frames: {frames}')


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
    
#result = cv2.VideoWriter('computed_colormasking.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
xx= 0
for info in information:
    for point in info:
        cv2.circle(images[xx], (point[1], point[2]), 2, (0, 0, 255), -1)
        cv2.putText(images[xx], str(point[0]), (point[1]-9, point[2]-9),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow("hope this works",images[xx])
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    #result.write(images[xx])
    
    xx += 1

cv2.waitKey(0)
cv2.destroyAllWindows()

#python C:\Users\deniz\Desktop\computer_2\DigitSoftware\template_matching\main.py