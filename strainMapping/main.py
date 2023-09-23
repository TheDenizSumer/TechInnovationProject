from deformationGradientFunction import F
import math
import numpy as np
import cv2
import os
import math 
from statistics import mode

def coordinates(video, frame_cap=0, remove_frames=True):
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
        print('Good to go')
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
            information.append(info)
        else:
            information.append(cont)

    
    return information, images



#def recognize_squares(coordinates):
#  squares = []
#  
#  return largest_square

print(recognize_squares(coordinates('deform_purple.mov', 135)[0][0]))