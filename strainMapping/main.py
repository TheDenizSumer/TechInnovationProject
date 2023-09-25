from deformationGradientFunction import F
import math
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
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
            information.append(info)
        else:
            information.append(cont)

    
    return information, images


#frame cap 135


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


information, images = coordinates('deform_purple.mov', 135)



# Create a grid of x and y values
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Generate some random data for the Z values
Z = np.sin(np.sqrt(X**2 + Y**2))
print(len(Z))

# Create a filled contour plot with a colormap
plt.contourf(X, Y, Z, cmap='viridis')  # You can choose any colormap you prefer
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Map with Color Mapping')
plt.colorbar()  # Add a colorbar
plt.show()

