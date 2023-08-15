import numpy as np
import cv2
from PIL import Image
import time

filepath = "C://Users//deniz//Desktop//Python Building Stress CAM//examples//plastic_wrap.png"
img = Image.open(filepath)
  
# get width and height
width = img.width
height = img.height


altW = 500
y = altW / width
altY = int(y*height)

image = cv2.imread('C://Users//deniz//Desktop//Python Building Stress CAM//examples//plastic_wrap.png')
image = cv2.resize(image, (altW, altY))




def getColorMask(img):
    lower = np.array([96, 25, 49])
    upper = np.array([156, 65, 176])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out  = cv2.inRange(hsv, lower, upper)
    return out
def get_contours(dilated_image):
    contours,_ = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours=[cv2.boundingRect(cnt) for cnt in contours]
    return contours

cv2.imshow('balls' ,image)
cv2.waitKey(0)


mask = getColorMask(image)


kernel = np.ones((3, 3), np.uint8)
combo = []
choice = ''
while choice != 'x':
    choice = input('d, e, x')
    if choice == 'd':
        mask = cv2.dilate(mask, kernel, iterations=1)
        cont = get_contours(mask)
        for cnt in cont:
            x,y,w,h=cnt
            cX, cY = x+int(w/2), y+int(h/2)
            cv2.circle(image, (cX, cY), 2, (0, 0, 255), -1)
        cv2.imshow('dot',image)
        cv2.imshow('dilated',mask)
        cv2.waitKey(0)
        combo.append('dilated')
        print(f'{len(get_contours(mask))} dots found')
    elif choice == 'e':
        mask = cv2.erode(mask, kernel, iterations=1)
        cont = get_contours(mask)
        for cnt in cont:
            x,y,w,h=cnt
            cX, cY = x+int(w/2), y+int(h/2)
            cv2.circle(image, (cX, cY), 2, (0, 0, 255), -1)
        cv2.imshow('dot',image)
        cv2.imshow('eroded',mask)
        cv2.waitKey(0)
        combo.append('eroded')
        print(f'{len(cont)} dots found')
    else:
        cv2.imshow('mask',mask)
        cv2.waitKey(0)
print(combo)




cv2.destroyAllWindows()
