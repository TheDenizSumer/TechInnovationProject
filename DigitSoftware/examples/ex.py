import numpy as np
import cv2
import os

image=cv2.imread('beam_with_stickers.png')

def getColorMask(img):
    lower = np.array([34, 174, 136])
    upper = np.array([54, 194, 216])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out  = cv2.inRange(hsv, lower, upper)
    #cv2.imshow("out", out)
    #cv2.imshow("Input", img)
    return out

def dilate(out):
    kernel=np.ones((3,3),np.uint8)
    dilated=cv2.dilate(out,kernel,iterations=1)
    cv2.imshow("Input", dilated)
    return dilated

def get_contours(dilated_image):
    contours,_ = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours=[cv2.boundingRect(cnt) for cnt in contours]
    return contours





contours = get_contours(dilate(getColorMask(image)))

X=0
for cnt in contours:
    x,y,w,h=cnt
    cX, cY = x+int(w/2), y+int(h/2)
    cv2.circle(image, (cX, cY), 2, (0, 0, 255), -1)
    cv2.putText(image, str(X), (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    X += 1

cv2.imshow("hope this works",image)

cv2.waitKey(0)
cv2.destroyAllWindows()


'''
import numpy as np
import cv2
import os

def getColorMask(img):
    lower = np.array([34, 174, 136])
    upper = np.array([54, 194, 216])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out  = cv2.inRange(hsv, lower, upper)
    cv2.imshow("Input", out)
    return out

def Dilate(img):
    kernel=np.ones((3,3),np.uint8)
    dilated=cv2.dilate(getColorMask(img),kernel,iterations=1)
    cv2.imshow("Input", dilated)
    return dilated

def get_contours(img):
    contours,_ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours=[cv2.boundingRect(cnt) for cnt in contours]
    return contours

image=cv2.imread('beam_with_stickers.png')

CM = getColorMask(image)
ID = Dilate(CM)
contours = get_contours(ID)



X=0
for cnt in contours:
    x,y,w,h=cnt
    cX, cY = x+int(w/2), y+int(h/2)
    cv2.circle(image, (cX, cY), 2, (0, 0, 255), -1)
    cv2.putText(image, str(X), (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    X += 1

cv2.imshow("hope this works",image)

cv2.waitKey(0)
cv2.destroyAllWindows()'''