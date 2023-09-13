import cv2
import np

def getColorMask(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out  = cv2.inRange(hsv, lower, upper)
    return out



cap = cv2.VideoCapture('deform2.MOV')

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

images = []
while(cap.isOpened()):
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (1000, 750))
    if ret == True:
        #cv2.imshow('Frame',frame)
        images.append(frame)
        break
        #if cv2.waitKey(25) & 0xFF == ord('q'):
        #    break
        
    else: 
        break

lower = np.array([ 42, 255,  35] ) 
upper = np.array([ 72, 275,  75])

while True:

    
    getColorMask(lower, upper)
    input