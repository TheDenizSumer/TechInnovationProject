import cv2
import numpy as np

image_hsv = None   # global ;(
pixel = (20,60,80) # some stupid default

# mouse callback function
def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]
        print(x, y)
        #you might want to adjust the ranges(+-10, etc):
        #upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        #lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
        #upper =  np.array([pixel[0] + 20, pixel[1] + 20, pixel[2] + 40])
        #lower =  np.array([pixel[0] - 20, pixel[1] - 20, pixel[2] - 40])
        upper =  np.array([pixel[0] + 30, pixel[1] + 20, pixel[2] + 40])
        lower =  np.array([pixel[0] - 30, pixel[1] - 20, pixel[2] - 40]) #[141  29  87] [111   9  47] [171  49 127]
        print(pixel, lower, upper)

        image_mask = cv2.inRange(image_hsv,lower,upper)
        cv2.imshow("mask",image_mask)



def main():
    import sys
    global image_hsv, pixel # so we can use it in mouse callback
    cap = cv2.VideoCapture('deform_purple.MOV')
    ret, frame = cap.read()
    image_src = frame  # pick.py my.png
    image_src = cv2.resize(image_src, (1000, 750))
    
    #image_src = cv2.imread('plastic_wrap.png')
    if ret:
        print('yay')
    if image_src is None:
        print ("the image read is None............")
        return
    cv2.imshow("bgr",image_src)
    ## NEW ##
    cv2.namedWindow('hsv')
    cv2.setMouseCallback('hsv', pick_color)

    # now click into the hsv img , and look at values:
    image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv",image_hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
#[ 18 158 207] [  8 148 167] [ 28 168 247]
#[ 17 173 202] [  7 163 162] [ 27 183 242]

#python H:\python\TechInnovationProject\DigitSoftware\template_matching\deform.MOV