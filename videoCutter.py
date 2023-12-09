import cv2

framecutoff = 135
a = [1, 12, 3, 4, 56]
print(a[:2])



video = 'deform_purple.mov'
cap = cv2.VideoCapture(video)


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

if (cap.isOpened()== False):
    print("Error opening video stream or file")
    raise "Error opening video stream or file"

#adding frames of the video to an array called images
images = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        images.append(frame)
    else:
        break



images = images[:framecutoff+1]


import cv2
import numpy as np
import glob
size = (frame_width, frame_height)

out = cv2.VideoWriter('deform_purple_cut.avi',cv2.VideoWriter_fourcc(*'DIVX'), 14, size)

for i in range(len(images)):
   out.write(images[i])
out.release()
