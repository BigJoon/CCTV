from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

from imutils.video.filevideostream import FileVideoStream
import jetson.inference
import jetson.utils


cap = cv2.VideoCapture('./outpy.mp4')

if(cap.isOpened() == False):
    print("Unable to read Camera feeed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

while(True):
    ret, frame = cap.read()
    

    if ret == True:
        cv2.imshow('frame',frame)
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break


    else:
        break

cap.release()

cv2.destroyAllWindows()
