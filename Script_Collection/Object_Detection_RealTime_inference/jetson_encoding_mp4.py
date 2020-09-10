#This script is used in jetson-inference folder.(jetson-inference/build/aarch64/bin/)
#How to use? python3 jetson_encoding_mp4.py

"""
Under the code you can find "change input video name" and "output video name".
For changing "input video" search "cv2.VideoCapture" and rewrite it.
For changing "output video" search "cv2.VideoWriter" and rewrite it.
"""

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

net = jetson.inference.detectNet("ssd-mobilenet-v2")
CLASSES=["","person","bicycle","bus","car","carrier","motorcycle","movable_signage","truck","bollard","chair","potted    _plant","table","tree_trunk","pole","fire_hydrant"]



cap = cv2.VideoCapture('./360p/14_360p.mp4')

if(cap.isOpened() == False):
    print("unable to read source video feed")



frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

##--------------For writing Video--------------------------------------------------------------###
out = cv2.VideoWriter('./360p_output/14_360p.mp4', cv2.VideoWriter_fourcc(*'mp4v'),30,(frame_width,frame_height))

##----------------------------------------------------------------------------------------------##

while(True):
    ret,frame = cap.read()

    if ret == True:
        frame = imutils.resize(frame, width=frame_width,height=frame_height)
        frame_cuda = cv2.cvtColor(frame,cv2.COLOR_RGB2RGBA).astype(np.float32)
        frame_cuda = jetson.utils.cudaFromNumpy(frame_cuda)
        
        startTime = time.time()
        predictions = net.Detect(frame_cuda,frame.shape[1],frame.shape[0])
        print(predictions)
        endTime = time.time()-startTime
        calcTime = round(endTime,3)
        print("inference time -----------",1/calcTime)

        for prediction in predictions:
            label = CLASSES[prediction.ClassID]
            confidence = "%d%%"%(int(prediction.Confidence*100))
            x=int(prediction.Left)
            y=int(prediction.Top)
            width = int(prediction.Width)
            height = int(prediction.Height)
            cv2.rectangle(frame,(x,y),(int(x+width), int(y+height)),(1,128,129),2)
            cv2.putText(frame,label+":"+confidence,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(1,128,128),2)

	#---------------for writing Video------------------#
        out.write(frame)
	#--------------------------------------------------#

        
	#cv2.imshow('frame',frame)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
    

    else:
        break

cap.release()
out.release()


cv2.destroyAllWindows()
