from threading import Thread
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
#import sys

#Initialize inference engine for using jetson-inference module
net = jetson.inference.detectNet("ssd-mobilenet-v2")

#Initialize IP Camera link
camlink1 = "rtsp://root:root@163.239.25.71:554/cam0_1"
camlink2 = "rtsp://root:root@163.239.25.71:554/cam0_1"
camlink3 = "rtsp://root:root@163.239.25.71:554/cam0_1"
camlink4 = "rtsp://root:root@163.239.25.71:554/cam0_2"
#camlink4 = -1
flip=0
dispW=640
dispH=480

#In thisOneDetectionModel Class send frame to network model and get result and save it.
class OneDetectionModel():
    #In this class, we'll gonna use mobile ssd v2 network and make detection happens

    def __init__(self):
        #initiation of network
        print("OneDetectionModel START")
        pass
        

    def detect_infer(self,frame_cuda,frame,camname):
        #But Dr.Song said we need queuing..! so before we give parameters we need to seperate them.
        predictions = net.Detect(frame_cuda,frame.shape[1],frame.shape[0])
        #From here i'll gonna make branch by CAMNAME:"Cam1", "Cam2", "Cam3"
        return predictions
        
class VideoStreamWidget(object):
    
    def __init__(self, link, camname, labels, colors,src=0):
        #self.net = net
        self.labels = labels
        self.colors = colors

        self.capture = cv2.VideoCapture(link)
        #self.capture = cv2.VideoCapture(-1)
        #self.capture.set(3,800)
        #self.capture.set(4,600)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.camname = camname
        self.link = link
        print(camname)
        print(link)
    
    def update(self):

        while True:
            if self.capture.isOpened():
                (self.status, self.frame)=self.capture.read()
            time.sleep(.01)
    def show_frame(self):
        frame = imutils.resize(self.frame, width=700,height=700)
        frame_cuda = cv2.cvtColor(frame,cv2.COLOR_RGB2RGBA).astype(np.float32)
        frame_cuda = jetson.utils.cudaFromNumpy(frame_cuda)
        startTime = time.time()
        
        #Call OneDetectionModel Class for Object Detection
        predictions=OneDetectionModel().detect_infer(frame_cuda,frame,self.camname)
        
        #############################################################################

        endTime = time.time()-startTime
        calcTime = round(endTime,3)
        
        #cv2.putText(frame,"{:.2f}".format(1/calcTime),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        #print("detected {:d} objects in image".format(len(predictions)))

        for prediction in predictions:
            #FOR DEBUG print(classes[prediction.ClassID])

            label = self.labels[prediction.ClassID]
            confidence = "%d%%"%(int(prediction.Confidence*100))
            x=int(prediction.Left)
            y=int(prediction.Top)
            width = int(prediction.Width)
            height = int(prediction.Height)
            cv2.rectangle(frame,(x,y),(x+width, y+height),self.colors[72/prediction.ClassID],2)
            cv2.putText(frame,label+":"+confidence,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,self.colors[72/prediction.ClassID],2)
    
        ######################################################################################
        
         
        cv2.imshow('Frame '+ self.camname, frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)


if __name__== '__main__':

    CLASSES= ["","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffci light","fire hydrant","","stop sign", "parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","","backpack","umbrella","","","handbag","tie","suitcase","frisbee","skis","snowboard","sprotsball","kite","baseball bat","baseball glove", "skateboard","surfboard","tennis racket","bottle","","wine glass", "cup","fork","knife","spoon","bowl","banana","apple","sandwich","oragne","broccoli","corrot","got dog","pizza","donut","cake","chair","couch","potted plant","bed","","dining table","","","toilet","","tv","laptop","mouse","remote","keyboard","cellphone","microwave","oven", "toaster","sink","refrigerator","","book","clock","vase","scissors","teddy bear", "hair drier","toothbrush"]

    COLORS = np.random.uniform(0,255, size=(len(CLASSES),3))
    network = OneDetectionModel()

    video_stream_widget = VideoStreamWidget(camlink1,"Cam1",CLASSES,COLORS)
    video_stream_widget2 = VideoStreamWidget(camlink2,"Cam2",CLASSES,COLORS)
    video_stream_widget3 = VideoStreamWidget(camlink3,"Cam3",CLASSES,COLORS)
    video_stream_widget4 = VideoStreamWidget(camlink4,"Cam4",CLASSES,COLORS)

    while True:
        try:
            video_stream_widget.show_frame()
            video_stream_widget2.show_frame()
            video_stream_widget3.show_frame()
            video_stream_widget4.show_frame()
        except AttributeError:
            pass
