#Modified by smartbuilds.io
#Date: 27.09.20
#Desc: This scrtipt script..

import cv2
#from imutils.video.pivideostream import PiVideoStream
from faceRecogLayer import run_detection
import imutils
import datetime
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class VideoCamera(object):
    def __init__(self, flip = False):
        # set up to capture video from that website
        self.vs = cv2.VideoCapture("http://10.160.137.64:8000/stream.mjpg")
        self.flip = flip
        self.counter = 0

        #if the timer is running (a face was seen)
        self.timerRunning = False

        #video writer to save video
        self.fourcc = cv2.VideoWriter_fourcc(*'FMP4')
        self.frameSize = [1,1]   
        self.video_tracked = cv2.VideoWriter('videos/defaultName.mp4', self.fourcc, 25.0, self.frameSize)

        #what time the timer started running
        self.startTime = 0

        #the default length of the timerr
        self.timerLength = 30

        #text fonts
        self.timerFont = ImageFont.truetype("Gidole-Regular.ttf", size=80)
        self.textFont = ImageFont.truetype("Gidole-Regular.ttf", size=40)

        #sleep to make sure stream is up before doing anything else
        time.sleep(2.0)

    def __del__(self):
        self.restartTimer()
        self.vs.stop()
        pass

    def flip_if_needed(self, frame):
        if self.flip:
            return np.flip(frame, 0)
        return frame

    def draw_box_on_image(self, boxes, prob, frame):
        '''Draws boxes and text on the image as needed/found by run_detection'''
        
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_draw = frame.copy()

        draw = ImageDraw.Draw(frame_draw)
        try:
            for box in boxes[0]:
                box = ((box[0],box[1]),(box[2],box[3]))
                draw.rectangle(box, outline=(255, 0, 0), width=6)
            self.faceFoundStartTimer(frame)  
        except TypeError:
            pass
            #draw.text((350,100), "No Face Found!", fill=(255,255,255,128), font = font)
        
        if (self.timerRunning == True):
             #get the time to draw by checking when the timer should start
            timeToDraw = self.timerLength - (int(time.time())-self.startTime)
            if timeToDraw >=0:
                 draw.text((300,10), str(timeToDraw), fill=(255,255,255,128), font = self.timerFont)
            else:
                draw.text((50,10), "Warning 30 seconds has passed!", fill=(255,255,255,128), font = self.textFont)
        image_array = np.array(frame_draw.resize((640, 360), Image.BILINEAR))
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        return image_array

    def get_frame(self):
        '''
        This method gets a frame of video to serve the website 
        @ret byte representation of a frame
        '''
        ret, frame = self.vs.read()

        #only run detection for faces every three frames
        self.counter = self.counter%3
        if (self.counter==0):
            prob = []
            self.boxes, self.prob = run_detection(frame)
        self.counter += 1

        #draw any boxes that need to be drawn on the image and return the frame
        frame = self.draw_box_on_image(self.boxes, self.prob, frame)
        ret, jpeg = cv2.imencode('.jpg', frame)

        #if procedure is started (face was seen) save video to videos
        if (self.timerRunning):
            #self.video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
            pass

        return jpeg.tobytes()

    def faceFoundStartTimer(self, frame):
        '''
        This method starts a timer to be drawn on the image.
        '''
        #if the timer is already running do nothing
        if (self.timerRunning == True):
            return 0
        
        #if not set the boolean to track that it is running
        self.timerRunning = True
        self.startTime  = int(time.time())

        #self.frameSize = frame.size
        #self.video_tracked.release()
        #self.video_tracked.open("videos/"+datetime.datetime.now()+".mp4", self.fourcc, 25.0, self.frameSize)

    def restartTimer(self):
        '''
        This restarts the timer to be able to run again.
        '''
        self.timerRunning = False
        #self.video_tracked.release()
        
