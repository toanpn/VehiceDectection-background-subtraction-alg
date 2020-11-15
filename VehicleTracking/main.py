import numpy as np
import cv2
import time
from PIL import Image

def frame_diff(prev,curr,next):
	diff1=cv2.absdiff(next,curr)
	diff2=cv2.absdiff(curr,prev)
	return cv2.bitwise_and(diff1,diff2)


KERNEL_WIDTH = 9
KERNEL_HEIGHT = 9
SIGMA_X = 4
SIGMA_Y = 4


# cap = cv2.VideoCapture("E:\\LUAN VAN\\repo\\VIDEO\\bike_counter_10min.mp4")
cap = cv2.VideoCapture(0)
threshold = 10
preframe = None
index = 0
A = np.full((480, 640), 0)
while(cap.isOpened()):
    ret, frame = cap.read() 
    # cv2.imshow('frame',frame)
    frameGauss = cv2.GaussianBlur(frame, ksize=(KERNEL_WIDTH, KERNEL_HEIGHT), sigmaX=SIGMA_X, sigmaY=SIGMA_Y)
    height, width, channels = frame.shape
    gray = cv2.cvtColor(frameGauss, cv2.COLOR_BGR2GRAY)
    cv2.imshow('a',gray)
    if index != 0:
        # temp = gray - preframe
        temp = cv2.absdiff(preframe, gray)
        # cv2.imshow('aav v',temp)
        ret,thresh1 = cv2.threshold(temp,threshold,255,cv2.THRESH_BINARY)
        # thresh1 = cv2.adaptiveThreshold(temp,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        #     cv2.THRESH_BINARY_INV,11,2)

        

        #
        thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31)))

        # Find contours in thresh_gray after closing the gaps
        # image, contours, hier = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        for c in contours:
            area = cv2.contourArea(c)

            # Small contours are ignored.
            if area < 500:
                cv2.fillPoly(thresh1, pts=[c], color=0)
                continue

            # rect = cv2.minAreaRect(c)
            # box = cv2.boxPoints(rect)
            # convert all coordinates floating point values to int
            # box = np.int0(box)
            # cv2.drawContours(frame, [box], 0, (176, 39, 158),1)
            # eclipse
            ellipse = cv2.fitEllipse(c)
            cv2.ellipse(frame,ellipse,(0,255,0),2)
            #cỉcle
            # (x,y),radius = cv2.minEnclosingCircle(c)
            # center = (int(x),int(y))
            # radius = int(radius)
            # cv2.circle(frame,center,radius,(0,255,0),2)


        #
        #
        #
        # cv2.drawContours(frame, contours, -1, (176, 39, 158), 2)

        cv2.imshow('temp',temp)
        cv2.imshow('thresh1',thresh1)
        cv2.imshow('a',frame)
    preframe = gray
    index += 1
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
