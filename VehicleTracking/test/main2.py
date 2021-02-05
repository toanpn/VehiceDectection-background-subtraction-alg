import numpy as np
import cv2
import time
from PIL import Image

cap = cv2.VideoCapture("E:\\LUAN VAN\\repo\\VIDEO\\car.mp4")
threshold = 15
preFrameGray = None
preFrameGray2 = None
preFrameGray3 = None
index = 0
A = np.full((480, 640), 0)

KERNEL_WIDTH = 9
KERNEL_HEIGHT = 9
SIGMA_X = 4
SIGMA_Y = 4

def CheckExitLineCrossing(y, CoorYEntranceLine):
	AbsDistance = abs(y - CoorYEntranceLine)
	if AbsDistance < 40 & y < CoorYEntranceLine:
		return 1
	return 0
ExitCounter = 0
while(cap.isOpened()):
    ret, thisFrame = cap.read() 

    height, width, channels = thisFrame.shape
    #change frame to gray
    thisFrameGray = cv2.cvtColor(thisFrame, cv2.COLOR_BGR2GRAY)
    thisFrameGray_gauss = cv2.GaussianBlur(thisFrameGray, ksize=(KERNEL_WIDTH, KERNEL_HEIGHT), sigmaX=SIGMA_X, sigmaY=SIGMA_Y)

    if index != 0:
        temp_gauss = cv2.absdiff(preFrameGray2, thisFrameGray_gauss)
        ret,thresh1 = cv2.threshold(temp_gauss,threshold,255,cv2.THRESH_BINARY)
        kernel = np.ones((30,30),np.uint8)
        cv2.imshow('thresh1',thresh1)
        #
        thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        drawing = np.zeros((thresh1.shape[0], thresh1.shape[1], 3), dtype=np.uint8)
        color = (255,255,255)
        frameContours = cv2.drawContours(drawing, contours, -1, (0,255,0), -1)
        hull_list = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)
        frameHull = cv2.drawContours(drawing, hull_list, -1, (0,255,0), -1)
        frame_final = thisFrame.copy()
        for c in hull_list:
            m=cv2.moments(c)
            if m['m00'] == 0: continue
            cx=int(m['m10']/m['m00'])
            cy=int(m['m01']/m['m00'])            
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame_final, (x, y), (x + w, y + h), (0, 255, 0), 1)
            CoordXCentroid = int((x+x+w)/2)
            CoordYCentroid = int((y+y+h)/2)
            ObjectCentroid = (CoordXCentroid,CoordYCentroid)
            cv2.circle(frame_final, ObjectCentroid, 1, (255, 0, 0), 5)
            if CheckExitLineCrossing(y,int(height*4/5)):  
                ExitCounter += 1
        cv2.putText(frame_final, "CONTER: {}".format(str(ExitCounter)), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 1), 2)
        
        cv2.imshow("frame_final", frame_final)
        cv2.imshow('frame',thisFrame)
        cv2.imshow('temp_gauss',temp_gauss)
        cv2.imshow('frameContours',frameContours)
        cv2.imshow('frameHull',frameHull)
    preFrameGray2 = thisFrameGray_gauss
    index += 1
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
