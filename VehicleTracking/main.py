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


cap = cv2.VideoCapture("E:\\LUAN VAN\\repo\VIDEO\\traffic_no_shadow\\20170812_162710.mp4")
# cap = cv2.VideoCapture(0)
threshold = 30
preframe = None
index = 0
A = np.full((480, 640), 0)
while(cap.isOpened()):
    ret, frame = cap.read() 
    # cv2.imshow('frame',frame)
    height, width, channels = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, ksize=(KERNEL_WIDTH, KERNEL_HEIGHT), sigmaX=SIGMA_X, sigmaY=SIGMA_Y)
    if index != 0:
        temp = cv2.absdiff(preframe, gray)
        # cv2.imshow('aav v',temp)
        ret,thresh1 = cv2.threshold(temp,threshold,255,cv2.THRESH_BINARY)
        # thresh1 = cv2.adaptiveThreshold(temp,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        #     cv2.THRESH_BINARY_INV,11,2)

        kernel = np.ones((30,30),np.uint8)
        #
        thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        # Find contours in thresh_gray after closing the gaps
        contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Find the convex hull object for each contour
        hull_list = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)
        # Draw contours + hull results
        drawing = np.zeros((thresh1.shape[0], thresh1.shape[1], 3), dtype=np.uint8)
        color = (255,255,255)
        for i in range(len(contours)):
            cv2.drawContours(drawing, contours, i, color,cv2.FILLED,cv2.MARKER_SQUARE)
            cv2.drawContours(drawing, hull_list, i, color,cv2.FILLED,cv2.MARKER_SQUARE)
            
        for i in range(len(hull_list)):
            M = cv2.moments(hull_list[i])
            if M['m00'] != 0 and M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.circle(drawing, (cx,cy), radius=5, color=(0, 255, 0), thickness=-1)
        # for c in contours:
        #     area = cv2.contourArea(c)

        #     # Small contours are ignored.
        #     if area < 50:
        #         cv2.fillPoly(thresh1, pts=[c], color=0)
        #         continue

        #     ellipse = cv2.fitEllipse(c)
        #     cv2.ellipse(frame,ellipse,(0,255,0),2)
        cv2.imshow('drawing',drawing)
        cv2.imshow('thresh1',thresh1)
        cv2.imshow('a',frame)
    preframe = gray
    index += 1
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
