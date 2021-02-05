import numpy as np
import cv2
from collections import OrderedDict
   
minDistaneThreshold = 65
maxDistanceFromLine = 200
buffer = 50

# constant
c_white = (255,255,255)
c_green = (0,255,0)
c_yellow = (0, 255, 255)
c_pink = (255,255,255)
c_cyan = (255, 255, 0)
threshold = 29
minSizeHull = 5000
listCertainLog = []
kernel = np.ones((3,7),np.uint8)

class Certain:  
    def __init__(self, x, y, distance = -1):  
        self.x = x  
        self.y = y 
        self.distance = distance 
          
    def CaculateDistance(self): 
        return abs(self.y - Y)
 
#region Function Declare
def IsSameCentain(cen1, cen2):
    _distance = ((((cen2.x - cen1.x )**2) + ((cen2.y-cen1.y)**2) )**0.5)
    print ("check same ", " cen1:" ,cen1.x, cen1.y," cen2:" ,cen2.x, cen2.y,"distance",_distance)
    if _distance > minDistaneThreshold:
        return False
    return True

# gaussian cho frame x
def denoise(frame):
    # frame = cv2.medianBlur(frame,5)
    frame = cv2.GaussianBlur(frame,(5,5),0)
    # frame = cv2.medianBlur(frame, 5)
    return frame
def CheckExitLineCrossing(y):
    AbsDistance = abs(y - Y)
    if (y < Y and Y - y < (maxDistanceFromLine + buffer)):
        return 1, AbsDistance
    return 0, _
def CheckExistInLog(x,y):
    tempCertain = Certain(x,y)
    for item in listCertainLog:
        if IsSameCentain(tempCertain,item):
            return True, item
    return False, _
def AddToLog(cx,cy):
    _certain = Certain(cx,cy,abs(cy-Y))
    listCertainLog.append(_certain)
def RemoveCertainInLog(item):
    listCertainLog.remove(item)
def UpdateCertainInLog(item, cx,cy):
    listCertainLog[listCertainLog.index(item)] = Certain(cx,cy,abs(cy-Y))


#endregion

# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture("E:\\LUAN VAN\\repo\\VIDEO\\car.mp4")

# cam = cv2.VideoCapture(0)

def nothing(x):
	pass

cv2.namedWindow('Configuration')
cv2.createTrackbar('threshold','Configuration',30,100,nothing)

ret,frame_init = cam.read()
height, width, channels = frame_init.shape 
Y = int(height*4/5)
ExistCounter = 0


if ret is True:
    frame_init = cv2.cvtColor(frame_init, cv2.COLOR_BGR2GRAY)
    # assign first frame to background
    PreFrame = denoise(frame_init)
    run = True
else:
    run = False

while(run):
    ret,frame_origin= cam.read()
    if ret is True:
        frame_origin_gray = cv2.cvtColor(frame_origin, cv2.COLOR_BGR2GRAY)
        frame_process = frame_origin_gray.copy()
        # draw linecheck and line threshold
        cv2.line(frame_origin, (0, Y), (width,Y), c_yellow,3)
        cv2.line(frame_origin, (0, Y - maxDistanceFromLine),(width, Y - maxDistanceFromLine), c_cyan, 1)
        # foreground will be background - curr frame
        foreGround = cv2.absdiff(denoise(frame_process.copy()),denoise(PreFrame.copy()))
        PreFrame = frame_process
        # View background and  
        cv2.imshow('foreGround Before Threshold Process',foreGround)
        threshold = cv2.getTrackbarPos('threshold','Configuration')
        thresh1 = cv2.threshold(foreGround, threshold, 255, cv2.THRESH_BINARY)[1]
        
        thresh1 = cv2.erode(thresh1, None, iterations=2)        
        thresh1 = cv2.dilate(thresh1, np.ones((3,9),np.uint8), iterations=10)

        cv2.imshow('foreGround after threshold, erord, dilate',thresh1)
        # cv2.imshow('thresh1',thresh1)
        frameContours = np.zeros((thresh1.shape[0], thresh1.shape[1], 3), dtype=np.uint8)
        contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        listNewContours = []
        for t in contours:
            if cv2.contourArea(t, True) < 0:
                listNewContours.append(t)
        contours = listNewContours
        frameContours = cv2.drawContours(frameContours, contours, -1, c_green, 2, 5)
        drawing = np.zeros((thresh1.shape[0], thresh1.shape[1], 3), dtype=np.uint8)
        cv2.imshow('frameContours',frameContours)
        hull_list = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            size = cv2.contourArea(hull)
            if size > minSizeHull:
                hull_list.append(hull)
        frameHull = cv2.drawContours(frame_process.copy(), hull_list, -1, c_green ,1)
        cv2.imshow('frameHull',frameHull)
        frame_output = frame_origin.copy()
        for c in hull_list:
            m=cv2.moments(c)
            if m['m00'] == 0: continue
            # cx=int(m['m10']/m['m00'])
            # cy=int(m['m01']/m['m00'])            
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame_output, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cx = int((x+x+w)/2)
            cy = int((y+y+h)/2)
            ObjectCentroid = (cx,cy)
            cv2.circle(frame_output, ObjectCentroid, 1, (255, 0, 0), 5)
            isPass, distance =  CheckExitLineCrossing(cy)
            if isPass:
                print ("-------PASS---------")
                isExistInLog, item = CheckExistInLog(cx,cy)
                print ("isExistInLog: ",isExistInLog, cx, cy)  
                if isExistInLog:
                    # neu exists: if distance > maxDistance -> remove in log, else update log
                    newDistanceToLine = abs(cy-Y)
                    print ("newDistanceToLine: ", newDistanceToLine)
                    if newDistanceToLine > maxDistanceFromLine:
                        RemoveCertainInLog(item)
                        print ("RemoveCertainInLog: ", len(listCertainLog))
                    else:
                        UpdateCertainInLog(item, cx,cy)
                        print ("UpdateCertainInLog: ",cx, cy, abs(cy-Y))

                else:   # neu chua: save -> log, counter++
                    if Y - cy <= maxDistanceFromLine: 
                        AddToLog(cx,cy)
                        print ("AddToLog: ", len(listCertainLog))
                        ExistCounter +=1
        cv2.putText(frame_output, "CONTER: {}".format(str(ExistCounter)), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 1), 2)
        for i in listCertainLog:
            ObjectCentroid = (i.x,i.y)
            cv2.circle(frame_output, ObjectCentroid, 1, (255, 0, 255), 5)
        cv2.imshow("frame_output", frame_output)		
        cv2.imshow('frame input',(frame_origin))
        key = cv2.waitKey(1) & 0xFF
    else:
        break

    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
