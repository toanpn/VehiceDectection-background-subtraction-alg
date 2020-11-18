import numpy as np
import cv2


class BackGroundSubtractor:
	
	def __init__(self,alpha,firstFrame):
		self.alpha  = alpha
		self.backGroundFrame = firstFrame

	def getForeground(self,frame):
		
		# background = curentFrame * alpha + preBackground * (1 - alpha)
		self.backGroundFrame =  frame * self.alpha + self.backGroundFrame * (1 - self.alpha)
        # get Mask
        # case float -> unit8
		cv2.imshow('self.backGroundFrame.astype(np.uint8)', self.backGroundFrame.astype(np.uint8))
		return cv2.absdiff(self.backGroundFrame.astype(np.uint8),frame)

cam = cv2.VideoCapture("E:\\LUAN VAN\\repo\\VIDEO\\video1.mp4")
# cam = cv2.VideoCapture(0)
KERNEL_WIDTH = 9
KERNEL_HEIGHT = 9
SIGMA_X = 4
SIGMA_Y = 4

IsViewMode = True
alpha = 0.007

# gaussian cho frame 
def denoise(frame):
    # frame = cv2.medianBlur(frame,5)
    # frame = cv2.GaussianBlur(frame,(5,5),0)
    frame = cv2.GaussianBlur(frame, ksize=(KERNEL_WIDTH, KERNEL_HEIGHT), sigmaX=SIGMA_X, sigmaY=SIGMA_Y)
    return frame


ret,frame = cam.read()
if ret is True:
	if not IsViewMode:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	backSubtractor = BackGroundSubtractor(alpha,denoise(frame))
	run = True
else:
	run = False

while(run):
	ret,frame = cam.read()

	if ret is True:
		if not IsViewMode:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow('input',denoise(frame))

		# get the foreground
		foreGround = backSubtractor.getForeground(denoise(frame))

		foreGround_toView = backSubtractor.getForeground(frame)
        # View image and  
		# cv2.imshow('foreGround_toView', foreGround_toView)
		# cv2.imshow('foreGround',frame + foreGround_toView)

		# Apply thresholding on the background and display the resulting mask
		ret, mask = cv2.threshold(foreGround, 15, 255, cv2.THRESH_BINARY)

		cv2.imshow('mask',mask)

		key = cv2.waitKey(10) & 0xFF
	else:
		break

	if key == 27:
		break

cam.release()
cv2.destroyAllWindows()