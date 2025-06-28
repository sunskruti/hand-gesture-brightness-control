
import cv2 
import mediapipe as mp 
from math import hypot 
import screen_brightness_control as sbc 
import numpy as np 


mpHands = mp.solutions.hands 
hands = mpHands.Hands() 

Draw = mp.solutions.drawing_utils  
cap = cv2.VideoCapture(0) 

while True: 
	_, frame = cap.read() 
	frame = cv2.flip(frame, 1) 
	frameR = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
	Process = hands.process(frameR) 

	landmarkList = [] 
	 
	if Process.multi_hand_landmarks: 
		for handlm in Process.multi_hand_landmarks: 
			for ida, landmarks in enumerate(handlm.landmark): 
				height, width, color_channels = frame.shape 
				x, y = int(landmarks.x*width), int(landmarks.y*height) 
				landmarkList.append([ida, x, y]) 

			
			Draw.draw_landmarks(frame, handlm,mpHands.HAND_CONNECTIONS) 

	if landmarkList != []:  
		x1, y1 = landmarkList[4][1], landmarkList[4][2]  
		x2, y2 = landmarkList[8][1], landmarkList[8][2] 

		cv2.circle(frame, (x1, y1), 10, (0,255,0), thickness=-1) 
		cv2.circle(frame, (x2, y2), 10, (0,255,0), thickness=-1) 
		cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 3) 
		 
		L = hypot(x2-x1, y2-y1) 
 
		b = np.interp(L, [15, 220], [0, 100]) 

		
		sbc.set_brightness(int(b)) 

	
	cv2.imshow('Image', frame) 
	if cv2.waitKey(1) & 0xff == ord('r'): 
		break
