import cv2
import mediapipe as mp
import time
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import ctypes
import numpy as np
import math

device = AudioUtilities.GetSpeakers()
interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))

cap=cv2.VideoCapture(0)
mpHands=mp.solutions.hands
mainfr=mpHands.Hands(min_detection_confidence=0.7)
mpDraw=mp.solutions.drawing_utils

pTime=0
volB=400

while True:
    list1=[]
    success,img=cap.read()
    imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=mainfr.process(imgrgb)
    if results.multi_hand_landmarks:
        for i in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,i,mpHands.HAND_CONNECTIONS)
            for id,val in enumerate(i.landmark):
                h,w,c=img.shape
                cx,cy=int (w*val.x),int(h*val.y)
                list1.append((cx,cy))
                # print(f'{id}) {cx}, {cy}')
                if (id==4 or id ==8):
                    cv2.circle(img,(cx,cy),12,(255,0,255),cv2.FILLED)
                    

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    perc=0
    if len(list1)!=0:
        cv2.line(img,(list1[4][0],list1[4][1]),(list1[8][0],list1[8][1]),(255,0,255),2)
        cx,cy=(list1[4][0]+list1[8][0])//2,(list1[4][1]+list1[8][1])//2
        cv2.circle(img,(int(cx),int(cy)),12,(255,0,255),cv2.FILLED)
        length=math.hypot(list1[4][0]-list1[8][0],list1[4][1]-list1[8][1])
        Vol1=volume.GetVolumeRange()
        minVol=Vol1[0]
        maxVol=Vol1[1]
        vol=np.interp(length,[20.0,300.0],[minVol,maxVol])
        volB=np.interp(length,[20.0,300.0],[400,150])
        perc=np.interp(length,[20.0,300.0],[0,100])
        print(vol)
        volume.SetMasterVolumeLevel(vol,None)
    
    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv2.rectangle(img,(50,int(volB)),(85,400),(0,255,0),cv2.FILLED)
    cv2.putText(img,f'{int(perc)}%',(50,130),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.putText(img,f'FPS : {int(fps)}',(30,60),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    cv2.imshow('Image',img)
    cv2.waitKey(1)