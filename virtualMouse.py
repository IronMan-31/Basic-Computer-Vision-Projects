import cv2
import mediapipe as mp
import numpy as np
import autopy as ap
import time

cap=cv2.VideoCapture(0)
mpHands=mp.solutions.hands
mainfr=mpHands.Hands(max_num_hands=1,min_detection_confidence=0.6,)
mpDraw=mp.solutions.drawing_utils
pTime=0
wScreen,hScreen=ap.screen.size()
frameR=100
smoothening=7
prevX,prevY=0,0

while True:
    xList=[]
    yList=[]
    _,img=cap.read()
    img=cv2.flip(img,1)
    img=cv2.resize(img,(640,480))
    imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=mainfr.process(imgrgb)
    if results.multi_hand_landmarks:
        for i in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,i,mpHands.HAND_CONNECTIONS)
            for val in i.landmark:
                h,w,c=img.shape
                cx,cy=int(w*val.x),int(h*val.y)
                xList.append(cx)
                yList.append(cy)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    h,w,_=img.shape
    cv2.putText(img,f'FPS : {int(fps)}',(20,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    cv2.rectangle(img,(frameR,frameR),(w-frameR,h-frameR),(255,0,255),1)
    if len(xList)!=0:
        x1,y1=xList[8],yList[8]
        x2,y2=xList[12],yList[12]
        if (y2<yList[10]):
            cv2.circle(img,(x2,y2),10,(255,0,255),cv2.FILLED)
            cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)
            length=np.hypot(y1-y2,x1-x2)
            if length<40:
                ap.mouse.click()
        elif (y2>yList[10]):
            x3=np.interp(x1,(frameR,w-frameR),(0,wScreen))
            x3=min(wScreen-1,x3)
            x3=prevX+(x3-prevX)/smoothening
            y3=np.interp(y1,(frameR,h-frameR),(0,hScreen))
            y3=min(y3,hScreen-1)
            y3=prevY+(y3-prevY)/smoothening
            ap.mouse.move(x3,y3)
            cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)
            prevX=x3
            prevY=y3
        cv2.rectangle(img,(min(xList)-20,min(yList)-20),(max(xList)+20,max(yList)+20),(0,255,0),2)
    cv2.imshow('Img',img)
    cv2.waitKey(1)