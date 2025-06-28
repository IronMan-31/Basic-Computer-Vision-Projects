import cv2
import mediapipe as mp
import time
import os

cap=cv2.VideoCapture(0)
mpHands=mp.solutions.hands
mainfr=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils 
imgList=os.listdir('VolumeGesture/images')
mainImg=[]
for i in imgList:
    main=cv2.imread(f'VolumeGesture/images/{i}')
    mainImg.append(main)

pTime=0
while True:
    coList=[]
    success,img=cap.read()
    imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=mainfr.process(imgrgb)
    if results.multi_hand_landmarks:
        for i in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,i,mpHands.HAND_CONNECTIONS)
            for id,val in enumerate(i.landmark):
                h,w,c=img.shape
                cx,cy=int(w*val.x),int(h*val.y)
                coList.append((cx,cy))
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    x,y=20,50
    Counter=0
    if (len(coList)!=0):
        x,y=2,200    
        if (coList[4][0]<coList[2][0]):
            img[0:128, 0:128]=mainImg[4]
            Counter=5
        elif (coList[20][1]<coList[18][1]):
            img[0:128, 0:128]=mainImg[3]
            Counter=4
        elif (coList[16][1]<coList[14][1]):
            img[0:128, 0:128]=mainImg[2]
            Counter=3
        elif (coList[12][1]<coList[10][1]):
            img[0:128, 0:128]=mainImg[1]
            Counter=2
        elif (coList[8][1]<coList[6][1]):
            img[0:128, 0:128]=mainImg[0]
            Counter=1
        else:
            img[0:128,0:128]=mainImg[5]
    cv2.putText(img,f'FPS : {int(fps)}',(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    cv2.putText(img,f'Fingers : {Counter}',(x,y+50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    cv2.imshow('Img',img)
    cv2.waitKey(1)