import cv2
import mediapipe as mp
import time
import math
import numpy as np

def FindAngle(img,p1,p2,p3,list1,draw=True):
    x1,y1=list1[p1]
    x2,y2=list1[p2]
    x3,y3=list1[p3]
    if draw:
        cv2.line(img,(x1,y1),(x2,y2),(255,255,255),2)
        cv2.line(img,(x2,y2),(x3,y3),(255,255,255),2)
        cv2.circle(img,(x1,y1),10,(0,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),10,(0,0,255),cv2.FILLED)
        cv2.circle(img,(x3,y3),10,(0,0,255),cv2.FILLED)
        cv2.circle(img,(x1,y1),15,(0,0,255),2)
        cv2.circle(img,(x2,y2),15,(0,0,255),2)
        cv2.circle(img,(x3,y3),15,(0,0,255),2)
    angle=math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))
    if (p1%2!=0):
        angle=360-angle
    cv2.putText(img,str(int(angle)),(x2-50,y2+50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),1)
    return angle
cap=cv2.VideoCapture('poses/pose2.mp4')
mpPose=mp.solutions.pose
mainfr=mpPose.Pose()
mpDraw=mp.solutions.drawing_utils
count=0
dir=0
pTime=0

while True:
    coList=[]
    strDir=""
    success,img=cap.read()
    img=cv2.resize(img,(980,800))
    imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=mainfr.process(imgrgb)
    if (results.pose_landmarks):
        # mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,val in enumerate(results.pose_landmarks.landmark):
            h,w,c=img.shape
            cx,cy=int(w*val.x),int(h*val.y)
            coList.append((cx,cy))
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS : {int(fps)}',(20,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    if (len(coList)!=0):
        angle=FindAngle(img,12,14,16,coList)
        range=np.interp(angle,(73,137),(0,100))
        if range==100:
            if dir==0:
                count+=0.5
                dir=1
        if range==0:
            if dir==1:
                count+=0.5
                dir=0
        if (dir==0):
            strDir="Going Down"
        else:
            strDir="Going Up"
        height=np.interp(range,(0,100),(350,600))
        cv2.putText(img,f'Curls : {int(count)}',(20,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(img,strDir,(20,150),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(img,f'{int(abs(range-100))}%',(20,320),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.rectangle(img,(20,600),(40,350),(0,255,0),2)
        cv2.rectangle(img,(20,600),(40,int(height)),(0,255,0),cv2.FILLED)
    cv2.imshow('Img',img)
    cv2.waitKey(1)