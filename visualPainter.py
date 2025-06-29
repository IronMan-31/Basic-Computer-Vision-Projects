import cv2
import mediapipe as mp
import numpy as np

cap=cv2.VideoCapture(0)
thickness=5
mpHands=mp.solutions.hands
mainfr=mpHands.Hands(min_detection_confidence=0.80)
mpDraw=mp.solutions.drawing_utils
imgList=['1.png','2.png','3.png','4.png']
mainImg=[]
headerImg=0
mainCol=(255,0,255)
for i in imgList:
    im=cv2.imread(f'volumeGesture/headerimages/{i}')
    mainImg.append(im)
imgCanvas=np.zeros((720,1080,3),np.uint8)
xp,yp=0,0
while True:
    coList=[]
    success,img=cap.read()
    img=cv2.flip(img,1)
    img=cv2.resize(img,(1080,720))
    img[0:125,0:1080]=mainImg[headerImg]
    imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=mainfr.process(imgrgb)
    if results.multi_hand_landmarks:
        for i in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,i,mpHands.HAND_CONNECTIONS)
            for val in i.landmark:
                h,w,c=img.shape
                cx,cy=int(w*val.x),int(h*val.y)
                coList.append((cx,cy))

    if len(coList)!=0:
        x1,y1=coList[8][0],coList[8][1]
        x2,y2=coList[12][0],coList[12][1]
        if (y2<coList[10][1]):
            cv2.rectangle(img,(x1,y1),(x2,y2),mainCol,cv2.FILLED)
            if y2<125:
                if 250<x2<450:
                    headerImg=0
                    mainCol=(255,0,255)
                elif 550<x2<750:
                    headerImg=1
                    mainCol=(0,255,0)
                elif 800<x2<950:
                    headerImg=2
                    mainCol=(255,50,0)
                elif 1000<x2<1200:
                    headerImg=3
                    mainCol=(0,0,0)
                    thickness=5
            xp,yp=0,0
        else:
            cv2.circle(img,(x1,y1),15,mainCol,cv2.FILLED)
            if xp==0 and yp==0:
                xp,yp=x1,y1
            if mainCol==(0,0,0):
                thickness=30
            else:
                thickness=5
            cv2.line(imgCanvas,(xp,yp),(x1,y1),mainCol,thickness)
            xp,yp=x1,y1
    imgin=cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _,imginv=cv2.threshold(imgin,50,255,cv2.THRESH_BINARY_INV)
    imginv=cv2.cvtColor(imginv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imginv)
    img=cv2.bitwise_or(img,imgCanvas)
    cv2.imshow('Img',img)
    cv2.waitKey(1)