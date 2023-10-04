import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

model = load_model('my_new_model.keras')
cap = cv2.VideoCapture(0)
score=0
rpred = []
lpred = []

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    right_eye =  reye.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
        r_eye=frame[y:y+h,x:x+w]
        r_eye = cv2.resize(r_eye,(299,299),3)
        r_eye= r_eye/255.0
        r_eye = np.array(r_eye)
        r_eye.shape
        r_eye=  np.reshape(r_eye,[1,299,299,3])
        rpred = model.predict(r_eye)
        break

    for (x,y,w,h) in left_eye:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        l_eye=frame[y:y+h,x:x+w]
        l_eye = cv2.resize(l_eye,(299,299),3)
        l_eye= np.array(l_eye/255.0)
        l_eye=np.reshape(l_eye,[1,299,299,3])
        lpred = model.predict(l_eye)
        break

    if(rpred[0]<=0.5 and lpred[0]<=0.5):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),1,cv2.LINE_AA)
    elif(rpred[0]>0.5 and lpred[0]>0.5):
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),1,cv2.LINE_AA)
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>7):
        try:
            sound.play()
        except:
            pass
    cv2.imshow('Frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release() 
cv2.destroyAllWindows()