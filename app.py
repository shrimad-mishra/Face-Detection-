import cv2 as cv
import numpy as np

face_detector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eyes_detector = cv.CascadeClassifier("haarcascade_eye.xml")

capture = cv.VideoCapture(0) #Replace 0 with 1 if you have externally USB attached camera
capture.set(3,640) #set width (3 represents width)
capture.set(4,480) #set height (4 represents height)
capture.set(10,70) #set brightness (10 represents brightness property)
while True:
    is_success, frame = capture.read()
    frame = cv.flip(frame,180)
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray_frame,scaleFactor=1.09,minNeighbors=10) #you can play with last 2 parameters to adjust
    eyes = eyes_detector.detectMultiScale(gray_frame,scaleFactor=1.15,minNeighbors=15) # the performance of model
    for (x,y,width,height) in faces:
        face_rect = cv.rectangle(frame,(x,y),(x+width,y+height),(255,0,0),2)
    for (x_,y_,width_,height_) in eyes:
        eye_rect = cv.rectangle(frame, (x_,y_),(x_+width_,y_+height_),(0,255,0),3)

    cv.imshow("Real Time Face Detection",frame)
    if cv.waitKey(1) & 0xFF == ord('e'): # Press e to exit program
        break

capture.release()
cv.destroyAllWindows()