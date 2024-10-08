import numpy as np
import cv2
import matplotlib.pyplot as plt
face_cascade = cv2.CascadeClassifier('object\haarcascade_frontalface_alt.xml')
def detect_face(img):


    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img,2.1)

    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10)

    return face_img
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read(0)

    frame = detect_face(frame)

    cv2.imshow('Video Face Detection', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()