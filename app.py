import cv2
from deepface import DeepFace
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Loading Face Cascade file from directory
face_cascade = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)


while True:
    rat, frame = cap.read()

    
    # Convert gray scale image for face detection
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

    # Face detection using cascade file
    faces = face_cascade.detectMultiScale(gray,2.1,2)

    # Prediction of Emotion using DeepFace
    prediction = DeepFace.analyze(frame,actions=['emotion'],enforce_detection=False)

    # Draw rectangle around the faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    # Put Text in the image
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(frame,prediction[0]['dominant_emotion'],(100,100),font,2,(0,0,255),2,cv2.LINE_4)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

    cv2.imshow("My frame", frame)

cap.release()
cv2.destroyAllWindows()