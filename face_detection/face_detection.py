import cv2 as cv

# https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)
while True:
  _, img = cap.read()
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.1, 4)
  for(x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
  cv.imshow('img', img)
  k = cv.waitKey(30)
  if k == 27: # 27 es el ascii para esc
    break
cap.release()