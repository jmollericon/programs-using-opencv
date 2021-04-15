import cv2 as cv

# https://github.com/andrewssobral/vehicle_detection_haarcascades
car_cascade = cv.CascadeClassifier('cars.xml')

cap = cv.VideoCapture('cars_video.mp4')

while True:
  _, img = cap.read()
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  cars = car_cascade.detectMultiScale(gray, 1.1, 4)
  for(x, y, w, h) in cars:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
  cv.imshow('img', img)
  k = cv.waitKey(30)
  if k == 27: # 27 es el ascii para esc
    break
cap.release()