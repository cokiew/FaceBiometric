import numpy as np 
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('/home/jorge/Documentos/Topicos/5_0003.jpg')
cv2.imshow('original',img)
#Se convierte a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
	img = img[y:y+h+1, x:x+w+1]
	
cv2.imwrite("new.jpg",img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])		
cv2.imshow('img',img)
cv2.waitKey(0)

