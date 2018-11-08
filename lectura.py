
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from skimage import io
from scipy import misc

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = io.imread('/home/jorge/Documentos/Topicos/subject11.surprised')
#lena = misc.imread('img3.JPG')

io.imsave("lena.JPG", img)
#plt.imshow(img)
#plt.show()

img = cv2.imread('/home/jorge/Documentos/Topicos/lena.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]
	eyes = eye_cascade.detectMultiScale(roi_gray)
	for (ex,ey,ew,eh) in eyes:
		cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


io.imsave("lena.JPG", img)

#cv2.imshow('img',img2)
#cv2.waitKey(0)

io.imshow(img)
io.show()

#plt.imshow(l, cmap=plt.cm.gray)
#plt.show()
