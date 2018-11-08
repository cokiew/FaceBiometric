import os
import numpy as np
from skimage import data,io 
import cv2

def archivos():#Se ira recorriendo el sistema de archivos de entrenamiento
	"""for base, dirs, files in os.walk('/home/jorge/Documentos/Topicos/'):
		directorios = [base] """
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	print("Procesando datos de entrenamiento")
	dirtraining = "/home/jorge/Documentos/yalefaces/Training/"
	nuevaruta = "/home/jorge/Documentos/Topicos/"
	directorios = [name for name in os.listdir(dirtraining) if os.path.isdir(os.path.join(dirtraining, name))]
	#print(directorios[0])

	itemdir = -1
	for d in directorios:
		label_dir = os.path.join(dirtraining,d) #label_dir estan las direcciones de cada carpeta de entrenamiento 
		#print(label_dir)
		itemdir += 1
		#print(itemdir)
		os.makedirs(os.path.join(nuevaruta+'/Training',str(0)+str(itemdir)))#Se crean los nuevos directorios donde se guardaran las imagenes  
		dir_actual = os.path.join(nuevaruta+'/Training',str(0)+str(itemdir))#Direccion donde se guardan las imagenes
		archivos = [os.path.join(label_dir,f) for f in os.listdir(label_dir)]
		#print(archivos)
		m = 0
		for j in archivos:
			m +=1
			img=data.imread(j)#lee imagen
			name=str(itemdir)+'_00'+str(m)+'.jpg'#nuevo formato de imagen
			io.imsave(os.path.join(dir_actual,name),img)#Se guarda la imagen en el nuevo direcotorio
			#Aplicacion de viola jones
			img = cv2.imread(os.path.join(dir_actual,name))
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)
			for (x,y,w,h) in faces:
				img = img[y:y+h+1, x:x+w+1]
				cv2.imwrite(os.path.join(dir_actual,name),img,[int(cv2.IMWRITE_JPEG_QUALITY),100])	#Aqui se reescribe la imagen ya segmentada la imagen 
    
	#Segmentacion de las imagenes para testeo
	print("Porcesando archivos de testeo")
	dirtraining2 = "/home/jorge/Documentos/yalefaces/Testing/"
	directorios = [name for name in os.listdir(dirtraining2) if os.path.isdir(os.path.join(dirtraining2, name))]
	#print(directorios)
	itemdir = -1
	for d in directorios:
		label_dir = os.path.join(dirtraining2,d) #label_dir estan las direcciones de cada carpeta de entrenamiento 
		#print(label_dir)
		itemdir += 1
		#print(itemdir)
		os.makedirs(os.path.join(nuevaruta+'/Testing',str(0)+str(itemdir)))#Se crean los nuevos directorios donde se guardaran las imagenes  
		dir_actual = os.path.join(nuevaruta+'/Testing',str(0)+str(itemdir))#Se obtiene las Direcciones de los ficheros donde se guardan las imagenes
		archivos = [os.path.join(label_dir,f) for f in os.listdir(label_dir)]
		#print(archivos)
		m = 0
		for j in archivos:
			m+=1
			img=data.imread(j)#lee imagen
			name=str(itemdir)+'_00'+str(m)+'.jpg'#nuevo formato de imagen
			io.imsave(os.path.join(dir_actual,name),img)#Se guarda la imagen en el nuevo direcotorio
			#Se aplica viola jones
			img = cv2.imread(os.path.join(dir_actual,name))
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)
			for (x,y,w,h) in faces:
				img = img[y:y+h+1, x:x+w+1]
				cv2.imwrite(os.path.join(dir_actual,name),img,[int(cv2.IMWRITE_JPEG_QUALITY),100])	#Aqui se reescribe la imagen ya segmentada la imagen

archivos() # Funcion que realiza la segmentacion de las imagenes