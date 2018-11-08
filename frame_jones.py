import wx
import os
from skimage.color import rgb2gray
from skimage import data,io
import numpy as np
import cv2
import matplotlib.pyplot as plt
class frame_jones(wx.Frame):
	train_path='path'
	test_path='path'
	save_path='path'
	def __init__(self, parent, title):
	 super(frame_jones, self).__init__(parent, title=title, size=(800,600))
	 self.InitUI()
	 self.Centre(True)
	 self.Show()	

	def InitUI(self):
	   icono = wx.Icon("/root/Documentos/Social_Service/img/logo.png")
           self.SetIcon(icono)
	    
	   #select train data
	   self.label=wx.StaticText(parent=self,label="directory train data",pos=(10,50))
           self.textrain = wx.TextCtrl(parent=self,pos=(160,20),size=(480,80))
	   self.textrain.Enable(False)
	   self.train = wx.Button(parent=self,id=-1,label="Select",pos=wx.Point(670,45),size=wx.Size(100,30))
           self.train.Bind(wx.EVT_BUTTON, self.select_train)
	   #select test data
	   self.label=wx.StaticText(parent=self,label="directory test data",pos=(10,90))
           self.textest = wx.TextCtrl(parent=self,pos=(160,60),size=(480,80))
	   self.textest.Enable(False)
	   self.test = wx.Button(parent=self,id=-1,label="Select",pos=wx.Point(670,85),size=wx.Size(100,30))
           self.test.Bind(wx.EVT_BUTTON, self.select_test)
	   #select directory result data
	   self.label=wx.StaticText(parent=self,label="directory save results",pos=(10,130))
           self.texsave = wx.TextCtrl(parent=self,pos=(160,100),size=(480,80))
	   self.texsave.Enable(False)
	   self.save = wx.Button(parent=self,id=-1,label="Select",pos=wx.Point(670,125),size=wx.Size(100,30))
           self.save.Bind(wx.EVT_BUTTON, self.select_save)
	   #process
	   self.processing = wx.Button(parent=self,id=-1,label="Process",pos=wx.Point(350,180),size=wx.Size(100,30))
           self.processing.Bind(wx.EVT_BUTTON, self.process_files)
	   #text
           self.text = wx.TextCtrl(self,size = (600,300),pos=(100,230),style = wx.TE_MULTILINE)

	def select_train(self, event):
	     opendir=wx.DirDialog(self,'Select train directory','',style=wx.DD_DEFAULT_STYLE)
	     try:
		 if opendir.ShowModal()==wx.ID_CANCEL:
		    return
		 path=opendir.GetPath()
	     except Exception:
		 wx.LogError('Failed to select directory')
		 raise
	     finally:
		 opendir.Destroy()
	     if len(path)>0:#train path
		self.train_path=path
		self.textrain.SetValue(path)
		
	def select_test(self, event):
	     opendir=wx.DirDialog(self,'Select test directory','',style=wx.DD_DEFAULT_STYLE)
	     try:
		 if opendir.ShowModal()==wx.ID_CANCEL:
		    return
		 path=opendir.GetPath()
	     except Exception:
		 wx.LogError('Failed to select directory')
		 raise
	     finally:
		 opendir.Destroy()

	     if len(path)>0:#test path
		self.test_path=path
		self.textest.SetValue(path)
		
	        
	def select_save(self, event):
	     opendir=wx.DirDialog(self,'Select save directory','',style=wx.DD_DEFAULT_STYLE)
	     try:
		 if opendir.ShowModal()==wx.ID_CANCEL:
		    return
		 path=opendir.GetPath()
	     except Exception:
		 wx.LogError('Failed to select directory')
		 raise
	     finally:
		 opendir.Destroy()

	     if len(path)>0:#save path
		self.save_path=path
		self.texsave.SetValue(path)
			
	
	def process_files(self, event):
	    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	    ruta=self.save_path
	    self.text.AppendText("Processing train data"+"\n") 
	    data_dir=self.train_path
        directories=[d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d))]

	    dir_name=-1 #nombres enteros para los directorios
        for d in directories:
		self.text.AppendText("Directory: "+d+"\n")    
	    label_dir= os.path.join(data_dir,d)
		dir_name+=1
		os.makedirs(os.path.join(ruta+'/Train',str(0)+str(dir_name)))#crea directorio destino
		new_path=os.path.join(ruta+'/Train',str(0)+str(dir_name))#ruta para guardar las nuevas imagenes
		file_names =[os.path.join(label_dir,f) for f in os.listdir(label_dir)]
		j=0;##para nombres de imagenes
		for f in file_names:
		    self.text.AppendText("Files: "+f+"\n")    
	    	    j+=1
	    	    img=data.imread(f)#lee imagen
	    	    name=str(dir_name)+'_000'+str(j)+'.jpg'#nuevo formato de imagen
	    	    io.imsave(os.path.join(new_path,name),img)#guarda la imagen
		        #viola & jones
	            img = cv2.imread(os.path.join(new_path,name))
		        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		        for (x,y,w,h) in faces:
    			   img = img[y:y+h+1, x:x+w+1]
		        cv2.imwrite(os.path.join(new_path,name),img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])	
	     
	    ###processing test data
	    self.text.AppendText("Processing test data"+"\n") 
	    data_dir=self.test_path
            directories=[d for d in os.listdir(data_dir)#extrae todos los subdirectorios
	 	        if os.path.isdir(os.path.join(data_dir,d))]
	    dir_name=-1
            for d in directories:
		 self.text.AppendText("Directory: "+d+"\n")    
	     	 label_dir= os.path.join(data_dir,d)
		 dir_name+=1
		 os.makedirs(os.path.join(ruta+'/Test',str(0)+str(dir_name)))#crea directorio destino
		 new_path=os.path.join(ruta+'/Test',str(0)+str(dir_name))#ruta para guardar las nuevas imagenes
		 file_names=[os.path.join(label_dir,f)
		            for f in os.listdir(label_dir)]
		 j=0;##para nombres de imagenes
		 for f in file_names:
		     self.text.AppendText("Files: "+f+"\n")    
	    	     j+=1
	    	     img=data.imread(f)#lee imagen
	    	     name=str(dir_name)+'_000'+str(j)+'.jpg'#nuevo formato de imagen
	    	     io.imsave(os.path.join(new_path,name),img)#guarda la imagen
		     #viola & jones
	             img = cv2.imread(os.path.join(new_path,name))
		     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		     for (x,y,w,h) in faces:
    			 img = img[y:y+h+1, x:x+w+1]
		     cv2.imwrite(os.path.join(new_path,name),img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
 
	    self.text.AppendText("Finish !!"+"\n")  		
	
	    
def main():
  app=wx.App()
  fr=frame_jones(None,title='Viola & Jones')
  fr.Show()
  app.MainLoop()
if __name__ == '__main__':
   main()



