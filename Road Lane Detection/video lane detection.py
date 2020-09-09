#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np

#Load the video 
cap = cv2.VideoCapture(r"C:\Desktop\Downloads\lane.mp4")

#The dimensions of the video are obtained
width = int(cap.get(3)) 
height = int(cap.get(4))
print(width)
print(height)
size = (height, width) 

#Saving the video file

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter("Lane.avi",fourcc, 20.0, size)

#The main program that displays the video with lane detection
while cap.isOpened():
    ret,frame = cap.read()
    
    if ret == True:
        
#         print(frame.shape)
       
        #Convert each frame of the video into grayscale format
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #Blur the image so as to reduce the intensity of the image
        blur = cv2.GaussianBlur(gray,(7,7),2)
        
        #Canned the image to get the edges in a better way
        canned = cv2.Canny(blur, 150,60)
        
        #Create a black image similar to the size of the frame
        mask = np.zeros_like(blur)
        
        #Find the approximate vertices of the road in the frames of the video
        vertices = np.array([[(0,720),(0,450),(300,250),(800,250),(1280,700),(1280,720)]])
        
        #Fill the frames with the polygon obtained from the vertices
        mask = cv2.fillPoly(mask,vertices,(200,0,0))
#         plt.imshow(mask)
        
        #Bitwise and to obtain only the frame that comes under the vertices drawn
        masked_image = cv2.bitwise_and(canned,mask)
        
        
        #Hough transformation to obtain the lines and draw them on the frame
        lines = cv2.HoughLinesP(masked_image,1,np.pi/180, 20, minLineLength=70, maxLineGap = 8)

        for line in lines:
            line_image = np.zeros((masked_image.shape[0],masked_image.shape[1]))
            for x1, y1, x2, y2 in line:
                cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),3)
        
        #Display the frames with the lanes detected on the video
        
        cv2.imshow('Frames',frame)
#         out.write(frame)
        
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()


# In[ ]:




