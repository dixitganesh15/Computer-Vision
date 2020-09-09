#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2


# In[2]:


img = cv2.imread(r'yolo004.png')
img.shape


# In[3]:


width = int(img.shape[1])
height = int(img.shape[0])


# In[4]:


img =cv2.resize(img,(width,height))


# In[5]:


net = cv2.dnn.readNet('yolov3-608.weights','yolov3-608.cfg')


# In[6]:


classes = []
with open('coco.names.txt','r')as f:
    classes = f.read().splitlines()


# In[7]:


# classes


# In[8]:


blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB = False, crop = False)


# In[9]:


# for each in blob:
#     for n, image in enumerate(each):
#         cv2.imshow(str(n), image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


# In[10]:


net.setInput(blob)
output_layers = net.getUnconnectedOutLayersNames()


# In[11]:


print(output_layers)


# In[12]:


layerOutputs = net.forward(output_layers)


# In[13]:


print(layerOutputs)


# In[14]:


boxes = []
confidences = []
class_ids = []

for each in layerOutputs:
    for detection in each:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.5:
            center_x = int(detection[0] * width) 
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            x = int(center_x - (w/2))
            y = int(center_y - (h/2))
            
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
print(len(boxes))
            
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.4)
print(indexes.flatten())

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0,255,size = (len(boxes), 3))

# print(indexes.flatten())

for i in indexes.flatten():
    x,y,w,h = boxes[i]
    label = str(class_ids[i])
    j = int(label)
    confidence = str(round(confidences[i],2))
    color = colors[i]
    cv2.rectangle(img, (x,y), (x+w, y+h),color,2 )
    cv2.putText(img, classes[j]+" "+ confidence+'%', (x,y-20), font, 1,(255,255,255),2)


# In[15]:


cv2.imshow("Yolo Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[16]:


cv2.imwrite('Object detected.jpg',img)

