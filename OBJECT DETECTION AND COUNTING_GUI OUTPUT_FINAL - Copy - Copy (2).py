import cv2
import numpy as np
import sys
import collections
from tkinter import filedialog
from tkinter import *
c=0
l=[]
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    global c
    c=c+1
    label = str(classes[class_id])
    l.append(label)
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
# read input image
i=filedialog.askopenfilename(filetypes=[('JPG','.jpg'),('JPEG','.jpeg')])
image = cv2.imread(i)
#image = cv2.VideoCapture(0)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392
classes = None
with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")

blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
net.setInput(blob)
outs = net.forward(get_output_layers(net))
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
for i in indices:
    #i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

# display output image    
ws=(1000,600)
y,x,_=image.shape
print('Objects Detected - > ',str(dict(collections.Counter(l))).replace('{','').replace('}',''))
print("Count is : ",c)
image=cv2.putText(image,'Count is {}, Objects Count are {}'.format(c,str(dict(collections.Counter(l))).replace('{','').replace('}','')),(int(x/135),int(y/1.01)),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,0,0))
cv2.imshow("Object Detection", cv2.resize(image,ws))

while True:
    key=cv2.waitKey(1)
    if key==27:
        cv2.imwrite("object-detection.jpg", image)
        cv2.destroyAllWindows()
        break
sys.exit()
