#What we are going to do
# READING INPUT IMAGE -> GETTING BLOB -> LOADING YOLOV3 ->
# IMPLEMENTING FORWARD PASS -> GETTING BUNDING BOXES ->
# NON MAXIMUM SUPRESSION -> DRAWING BOUNDING BOXES WITH LABELS

import numpy as np
import cv2
import time

#Reading Image

img = cv2.imread("images\woman-working-in-the-office.jpg")
#img = cv2.imread("D:\Pictures\image.jpg")
cv2.namedWindow("OriginalImage",cv2.WINDOW_NORMAL)
cv2.imshow("OriginalImage",img)
cv2. waitKey(0)
cv2.destroyWindow("OriginalImage")

# print("Image Shape", img.shape)
h, w = img.shape[:2] #getting height n width of the image using slicing

#Using Blob on image (BLOB-> Binary Large OBject)
# Getting blob from input image
# The 'cv2.dnn.blobFromImage' function returns 4-dimensional blob
# from input image after mean subtraction, normalizing, and RB channels swapping
# Resulted shape has number of images, number of channels, width and height

blob = cv2.dnn.blobFromImage(img,1/255.0,(416,416),swapRB=True,crop=False)
# print(blob.shape) #Shape: (1,3,416,416)
blob_to_show = blob[0,:,:,:].transpose(1,2,0) #Shape : (416,416,3)

cv2.namedWindow("Blob Image", cv2.WINDOW_NORMAL)
cv2.imshow("Blob Image",cv2.cvtColor(blob_to_show,cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyWindow("Blob Image")

#Importing YOLO
#Data available for download
# Available at: https://www.kaggle.com/datasets/valentynsichkar/yolo-coco-data

with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]

# print(labels)

yolo = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                  'yolo-coco-data/yolov3.weights')
all_layers = yolo.getLayerNames()
UnconnectedOutLayers = yolo.getUnconnectedOutLayers()
output_layers = \
    [all_layers[i-1] for i in yolo.getUnconnectedOutLayers()]

# Setting minimum probability to eliminate weak predictions
probability_minimum = 0.5

# Setting threshold for filtering weak bounding boxes
threshold = 0.3

colours = np.random.randint(0,255,size=(len(labels),3),dtype='uint8')

# Implementing forward pass

yolo.setInput(blob)
start = time.time()
output = yolo.forward(output_layers)
end = time.time()
# print(end-start)

#Preparing bounding boxes

bounding_boxes = []
confidences = []
class_numbers = []


for result in output:
    for detected_objects in result:
        scores = detected_objects[5:]
        class_current = np.argmax(scores)
        confidence_current = scores[class_current]


        if confidence_current>probability_minimum:
            box_current = detected_objects[0:4] *np.array([w,h,w,h])
            x_center,y_center,width,height = box_current
            x_min = int(x_center-(width/2))
            y_min = int(y_center-(height/2))

            bounding_boxes.append([x_min,y_min,int(width),int(height)])
            confidences.append(float(confidence_current))
            class_numbers.append(class_current)

# print(bounding_boxes)
# print(confidences)

#Non maximum suppression
# Sometimes there are more than 1 bounding box
# so in this technique we exclude that bounding box
# which has low confidence than other and take bounding
# box with highest confidence in consideration

result_nms = cv2.dnn.NMSBoxes(bounding_boxes,confidences,probability_minimum
                              ,threshold)

counter = 1

#Implementing bounding box over image

if len(result_nms)>0:
    for i in result_nms.flatten():
        print(counter,labels[int(class_numbers[i])])

        counter+=1

        x_min,y_min,width,height = bounding_boxes[i][0],bounding_boxes[i][1],bounding_boxes[i][2],bounding_boxes[i][3]

        color_box_current = colours[class_numbers[i]].tolist()
        cv2.rectangle(img, (x_min, y_min),
                      (x_min + width, y_min + height),
                      color_box_current, 2)
        #Putting label
        text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                               confidences[i])

        # Putting text with label and confidence on the original image
        cv2.putText(img, text_box_current, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, color_box_current, 2)

cv2.namedWindow("DetectedObject",cv2.WINDOW_NORMAL)
cv2.imshow("DetectedObject",img)
cv2.waitKey(0)
cv2.destroyWindow("DetectedObject")
