#What we are going to do
# READING INPUT IMAGE -> GETTING BLOB -> LOADING YOLOV3 ->
# IMPLEMENTING FORWARD PASS -> GETTING BUNDING BOXES ->
# NON MAXIMUM SUPRESSION -> DRAWING BOUNDING BOXES WITH LABELS

import numpy as np
import cv2
import time

#Reading Image

img = cv2.imread("images\woman-working-in-the-office.jpg")

# cv2.namedWindow("OriginalImage",cv2.WINDOW_NORMAL)
# cv2.imshow("OriginalImage",img)
# cv2. waitKey(0)
# cv2.destroyWindow("OriginalImage")

# print("Image Shape", img.shape)
h, w = img.shape[:2] #getting height n width of the image using slicing

#Using Blob on image (BLOB-> Binary Large OBject)
# Getting blob from input image
# The 'cv2.dnn.blobFromImage' function returns 4-dimensional blob
# from input image after mean subtraction, normalizing, and RB channels swapping
# Resulted shape has number of images, number of channels, width and height
# E.G.:
# blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)

blob = cv2.dnn.blobFromImage(img,1/255.0,(416,416),swapRB=True,crop=False)
# print(blob.shape) #Shape: (1,3,416,416)
blob_to_show = blob[0,:,:,:].transpose(1,2,0) #Shape : (416,416,3)

# cv2.namedWindow("Blob Image", cv2.WINDOW_NORMAL)
# cv2.imshow("Blob Image",cv2.cvtColor(blob_to_show,cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyWindow("Blob Image")

#Importing YOLO
#Data available for download
# Available at: https://www.kaggle.com/datasets/valentynsichkar/yolo-coco-data

with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]

# print(labels)

yolo = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                  'yolo-coco-data/yolov3.weights')
all_layers = yolo.getLayerNames()






