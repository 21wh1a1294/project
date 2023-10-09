import numpy as np 
import cv2
net = cv2.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt','colorization_release_v2.caffemodel')
pts = np.load('pts_in_hull.npy')
image = cv2.imread('area_bw.jpeg')
scaled = image.astype("float32")/255.0
lab = cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB)
