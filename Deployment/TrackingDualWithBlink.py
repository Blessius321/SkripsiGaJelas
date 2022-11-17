from math import floor
import os
import cv2 
from imutils import face_utils
import dlib
import cv2
import numpy as np
from dualModel import DualModel
from torch.optim import Adam
from torch import nn 
from sklearn.model_selection import train_test_split
import numpy as np 
import torch 
import os
import cv2
from time import time 
from scipy.spatial import distance as dist
from usefulFunctions import *


label = ['kiri atas', 'kanan atas', 'kiri bawah', 'kanan bawah']

getTime = []
cnnTime = []
facialLandmarkTime = []
totalTime = []

thresh = calibration()
mov_avrg = []
class_mov_avrg = []

with torch.no_grad():
    while True:
        # timer2 = time()
        (rects, gray, image) = getRect()
        # facialLandmarkTime.append(time() - timer2)
        if len(rects) > 0:
            mov_avrg.append(isBlinking(getEyes(rects, gray, image, isShape=True), thresh))
            if len(mov_avrg) > 5:
                mov_avrg.pop(0)
            if sum(mov_avrg)/len(mov_avrg) < 0.2 :
                mata = getEyes(rects, gray, image)
                muka = getFace(rects, image)
                # getTime.append(time() - timer)
                output = model(mata, muka)
                prediction = torch.max(output, 1)
                # cnnTime.append(time() - timer - getTime[-1])
                class_mov_avrg.append(int(prediction.indices))
                if len(class_mov_avrg) > 10:
                    class_mov_avrg.pop(0)
                print(label[round(sum(class_mov_avrg) / len(class_mov_avrg))])
                # totalTime.append(time() - timer)
            else:
                print("eye is closed")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

print(f"average getTime = {sum(getTime)/len(getTime)}")
print(f"average cnnTime = {sum(cnnTime)/len(cnnTime)}")
print(f"average facialLandmarkTime = {sum(facialLandmarkTime)/len(facialLandmarkTime)}")
print(f"average FPS = {1/(sum(totalTime)/len(totalTime))}")
            

