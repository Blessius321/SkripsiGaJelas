import cv2 
import torch 
import cv2
from Deployment.usefulFunctions import *


label = ['kiri atas', 'kanan atas', 'kiri bawah', 'kanan bawah', "unkown"]

thresh = 0
mov_avrg = []
class_mov_avrg = []

def init():
    global thresh 
    thresh = calibration()

def findGaze():
    with torch.no_grad():
        (rects, gray, image) = getRect()
        if len(rects)>0:
            mov_avrg.append(isBlinking(getEyes(rects, gray, image, isShape=True), thresh))
            if len(mov_avrg) > 5:
                mov_avrg.pop(0)
            if sum(mov_avrg)/len(mov_avrg) < 0.2 :
                mata = getEyes(rects, gray, image, showFrames =False)
                muka = getFace(rects, image, showFrames = False)
                if torch.is_tensor(mata) and torch.is_tensor(muka):
                    output = model(mata, muka)
                    prediction = torch.max(output, 1)
                    class_mov_avrg.append(int(prediction.indices))
                    if len(class_mov_avrg) > 5:
                        class_mov_avrg.pop(0)
                    print(label[round(sum(class_mov_avrg) / len(class_mov_avrg))])
                    return round(sum(class_mov_avrg) / len(class_mov_avrg))
                else: 
                    pass
            else:
                print("eye is closed")
                return -1


