import cv2 
import torch 
import cv2
from Deployment.usefulFunctions import *


label = ['kiri atas', 'kanan atas', 'kiri bawah', 'kanan bawah', "unkown"]

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
                if torch.is_tensor(mata) and torch.is_tensor(muka):
                    output = model(mata, muka)
                    prediction = torch.max(output, 1)
                    class_mov_avrg.append(int(prediction.indices))
                    if len(class_mov_avrg) > 5:
                        class_mov_avrg.pop(0)
                    print(label[round(sum(class_mov_avrg) / len(class_mov_avrg))])
                else: 
                    pass
            else:
                print("eye is closed")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

