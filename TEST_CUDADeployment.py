import cv2 
import torch 
from time import time
import Deployment.usefulFunctions as func

label = ['kiri atas', 'kanan atas', 'kiri bawah', 'kanan bawah', "unkown"]

inferenceTime = []

thresh = func.calibration()
mov_avrg = []
class_mov_avrg = []

with torch.no_grad():
    while True:
        timer = time()
        (rects, gray, image) = func.getRect()
        if len(rects) > 0:
            mov_avrg.append(func.isBlinking(func.getEyes(rects, gray, image, isShape=True), thresh))
            if len(mov_avrg) > 5:
                mov_avrg.pop(0)
            if sum(mov_avrg)/len(mov_avrg) < 0.2 :
                mata = func.getEyes(rects, gray, image)
                muka = func.getFace(rects, image)
                if torch.is_tensor(mata) and torch.is_tensor(muka):
                    output = func.model(mata, muka)
                    prediction = torch.max(output, 1)
                    class_mov_avrg.append(int(prediction.indices))
                    if len(class_mov_avrg) > 10:
                        class_mov_avrg.pop(0)
                    print(label[round(sum(class_mov_avrg) / len(class_mov_avrg))])
                else: 
                    pass
            else:
                print("eye is closed")
            inferenceTime.append(time() - timer)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"Average inference time is {sum(inferenceTime)/len(inferenceTime)}")
                break

