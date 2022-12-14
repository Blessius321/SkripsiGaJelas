from Deployment.usefulFunctions import *
import time

thresh = calibration()
print(f"Calibrated value is: {thresh}")
time.sleep(5)
mov_avrg = []
class_mov_avrg = []

while True:
    (rects, gray, image) = getRect()
    if len(rects) > 0:
        mov_avrg.append(isBlinking(getEyes(rects, gray, image, isShape=True), thresh))
        if len(mov_avrg) > 5:
            mov_avrg.pop(0)
        if sum(mov_avrg)/len(mov_avrg) < 0.2 :
            print("Mata Terbuka")
            print(sum(mov_avrg)/len(mov_avrg))
        else:
            print("Mata tertutup")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

