from time import sleep
import tkinter as tk
from math import *
from imutils import face_utils
import dlib
import cv2
import os


p = "Model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
cap = cv2.VideoCapture(5)
_ = cap.read()

classes = [0, 1, 2, 3, 4]
label = ['kiri atas', 'kanan atas', 'kiri bawah', 'kanan bawah', 'unknown']

posX = [0, 960]
posY = [0, 540]

def getRect():
    _, image = cap.read()
    while image is None:
        image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    return (rects, gray, image)

def getEyes(rects, gray, image):
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # image = image[shape[38][1]-40:shape[42][1]+40 , shape[37][0]-60:shape[40][0]+60]
        image = image[shape[38][1]-10:shape[42][1]+10 , shape[37][0]-20:shape[40][0]+20] 
        image = cv2.resize(image, (100,50))
    
    return image

def getFace(rects, image):
    for (i,rect) in enumerate(rects):
        (x,y,w,h) = face_utils.rect_to_bb(rect=rect)
        image = image[y:y+h, x:x+w]
        image = cv2.resize(image, (100,100))
    return image


def makeNewWindow(x, y, text= "LOOK AT ME!!!!!", xlen = 960, ylen = 540, bg = "green"):
    newWindow = tk.Toplevel(window)
    newWindow.title("LOOK HERE")
    newWindow.geometry(f"{xlen}x{ylen}+{x}+{y}")
    newWindow.configure(bg= bg)
    text = tk.Label(newWindow, text=text, font="Arial 40 bold").place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    return newWindow


#main loop
nama = input("Masukkan nama subjek: ")
path = "./DatasetWithNegative"
while os.path.isdir(f"{path}/{nama}"):
   print("THIS USER ALREADY EXIST!!")
   nama = input("Masukkan nama subjek: ") 

os.mkdir(f"{path}/{nama}")
datasetPath = f"{path}/{nama}"
os.mkdir(f"{datasetPath}/eye")
os.mkdir(f"{datasetPath}/face")

print("harap lihat ke tulisan \"LOOK AT ME!!!\"")
input("press enter to continue....")

window = tk.Tk()
window.title(" ")
window.geometry("1920x1080+0+0")

for i in classes:
    tk.Tk.update_idletasks(window)
    tk.Tk.update(window)

    if not i == 4:
        newWindow = makeNewWindow(x = posX[i%2], y=posY[0 if i < 2 else 1])
    else:
        newWindow = makeNewWindow(x= 0, y= 0, text= "JANGAN LIAT LAYAR", xlen= 1920, ylen= 1080, bg="red")

    tk.Tk.update_idletasks(window)
    tk.Tk.update(window)
    sleep(2)
    
    for j in range(0, 200):
        tk.Tk.update_idletasks(window)
        tk.Tk.update(window)
        (rects, gray, image) = getRect()
        eyeImage = getEyes(rects, gray, image)
        faceImage = getFace(rects, image)
        cv2.imwrite(f"{datasetPath}/eye/{i},{j}.png", eyeImage)
        cv2.imwrite(f"{datasetPath}/face/{i},{j}.png", faceImage)

    newWindow.destroy()
window.destroy()