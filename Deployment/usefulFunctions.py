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

# Buka kamera
cap = cv2.VideoCapture(0)

# Shape predictor for face landmark
p = "/Model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# CUDA detection
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# ganti angka buat ganti kamera
model = DualModel().to(device=device)

# MODEL PATH
model.load_state_dict(torch.load("Model/fourClassModelDualModel.pth", map_location=device))
model.eval()


# Get bounding box 
def getRect():
    _, image = cap.read()
    while image is None:
        image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    return (rects, gray, image)

def getEyes(rects, gray, image, isShape = False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        if isShape:
            return shape[36: 42]
        image = image[shape[38][1]-10:shape[42][1]+10 , shape[37][0]-20:shape[40][0]+20] 
        # image = image[shape[38][1]-40:shape[42][1]+40 , shape[37][0]-60:shape[40][0]+60]
        # image = image[shape[1][1]-10:shape[5][1]+10 , shape[0][0]-20:shape[3][0]+20]
        # image = image[shape[1][1]-5:shape[5][1]+5 , shape[0][0]-10:shape[3][0]+10]  
        image = cv2.resize(image, (100,50))
        # image = np.array(image)
        cv2.imshow("mata", image)
        top = max([max(x) for x in image])
        image = (torch.from_numpy(np.array([[image]])).to(dtype=torch.float, device=device, non_blocking=True)) / top
    
    return image

def getFace(rects, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for (i,rect) in enumerate(rects):
        (x,y,w,h) = face_utils.rect_to_bb(rect=rect)
        image = image[y:y+h, x:x+w]
        image = cv2.resize(image, (100,100))
        # image = np.array(image)
        cv2.imshow("muka", image)

        top = max([max(x) for x in image])
        image = (torch.from_numpy(np.array([[image]])).to(dtype=torch.float, device=device, non_blocking=True)) / top
    return image

def isBlinking(eye, thresh):
    return True if eye_aspect_ratio(eye) < thresh - 0.3 else False

def eye_aspect_ratio(eye):
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear

def calibration():
    timer = time()
    threshs = []
    print("Calibrating")
    while time() - timer < 5:
        _, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        mata = getEyes(rects, gray, image, isShape=True)
        threshs.append(eye_aspect_ratio(mata))    
    avgthresh = sum(threshs) / len(threshs)
    print(f"the threshold is: {avgthresh}")
    return avgthresh