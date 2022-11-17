import matplotlib
matplotlib.use("Agg")

from Deployment.dualModelFiveClass import DualModel
from torch.optim import Adam, SGD
from torch import nn, flatten, Tensor
from sklearn.model_selection import train_test_split
import numpy as np 
import torch 
import os
import cv2
import copy
from trainHelperFunction import *

MODELNAME = "DualModelVx"

mata, muka = loadDualModel("DatasetWithNegative")

print(f"dataset mata sebesar {len(mata)}")
print(f"dataset muka sebesar {len(muka)}")

trainDualModel(mata, muka, 10, modelName = MODELNAME)



