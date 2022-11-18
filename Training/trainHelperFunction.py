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
import datetime

INIT_LR = 0.001
BATCH_SIZE = 64
EPOCHS = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on:{device}")

def loadMata(path):
  folderList = os.listdir(path)
  totalHolder = []

  for namaFolder in folderList:
      datasetPath = f"{path}/{namaFolder}/eye"
      nameList = os.listdir(datasetPath)
      for name in nameList:
          im = cv2.imread(f"{datasetPath}/{name}")
          im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
          top = max([max(x) for x in im])
          imclass = name.split(",")[0].strip()
          # totalHolder.append(((torch.tensor([[im]]).to(dtype=torch.float, device=device))/top,
          #                     torch.tensor([[int(imClass) / dims[want]]]).to(dtype=torch.float, device=device)))
          # totalHolder.append(((torch.tensor([[im]]).to(dtype=torch.float,device=device))/top,
                          # torch.tensor([[int(imclass)]]).to(dtype=torch.float,device=device)))
          totalHolder.append(((torch.tensor([[np.array(im)]]).to(dtype=torch.float,device=device))/top, torch.tensor([int(imclass)]).to(device=device)))

  return totalHolder

def loadDualModel(path):
  folderList = os.listdir(path)
  eyeHolder = []
  faceHolder = []

  for namaFolder in folderList:
      eyePath = f"{path}/{namaFolder}/eye"
      facePath = f"{path}/{namaFolder}/face"
      nameList = os.listdir(eyePath)
      for name in nameList:
          eyeIm = cv2.imread(f"{eyePath}/{name}")
          faceIm = cv2.imread(f"{facePath}/{name}")
          eyeIm = cv2.cvtColor(eyeIm, cv2.COLOR_BGR2GRAY)
          faceIm = cv2.cvtColor(faceIm, cv2.COLOR_BGR2GRAY)
          eyeTop = max([max(x) for x in eyeIm])
          faceTop = max([max(x) for x in faceIm])
          imclass = name.split(",")[0].strip()
          # totalHolder.append(((torch.tensor([[im]]).to(dtype=torch.float, device=device))/top,
          #                     torch.tensor([[int(imClass) / dims[want]]]).to(dtype=torch.float, device=device)))
          # totalHolder.append(((torch.tensor([[im]]).to(dtype=torch.float,device=device))/top,
                          # torch.tensor([[int(imclass)]]).to(dtype=torch.float,device=device)))
          eyeHolder.append(((torch.tensor([[np.array(eyeIm)]]).to(dtype=torch.float,device=device))/eyeTop, torch.tensor([int(imclass)]).to(device=device)))
          faceHolder.append(((torch.tensor([[np.array(faceIm)]]).to(dtype=torch.float,device=device))/faceTop, torch.tensor([int(imclass)]).to(device=device)))

  return eyeHolder, faceHolder

def loadMuka(path):
  folderList = os.listdir(path)
  totalHolder = []

  for namaFolder in folderList:
      datasetPath = f"{path}/{namaFolder}/face"
      nameList = os.listdir(datasetPath)
      for name in nameList:
          im = cv2.imread(f"{datasetPath}/{name}")
          im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
          top = max([max(x) for x in im])
          imclass = name.split(",")[0].strip()
          # totalHolder.append(((torch.tensor([[im]]).to(dtype=torch.float, device=device))/top,
          #                     torch.tensor([[int(imClass) / dims[want]]]).to(dtype=torch.float, device=device)))
          # totalHolder.append(((torch.tensor([[im]]).to(dtype=torch.float,device=device))/top,
                          # torch.tensor([[int(imclass)]]).to(dtype=torch.float,device=device)))
          totalHolder.append(((torch.tensor([[np.array(im)]]).to(dtype=torch.float,device=device))/top, torch.tensor([int(imclass)]).to(device=device)))

  return totalHolder


def trainDualModel(mata, muka, epoch = EPOCHS, modelPath = None, modelName = None, flag = None):
    model = DualModel().to(device=device)
    if not(modelPath is None):
      print("model is pretrained")
      model.load_state_dict(torch.load(modelPath))
    else:
      print("this is a blank model")
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=INIT_LR, momentum=0.5)
    brokenTensor = 0
    
    for epoch in range(epoch): 
        print(f"Reported Broken Tensor = {brokenTensor}")
        brokenTensor = 0
        print(f"epoch {epoch} starting")
        running_loss = 0
        for i, (mataIm, label) in enumerate(mata):
            (mukaIm, mukaLabel) = muka[i]
            if mata[i][0][0].shape == torch.Size([1, 50, 100]) and muka[700][0][0].shape == torch.Size([1, 100, 100]):
              if not mukaLabel == label:
                print("SALAH CUY")
                return
              optimizer.zero_grad()
              output = model(mataIm, mukaIm)
              # print(label)
              # print(output)
              loss = criterion(output, label)
              loss.backward()
              optimizer.step()

              running_loss += loss.item()
              if i%300 == 299:
                  print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/ 300:.3f}') 
                  running_loss = 0.0
            else:
              brokenTensor+= 1
              # print("tensor doesn't match")
              pass

    if modelPath is None:
      if modelName is None:
        torch.save(model.state_dict(), f"Model/TrainedModelOn{datetime.datetime.now()}{type(model).__name__}.pth")
      else:
        torch.save(model.state_dict(), f"Model/{modelName}.pth")
    else:
      torch.save(model.state_dict(), modelPath) 
    print("Model is Saved!!")          

    return

def testing(mataTest, mukaTest, MODELNAME):
  if os.path.exists(f"Model/{MODELNAME}"):
    print("Model Doesn't exist")
    return None
  model = DualModel().to(device=device)
  model.load_state_dict(f"Model/{MODELNAME}")
  
  correct = 0
  total = 0
  with torch.no_grad():
    for i, (mataIm, mataLabel) in enumerate(mataTest):
      (mukaIm, mukaLabel) = mukaTest[i]
      output = model(mataIm, mukaIm)
      _, predicted = torch.max(output.data, 1)
      total += mataLabel.size(0)
      correct += (predicted == mataLabel).sum().item()
  print(f"accuracy of the model on {len(mataTest)} images: {100 * correct /total}%")
