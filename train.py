import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import train_test_split
from Training.trainHelperFunction import *

MODELNAME = "DualModelV1.2"

mata, muka = loadDualModel("DatasetWithNegative")

print(f"dataset mata sebesar {len(mata)}")
print(f"dataset muka sebesar {len(muka)}")

mataTrain, mataTest = train_test_split(mata, test_size=0.25, random_state=10)
mukaTrain, mukaTest = train_test_split(muka, test_size=0.25, random_state=10)

# trainDualModel(mataTrain, mukaTrain, 10, modelName = MODELNAME)

testing(mataTest, mukaTest, MODELNAME)



