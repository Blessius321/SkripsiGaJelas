import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import train_test_split
from Training.trainHelperFunction import *

def main():

    mata, muka = loadDualModel("DatasetWithNegative")
    print(f"dataset mata sebesar {len(mata)}")
    print(f"dataset muka sebesar {len(muka)}")
    mataTrain, mataTest = train_test_split(mata, test_size=0.25, random_state=10)
    mukaTrain, mukaTest = train_test_split(muka, test_size=0.25, random_state=10)

    print(f"training set mata: {len(mataTrain)}")
    print(f"training set muka: {len(mukaTrain)}")
    print(f"test set mata: {len(mataTest)}")
    print(f"test set muka: {len(mukaTest)}")
    # test_trainDualModel(mataTrain, mukaTrain, mataTest, mukaTest, 30, modelName = "Testing_Model.pth")
    confusionMatrix(mataTest, mukaTest, "Testing_Model.pth")

        

if __name__ == '__main__':
    main()