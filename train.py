import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import train_test_split
from Training.trainHelperFunction import *
import argparse

def main(arg):

    mata, muka = loadDualModel("DatasetWithNegative")
    print(f"dataset mata sebesar {len(mata)}")
    print(f"dataset muka sebesar {len(muka)}")
    mataTrain, mataTest = train_test_split(mata, test_size=0.25, random_state=10)
    mukaTrain, mukaTest = train_test_split(muka, test_size=0.25, random_state=10)

    if arg.mode == "train":
        print(f"training set mata: {len(mataTrain)}")
        print(f"training set muka: {len(mukaTrain)}")
        trainDualModel(mataTrain, mukaTrain, 10, modelName = arg.modelName)

    elif arg.mode == "test":
        print(f"test set mata: {len(mataTest)}")
        print(f"test set muka: {len(mukaTest)}")
        testing(mataTest, mukaTest, arg.modelName)

    elif arg.mode == "train test":
        print(f"training set mata: {len(mataTrain)}")
        print(f"training set muka: {len(mukaTrain)}")
        print(f"test set mata: {len(mataTest)}")
        print(f"test set muka: {len(mukaTest)}")
        trainDualModel(mataTrain, mukaTrain, 6, modelName = arg.modelName)
        testing(mataTest, mukaTest, arg.modelName)

if __name__ == "__main__":
    parser =argparse.ArgumentParser(description="train and test model using dataset in the DatasetFolder")
    parser.add_argument(
        '--mode',
        type= str,
        default= "train test",
        help='''\
            Mode eksekusi, pilihan: train, test, train test''')
    parser.add_argument(
        '--modelName',
        type= str,
        default= "DualModelV1.2.pth",
        help='''\
            Nama model pada folder 'Model' yang akan diload''')
    arg = parser.parse_args()
    validMode = ["train", 'test', 'train test']
    if not arg.mode in validMode:
        print("mode is not valid, use 'train', 'test', or 'train test'")
    else:
        if os.path.isfile(f"Model/{arg.modelName}"):
            print("model is pretrained model")
        else:
            print("model is a blank model")
        main(arg)

