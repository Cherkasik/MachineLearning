import numpy as np
import pandas as pd
from sklearn import preprocessing

def LoadFromFile():
    '''
    Loads dataset from csv file into numpy array

    Returns:
    npData - numpy array with dataset
    normalized - normalized numpy array with dataset
    labels - labels of dataset
    numberOfClasses - number of classes in dataset
    '''
    data = pd.read_csv('./dataset.csv')
    classCount = { name: index for index, name in enumerate(sorted(list(set(data.Class)))) }
    numberOfClasses = len(set(data.Class))
    print(data)
    data = data.replace({ 'Class': classCount })
    print(data)
    npData = np.array(data)
    normalized = preprocessing.normalize(npData[:, :-1], axis = 0)
    labels = npData[:, -1]
    return npData, normalized, labels, numberOfClasses

if __name__ == '__main__':
    LoadFromFile()
