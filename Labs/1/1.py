import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import f1_score
import math
import matplotlib.pyplot as plt

distanceFunctions = ['manhattan', 'euclidean', 'chebyshev']
kernelFunctions = ['uniform', 'triangular', 'epanechnikov', 'quartic', 'triweight', 'tricube', 'gaussian', 'cosine', 'logistic', 'sigmoid']
windowTypes = ['fixed', 'variable']
windowWidths = [0.01, 0.05, 0.1, 0.2]
windowNeighbours = [1, 2, 3, 4, 5, 10, 20, 30, 40,  50, 100, 104]

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
    data = data.replace({ 'Class': classCount })
    npData = np.array(data)
    normalized = preprocessing.normalize(npData[:, :-1], axis = 0)
    labels = npData[:, -1]
    return npData, normalized, labels, numberOfClasses

def Kernel(r, kernelFunctionType):
    '''
    kernel function

    Accepts:
    r - number
    kernelFunctionType - string name of kernel function type

    Returns:
    calculated kernel
    '''
    if kernelFunctionType == 'uniform':
        return 1 / 2 - math.floor(abs(r)) if r < 1 else 0

    if kernelFunctionType == 'triangular':
        return (1 - abs(r)) if r < 1 else 0

    if kernelFunctionType == 'epanechnikov':
        return (3 / 4 * (1 - r ** 2)) if r < 1 else 0

    if kernelFunctionType == 'quartic':
        return (15 / 16 * (1 - r ** 2) ** 2) if r < 1 else 0

    if kernelFunctionType == 'triweight':
        return (35 / 32 * (1 - r ** 2) ** 3) if r < 1 else 0

    if kernelFunctionType == 'tricube':
        return (70 / 81 * (1 - abs(r ** 3)) ** 3) if r < 1 else 0

    if kernelFunctionType == 'gaussian':
        return 1 / (2 * math.pi) ** (1 / 2) * math.e ** (-(1 / 2) * r ** 2)

    if kernelFunctionType == 'cosine':
        return (math.pi / 4 * math.cos(math.pi / 2 * r)) if r < 1 else 0

    if kernelFunctionType == 'logistic':
        return 1 / (math.e ** r + 2 + math.e ** (-r))

    if kernelFunctionType == 'sigmoid':
        return 2 / math.pi * 1 / (math.e ** r + math.e ** (-r))

def Distance(x, y, distanceFunctionType):
    '''
    distance function

    Accepts:
    x, y - array of X and Y values
    distanceFunctionType - string name of distance function type

    Returns:
    calculated distance
    '''
    if distanceFunctionType == 'manhattan':
        return sum([abs(xi - yi) for xi, yi in zip(x, y)])

    if distanceFunctionType == 'euclidean':
        return sum([(xi - yi) ** 2 for xi, yi in zip(x, y)]) ** (1 / 2)

    if distanceFunctionType == 'chebyshev':
        return max([abs(xi - yi) for xi, yi in zip(x, y)])

def ClassifyObject(features, labels, testFeature, distanceFunctionType, kernelFunctionType, windowType, windowParameter):
    '''
    classification of objects

    Accepts:
    features, labesls - array of features and labels
    distanceFunctionType - string name of distance function type
    kernelFunctionType - string name of kernel function type
    windowType - string name of window type
    windowParameter - number that set parameter of window type function

    Returns:
    calculated distance
    '''
    distancesAndLabels = []
    featuresLen = len(features);
    for index in range(featuresLen):
        distancesAndLabels.append({'distance': Distance(testFeature, features[index], distanceFunctionType), 'label': labels[index]})
    distancesAndLabels = sorted(distancesAndLabels, key=lambda k: k['distance'])
    if windowType == 'variable':
        # get radius as distance to vector number windowParameter (as they are sorted by distance) and if there are block of save vectors there do a little step out of them
        windowRadius = distancesAndLabels[windowParameter]['distance'] \
            if distancesAndLabels[windowParameter-1]['distance'] < distancesAndLabels[windowParameter]['distance'] \
            else distancesAndLabels[windowParameter-1]['distance'] + 0.000001
    else:
        windowRadius = windowParameter

    weightedClassSum = 0
    kernelsSum = 0

    for index in range(len(features)):
        kernelValue = Kernel(
            distancesAndLabels[index]['distance']/windowRadius if windowRadius != 0 else 0,
            kernelFunctionType
        )
        weightedClassSum += distancesAndLabels[index]['label'] * kernelValue
        kernelsSum += kernelValue

    predictedValue = weightedClassSum / kernelsSum if kernelsSum != 0 else weightedClassSum

    return predictedValue

def FindBestParametersNative(features, labels, classesNumber):
    '''
    finds best parameters for naive regression

    Accepts:
    features, labels - arrays of features and labels
    classesNumber - number of classes in dataset

    Returns:
    best combination of distance function, kernel function, window type and window parameter
    '''
    results = []
    for distanceFunction in distanceFunctions:
        for kernelFunction in kernelFunctions:
            for windowType in windowTypes:
                for windowParameter in windowWidths if windowType == 'fixed' else windowNeighbours:
                    predictedLabels = []
                    featuresLen = len(features)
                    for index in range(featuresLen):
                        predictedValue = ClassifyObject(
                            features[np.arange(featuresLen) != index],
                            labels[np.arange(len(labels)) != index],
                            features[index],
                            distanceFunction,
                            kernelFunction,
                            windowType,
                            windowParameter
                        )
                        predictedLabel = round(predictedValue)
                        predictedLabels.append(predictedLabel)

                    fScore = f1_score(labels, predictedLabels, labels=[i for i in range(classesNumber)], average='weighted')
                    results.append(
                        {
                            'fScore': fScore,
                            'distanceFunction': distanceFunction,
                            'kernelFunction': kernelFunction,
                            'windowType': windowType,
                            'windowParameter': windowParameter},
                    )
    sortedResults = sorted(results, key=lambda configuration: configuration['fScore'], reverse=True)
    return sortedResults[0]

def FindBestParametersOneHot(features, labels, classesNumber):
    '''
    finds best parameters for one hot regression

    Accepts:
    features, labels - arrays of features and labels
    classesNumber - number of classes in dataset

    Returns:
    best combination of distance function, kernel function, window type and window parameter
    '''
    results = []
    encodedLabels = np.array(pd.get_dummies(pd.Series(labels)))
    for distanceFunction in distanceFunctions:
        for kernelFunction in kernelFunctions:
            for windowType in windowTypes:
                for windowParameter in windowWidths if windowType == 'fixed' else windowNeighbours:
                    predictedLabels = []
                    featuresLen = len(features)
                    for i in range(featuresLen):
                        predictedLabelValues = []
                        for j in range(len(encodedLabels[i])):
                            predictedLabelValue = ClassifyObject(
                                features[np.arange(featuresLen) != i],
                                encodedLabels[np.arange(len(encodedLabels)) != i][:, j],
                                features[i],
                                distanceFunction,
                                kernelFunction,
                                windowType,
                                windowParameter
                            )
                            predictedLabelValues.append(predictedLabelValue)
                        predictedValue = [k for k, l in enumerate(predictedLabelValues) if l == max(predictedLabelValues)][0]
                        predictedLabel = round(predictedValue)
                        predictedLabels.append(predictedLabel)
                    fScore = f1_score(labels, predictedLabels, labels=[i for i in range(classesNumber)], average='weighted')
                    results.append(
                        {
                            'fScore': fScore,
                            'distanceFunction': distanceFunction,
                            'kernelFunction': kernelFunction,
                            'windowType': windowType,
                            'windowParameter': windowParameter,
                        })
    sortedResults = sorted(results, key=lambda configuration: configuration['fScore'], reverse=True)
    return sortedResults[0]

def NaiveRegression():
    print('Running Naive regression...')
    # bestParameters = FindBestParametersNative(features, labels, classesNumber)
    bestParameters = {'distanceFunction': 'manhattan', 'kernelFunction': 'triweight', 'windowType': 'variable', 'windowParameter': 1}
    # distanceFunction - manhattan, kernelFunction - any (uniform), windowType - variable, windowParameter - 1
    fScores = []
    featuresLen = len(features)
    windowParameters = range(featuresLen - 1)
    for windowParameter in windowParameters:
        predictedLabels = []
        for i in range(featuresLen):
            predictedValue = ClassifyObject(
                features[np.arange(featuresLen) != i],
                labels[np.arange(len(labels)) != i],
                features[i],
                bestParameters['distanceFunction'],
                bestParameters['kernelFunction'],
                bestParameters['windowType'],
                windowParameter
            )
            predictedLabel = round(predictedValue)
            predictedLabels.append(predictedLabel)
        fScore = f1_score(labels, predictedLabels, labels=[i for i in range(classesNumber)], average='weighted')
        fScores.append({'fScore': fScore, 'windowParameter': windowParameter})
    plt.plot([point['windowParameter'] for point in fScores], [point['fScore'] for point in fScores])
    plt.xlabel('nearest neighbour')
    plt.ylabel('f Score')
    plt.show()

def OneHotRegression():
    print('Running regression with One Hot conversion')
    #bestParameters = FindBestParametersOneHot(features, labels, classesNumber)
    bestParameters = {'fScore': 0.46567037834213315, 'distanceFunction': 'manhattan', 'kernelFunction': 'uniform', 'windowType': 'variable', 'windowParameter': 20}
    # distanceFunction - manhattan, kernelFunction - triweight, windowType - variable, windowParameter - 20
    fScores = []
    encodedLabels = np.array(pd.get_dummies(pd.Series(labels)))
    featuresLen = len(features)
    windowParameters = range(featuresLen - 1)
    for windowParameter in windowParameters:
        predictedLabels = []
        for i in range(featuresLen):
            predictedLabelValues = []
            for j in range(len(encodedLabels[i])):
                predictedLabelValue = ClassifyObject(
                    features[np.arange(len(features)) != i],
                    encodedLabels[np.arange(len(encodedLabels)) != i][:, j],
                    features[i],
                    bestParameters['distanceFunction'],
                    bestParameters['kernelFunction'],
                    bestParameters['windowType'],
                    windowParameter
                )
                predictedLabelValues.append(predictedLabelValue)
            predictedValue = [i for i, j in enumerate(predictedLabelValues) if j == max(predictedLabelValues)][0]
            predictedLabel = round(predictedValue)
            predictedLabels.append(predictedLabel)
        fScore = f1_score(labels, predictedLabels, labels=[i for i in range(classesNumber)], average='weighted')
        fScores.append({'fScore': fScore, 'windowParameter': windowParameter})
    plt.plot([point['windowParameter'] for point in fScores], [point['fScore'] for point in fScores])
    plt.xlabel('nearest neighbour')
    plt.ylabel('f score')
    plt.show()

if __name__ == '__main__':
    dataset, features, labels, classesNumber = LoadFromFile()
    regression = input()
    if regression == 'naive':
        NaiveRegression()
    elif regression == 'one hot':
        OneHotRegression()

