import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import math

INT_MAX = 2 * (10 ** 9)

def ReadData(path):
    data = pd.read_csv(path)
    data['classNumeric'] = data['class'].apply(lambda value: {'P': 1, 'N': -1}[value])
    features = data[['x', 'y']].to_numpy()
    values = data['classNumeric'].to_numpy()
    return features, values

def SplitIndicesData(n, batchesNumber=5):
    ids = np.arange(n)
    np.random.shuffle(ids)
    return np.array_split(ids, batchesNumber)

def GetTrainDataset(features, values, idsBatches, testNum):
    trainIds = np.array([], dtype=np.int64)
    for i in range(len(idsBatches)):
        if i != testNum:
            trainIds = np.concatenate((trainIds, idsBatches[i]), axis=0)
    return features[trainIds], values[trainIds]

class Comparison:
    def __init__(self, featureIdx, value):
        self.featureIdx = featureIdx
        self.value = value

class Tree:
    def __init__(self, significance, leftClass, rightClass, comparison):
        self.significance = significance
        self.comparison = comparison
        self.leftClass = leftClass
        self.rightClass = rightClass

    def Classify(self, row):
        classIdentified = self.leftClass if self.Accept(row) else self.rightClass
        return classIdentified, self.significance

    def Accept(self, row):
        return row[self.comparison.featureIdx] < self.comparison.value

class AdaBoost:
    def __init__(self, modelsNumber):
        self.modelsNumber = modelsNumber
        self.classQuantity = 2
        self.forest = []

    def Fit(self, X, Y):
        self.features = X
        self.values = Y
        self.weights = self.InitWeights()
        self.indices = np.arange(len(self.values))
        for _ in range(self.modelsNumber):
            self.NextModel()

    def NextModel(self):
        comp = self.GetComparison(self.indices)
        left, right = self.SplitOnClasses(self.indices, comp)
        significance, indicesWithError = self.CalcSignificance(self.weights, self.indices, comp, left, right)
        self.forest.append(Tree(significance, left, right, comp))
        self.weights = self.UpdateWeights(self.weights, significance, len(self.indices), indicesWithError)
        self.indices = self.ChooceNewIndices(self.weights, len(self.indices))

    def Predict(self, X):
        return np.fromiter(map(lambda x: self.Classify(x), X), int)

    def Classify(self, x):
        classesScore = np.zeros(2)
        for tree in self.forest:
            classValue, score = tree.Classify(x)
            classesScore[1 if classValue == 1 else 0] += score
        return 1 if np.argmax(classesScore) == 1 else -1

    def ChooceNewIndices(self, weights, n):
        indices = np.zeros(n, dtype=int)
        for i in range(n):
            randomValue = np.random.uniform(low=0.0, high=1.0)
            for idx in range(n):
                randomValue -= weights[idx]
                if randomValue <= 0:
                    indices[i] = idx
                    break
        return indices

    def UpdateWeights(self, weights, significance, n, indicesWithError):
        for i in range(n):
            v = significance if i in indicesWithError else -significance
            weights[i] = weights[i] * math.exp(v)
        x = sum(weights)
        for i in range(n):
            weights[i] = weights[i] / x
        return weights

    def InitWeights(self):
        valuesNumber = len(self.values)
        a = np.empty(valuesNumber)
        a.fill(1 / valuesNumber)
        return a

    def CalcSignificance(self, weights, indices, comp, left, right):
        totalError = 0
        indicesWithError = set()
        for idx, i in enumerate(indices):
            classValue = self.values[i]
            actual = left if self.Accept(self.features[i], comp.featureIdx, comp.value) else right
            if actual != classValue:
                totalError += weights[idx]
                indicesWithError.add(idx)
        if totalError == 0:
            return np.inf, indicesWithError
        return math.log((1 - totalError) / totalError) / 2, indicesWithError

    def GetComparison(self, indices):
        minGini = INT_MAX
        minGiniValue = 0
        minGiniFeature = 0
        for featureIdx in range(len(self.features[0])):
            ids = list(map(lambda i: (i, self.features[i][featureIdx]), indices))
            ids = sorted(ids, key=lambda pair: pair[1])
            ids = np.fromiter(map(lambda pair: pair[0], ids), int)
            leftCounters = {}
            rightCounters = {}
            for i in ids:
                currClass = self.values[i] - 1
                if not currClass in rightCounters:
                    leftCounters[currClass] = 0
                    rightCounters[currClass] = 0
                rightCounters[currClass] += 1
            idx = 0
            while idx < len(ids):
                currValue = self.CalcValue(idx, featureIdx, ids)
                currIdx = ids[idx]
                while True:
                    currClass = self.values[currIdx] - 1
                    rightCounters[currClass] -= 1
                    leftCounters[currClass] += 1
                    idx += 1
                    if idx >= len(ids):
                        break
                    currIdx = ids[idx]
                    if self.features[currIdx][featureIdx] >= currValue:
                        break
                currGini = self.Gini(idx, leftCounters, len(ids) - idx, rightCounters)
                if minGini > currGini:
                    minGini = currGini
                    minGiniValue = currValue
                    minGiniFeature = featureIdx
        return Comparison(minGiniFeature, minGiniValue)

    def Entropy(self, group, groupQuantity):
        if groupQuantity == 0:
            return 0
        entropy = 0
        for currClass in range(self.classQuantity):
            if currClass in group:
                p = group[currClass] / groupQuantity
                if (p != 0):
                    entropy += p * math.log(p)
        return -entropy

    def Gini(self, leftQuantity, left, rightQuantity, right):
        entropyLeft = self.Entropy(left, leftQuantity)
        entropyRight = self.Entropy(right, rightQuantity)
        n = leftQuantity + rightQuantity
        return leftQuantity * entropyLeft / n + rightQuantity * entropyRight / n

    def SplitOnClasses(self, indices, comp):
        left, right = np.zeros(2), np.zeros(2)
        for i in indices:
            classValue = 1 if self.values[i] == 1 else 0
            if self.Accept(self.features[i], comp.featureIdx, comp.value):
                left[classValue] += 1
            else:
                right[classValue] += 1
        def IsClass(yy):
            return 1 if np.argmax(yy) == 1 else -1

        return IsClass(left), IsClass(right)

    def Accept(self, row, feature, value):
        return row[feature] < value

    def CalcValue(self, i, feature_index, ids):
        return INT_MAX if i + 1 == len(ids) else (self.features[ids[i]][feature_index] + self.features[ids[i + 1]][feature_index]) / 2

def CrossValidation(X, Y, idsBatches, boost):
    yClassified = []
    yActual = []
    for testNum in range(len(idsBatches)):
        xTrain, yTrain = GetTrainDataset(X, Y, idsBatches, testNum)
        boost.Fit(xTrain, yTrain)
        for i in idsBatches[testNum]:
            yPrediction = boost.Classify(X[i])
            yClassified.append(yPrediction)
            yActual.append(Y[i])
    return accuracy_score(yActual, yClassified)

def DrawModelsAmountPlot(features, values):
    idsBatches = SplitIndicesData(len(values))
    points = []
    for modelsAmount in range(5, 30):
        boost = AdaBoost(modelsAmount)
        points.append({'models': modelsAmount, 'score': CrossValidation(features, values, idsBatches, boost)})
    plt.plot([point['models'] for point in points], [point['score'] for point in points])
    plt.xlabel('models amount')
    plt.ylabel('accuracy score')
    plt.legend()
    plt.show()

def DrawCurrModelPlot(X, Y, boost, title):
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = boost.Predict(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    filename = input()
    path = './data/' + filename + '.csv'
    features, values = ReadData(path)
    DrawModelsAmountPlot(features, values)
    initialModels = 10
    boost = AdaBoost(initialModels)
    boost.Fit(features, values)
    for i in range(16):
        DrawCurrModelPlot(features, values, boost, filename + " with " + str(initialModels + i) + " models")
        boost.NextModel()
