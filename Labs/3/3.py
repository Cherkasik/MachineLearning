import numpy as np
import pandas as pd
import sklearn.svm as svm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

params = {
    'cvParts': 5,
    'regularizationStrength': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [1, 2, 3, 4, 5],
    'gamma': ['scale', 'auto']
}

bestParams = {
    'chips': {
        'linear': {
            'cvParts': 5,
            'regularizationStrength': 10.0,
            'kernel': 'linear',
            'degree': 1,
            'gamma': 'auto',
            'fScore': 0.4323478260869565
        },
        'poly': {
            'cvParts': 5,
            'regularizationStrength': 0.1,
            'kernel': 'poly',
            'degree': 2,
            'gamma': 'scale',
            'fScore': 0.7741252999873689
        },
        'rbf': {
            'cvParts': 5,
            'regularizationStrength': 10.0,
            'kernel': 'rbf',
            'degree': 1,
            'gamma': 'scale',
            'fScore': 0.8436223036223037
        },
        'sigmoid': {
            'cvParts': 5,
            'regularizationStrength': 1000.0,
            'kernel': 'sigmoid',
            'degree': 1,
            'gamma': 'scale',
            'fScore': 0.5583877995642701
        }
    },
    'geyser': {
        'linear': {
            'cvParts': 5,
            'regularizationStrength': 1.0,
            'kernel': 'linear',
            'degree': 1,
            'gamma': 'auto',
            'fScore': 0.8698298757122288
        },
        'poly': {
            'cvParts': 5,
            'regularizationStrength': 1000.0,
            'kernel': 'poly',
            'degree': 3,
            'gamma': 'auto',
            'fscore': 0.8715151515151515
        },
        'rbf': {
            'cvParts': 5,
            'regularizationStrength': 100.0,
            'kernel': 'rbf',
            'degree': 1,
            'gamma': 'scale',
            'fScore': 0.8648113839529138
        },
        'sigmoid': {
            'cvParts': 5,
            'regularizationStrength': 100.0,
            'kernel': 'sigmoid',
            'degree': 1,
            'gamma': 'scale',
            'fScore': 0.4796578454958567
        }
    }
}

def GetFeaturesLabels(data):
    features = data[:, :-1]
    labels = data[:, -1]
    return features, labels

def CVSplit(data, cvPartsAmount, testPartsAmount):
    parts = np.array_split(data, cvPartsAmount)
    test = parts.pop(testPartsAmount)
    train = np.concatenate(parts)
    return train, test

def SVC(data, params):
    features, labels = GetFeaturesLabels(data)
    np.random.shuffle(data)
    model = svm.SVC(C=params['regularizationStrength'], kernel=params['kernel'], degree=params['degree'], gamma=params['gamma'])
    fScores = []
    for testPartsAmount in range(params['cvParts']):
        train, test = CVSplit(data, params['cvParts'], testPartsAmount)
        trainFeatures, trainLabels = GetFeaturesLabels(train)
        testFeatures, testLabels = GetFeaturesLabels(test)
        model.fit(trainFeatures, trainLabels)
        predictedLabels = model.predict(testFeatures)
        fScores.append(f1_score(testLabels, predictedLabels, average='binary', pos_label='P'))
    return np.mean(fScores)

def FindBestParams(data):
    results = []
    for kernel in params['kernel']:
        for regularizationStrength in params['regularizationStrength']:
            if kernel == 'linear':
                parameters = {'cvParts': params['cvParts'], 'kernel': kernel, 'regularizationStrength': regularizationStrength, 'gamma': 'auto', 'degree': 1}
                results.append({'fscore': SVC(data, parameters), 'parameters': parameters})
            else:
                for gamma in params['gamma']:
                    if kernel == 'rbf' or kernel == 'sigmoid':
                        parameters = {'cvParts': params['cvParts'], 'kernel': kernel, 'regularizationStrength': regularizationStrength, 'gamma': gamma, 'degree': 1}
                        results.append({'fscore': SVC(data, parameters), 'parameters': parameters})
                    else:
                        for degree in params['degree']:
                            if kernel == 'poly':
                                parameters = {'cvParts': params['cvParts'], 'kernel': kernel, 'regularizationStrength': regularizationStrength, 'gamma': gamma, 'degree': degree}
                                results.append({'fscore': SVC(data, parameters), 'parameters': parameters})
    results = sorted(results, key=lambda k: k['fscore'], reverse=True)
    print(results[0])

def GetColor(labels):
    return ['green' if label == 'P' else 'red' for label in labels]

def Plot(data, filename):
    coords, types = GetFeaturesLabels(data)
    for kernel in params['kernel']:
        plt.scatter(coords[:, 0], coords[:, 1], c=GetColor(types), cmap=plt.cm.Paired)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        # multiply number of dots
        YY, XX = np.meshgrid(yy, xx)
        # get all possible pairs
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        model = svm.SVC(kernel=kernel, C=bestParams[filename][kernel]['regularizationStrength'], gamma=bestParams[filename][kernel]['gamma'], degree=bestParams[filename][kernel]['degree'])
        model.fit(coords, types)
        Z = model.decision_function(xy).reshape(XX.shape)
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        # mismatched dots highlight
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
        plt.ylabel(kernel)
        plt.show()

if __name__ == '__main__':
    filename = input()
    path = './data/' + filename + '.csv'
    data =  np.array(pd.read_csv(path))
    #FindBestParams(data)
    Plot(data, filename)