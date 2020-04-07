import math
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn import preprocessing
import numpy as np

gradientParameters = {
    'learningRate': [0.001, 0.01, 0.5, 1, 5],
    'regularizationStrength': [0.001, 0.01, 0.5, 1, 5, 10, 30],
    'numIterations': 10,
    'crossValParameter': [5, 10, 20, 50]
}

geneticParameters = {
    'sizePopulation': [20, 50],
    'parentProportion': [0.1, 0.5, 0.9],
    'generationsNumber': 10,
    'randomGeneAmplitude': [10, 100, 1000, 10000],
    'mutationAmplitude': [0.1, 0.5, 1, 5],
    'regularizationStrength': [0.00001, 0.001, 0.1]
}

def ReadData():
    file = open('./data/8.txt')
    featureCount = int(file.readline())
    trainCount = int(file.readline())
    trainFeatures, trainLabels = [], []
    for i in range(trainCount):
        temp = list(map(int, file.readline().split()))
        trainFeatures.append(temp[:-1])
        trainLabels.append(temp[-1])
    testCount = int(file.readline())
    testFeatures, testLabels = [], []
    for i in range(testCount):
        temp = list(map(int, file.readline().split()))
        testFeatures.append(temp[:-1])
        testLabels.append(temp[-1])
    return featureCount, testCount, trainCount, trainFeatures, trainLabels, testFeatures, testLabels

def NRMSE(predicted, trueValue):
    yLen = len(trueValue)
    sumSquares = sum([(iPredicted - iValue) ** 2 for iPredicted, iValue in zip(predicted, trueValue)])
    return (math.sqrt(sumSquares / yLen) / (max(trueValue) - min(trueValue)))

def LeastSquares(featureCount, trainCount, trainFeatures, trainLabels, testFeatures, testLabels):
    trainFeatures = preprocessing.normalize(trainFeatures, axis=0)
    testFeatures = preprocessing.normalize(testFeatures, axis=0)
    metrics = []
    for i in range(0, 1500, 1):
        regularization = i / 10
        model = Ridge(solver='lsqr', max_iter=100, alpha=regularization)
        model.fit(trainFeatures, trainLabels)
        training_NRMSE = NRMSE(model.predict(trainFeatures), trainLabels)
        test_NRMSE = NRMSE(model.predict(testFeatures), testLabels)
        metrics.append({'regularization': regularization, 'trainingNRMSE': training_NRMSE, 'testNRMSE': test_NRMSE})
    plt.plot([point['regularization'] for point in metrics], [point['trainingNRMSE'] for point in metrics])
    plt.xlabel('regularization')
    plt.ylabel('trainingNRMSE')
    plt.show()
    plt.plot([point['regularization'] for point in metrics], [point['testNRMSE'] for point in metrics])
    plt.xlabel('regularization')
    plt.ylabel('testNRMSE')
    plt.show()

def CrossValidationSplit(data, k):
    np.random.shuffle(data)
    features = data[:, :-1]
    labels = data[:, -1]
    split = len(labels) // k
    trainFeatures = np.array(features[split:])
    crossValFeatures = np.array(features[:split])
    trainLabels = np.array(labels[split:])
    crossValLabels = np.array(labels[:split])
    return trainFeatures, crossValFeatures, trainLabels, crossValLabels

def GradientPredict(features, weightsVector):
    return [sum([weight * feature_val for weight, feature_val in zip(weightsVector, features[i])]) for i in range(len(features))]

def GradientDescent(parameters, featureCount, trainCount, trainFeatures, trainLabels, testFeatures, testLabels):
    trainLosses, testLosses = [], []
    trainFeatures = preprocessing.normalize(trainFeatures, axis=0)
    testFeatures = preprocessing.normalize(testFeatures, axis=0)
    trainData = np.hstack((trainFeatures, np.reshape(trainLabels, (-1, 1))))
    weights = np.random.rand(featureCount + 1)
    crossValParameter = parameters['crossValParameter']
    for iteration in range(parameters['numIterations']):
        # print('Iteration {}'.format(iteration + 1))
        crossValTrainFeatures, crossValTestFeatures, crossValTrainLabels, crossValTestLabels = CrossValidationSplit(trainData, crossValParameter)
        crossValTrainFeatures = np.hstack((
            crossValTrainFeatures,
            np.ones((crossValTrainFeatures.shape[0], 1), dtype=crossValTrainFeatures.dtype)
        ))
        crossValTestFeatures = np.hstack((
            crossValTestFeatures,
            np.ones((crossValTestFeatures.shape[0], 1), dtype=crossValTestFeatures.dtype)
        ))
        predicted = GradientPredict(crossValTrainFeatures, weights)
        absoluteError = [predictedValue - crossValTrainLabel for predictedValue, crossValTrainLabel in zip(predicted, crossValTrainLabels)]
        gradient = crossValTrainFeatures.T.dot(absoluteError) / crossValTrainFeatures.shape[0] + parameters['regularizationStrength'] * weights
        weights = weights * (1 - parameters['learningRate'] * parameters['regularizationStrength']) + parameters['learningRate'] * (-gradient)
        predictedLabels = GradientPredict(crossValTestFeatures, weights)
        trainLoss = NRMSE(predictedLabels, crossValTestLabels)
        trainLosses.append(trainLoss)
        predictedLabels = GradientPredict(testFeatures, weights)
        testLoss = NRMSE(predictedLabels, testLabels)
        testLosses.append(testLoss)
        # print('Cross validation loss: {}, Test loss: {}'.format(trainLoss, testLoss))

    return trainLosses, testLosses

def CalculateBestGradientParameters(featureCount, testCount, trainCount, trainFeatures, trainLabels, testFeatures, testLabels):
    results = []
    for learningRate in gradientParameters['learningRate']:
        for regularizationStrength in gradientParameters['regularizationStrength']:
            for crossValParameter in gradientParameters['crossValParameter']:
                parametersCombination = {
                    'learningRate': learningRate,
                    'regularizationStrength': regularizationStrength,
                    'crossValParameter': crossValParameter,
                    'numIterations': gradientParameters['numIterations']
                }
                trainLoss, testLoss = GradientDescent(parametersCombination, featureCount, trainCount, trainFeatures, trainLabels, testFeatures, testLabels)
                results.append({'trainLoss': trainLoss, 'testLoss': testLoss, 'lastTestLoss': testLoss[-1], 'parameters': parametersCombination})
    results = sorted(results, key=lambda k: k['lastTestLoss'])
    print(results[0])
    plt.plot(results[0]['trainLoss'], label='trainLoss')
    plt.plot(results[0]['testLoss'], label='testLoss')
    plt.xlabel('iterations')
    plt.ylabel('loss(NRMSE)')
    plt.legend()
    plt.show()

def Reproduce(parents, childrenNumber, parentsNumber, chromosomesNumberPerIndividual):
    children = []
    for index in range(childrenNumber):
        parent1Index = index % parentsNumber
        parent2Index = (index + 1) % parentsNumber
        child = np.empty(chromosomesNumberPerIndividual)
        crossoverPoint = round(chromosomesNumberPerIndividual / 2)
        child[0:crossoverPoint] = parents[parent1Index, 0:crossoverPoint]
        child[crossoverPoint:] = parents[parent2Index, crossoverPoint:]
        children.append(child)
    return np.array(children)

def SelectParents(chromosomes, features, labels, parentsNumber, regularizationStrength):
    fitnessTemp = []
    for chromosome in chromosomes:
        predictedLabels = GeneticPredict(features, chromosome)
        fitnessTemp.append({
            'chromosome': chromosome,
            'fitness': -(NRMSE(predictedLabels, labels) + regularizationStrength * sum([abs(value) for value in chromosome]))
        })
    parents = sorted(fitnessTemp, key=lambda k: k['fitness'], reverse=True)[:parentsNumber]
    return np.array([parent['chromosome'] for parent in parents])

def GeneticPredict(features, chromosomes):
    predictedLabels = []
    for feature in features:
        predictedLabels.append(sum([parameter * chromosome for parameter, chromosome in zip(feature, chromosomes)]) + chromosomes[-1])
    return predictedLabels

def Mutate(children, mutationAmplitude):
    return np.array([[
        chromosome + random.uniform(-1, 1) * (chromosome * mutationAmplitude)
        for chromosome in child] for child in children])

def Genetic(parameters, trainCount, featureCount, trainFeatures, trainLabels, testFeatures, testLabels):
    trainLosses, testLosses = [], []
    populationSize = parameters['sizePopulation']
    parentsNumber = int(populationSize * parameters['parentProportion'])
    childrenNumber = populationSize - parentsNumber
    chromosomesNumberPerIndividual = featureCount + 1
    chromosomes = np.random.uniform(
        low=-parameters['randomGeneAmplitude'],
        high=parameters['randomGeneAmplitude'],
        size=(populationSize, chromosomesNumberPerIndividual)
    )
    for i in range(parameters['generationsNumber']):
        print(i + 1)
        parents = SelectParents(chromosomes, trainFeatures, trainLabels, parentsNumber, parameters['regularizationStrength'])
        children = Reproduce(parents, childrenNumber, parentsNumber, chromosomesNumberPerIndividual)
        children = Mutate(children, parameters['mutationAmplitude'])
        chromosomes = np.concatenate((parents, children), axis=0)
        trainLosses.append(NRMSE(GeneticPredict(trainFeatures, parents[0]), trainLabels))
        testLosses.append(NRMSE(GeneticPredict(testFeatures, parents[0]), testLabels))
    return trainLosses, testLosses

def CalculateBestGeneticParameters(trainCount, featureCount, trainFeatures, trainLabels, testFeatures, testLabels):
    results = []
    for mutationAmplitude in geneticParameters['mutationAmplitude']:
        for regularizationStrength in geneticParameters['regularizationStrength']:
            for sizePopulation in geneticParameters['sizePopulation']:
                for randomGeneAmplitude in geneticParameters['randomGeneAmplitude']:
                    for parentProportion in geneticParameters['parentProportion']:
                        parametersCombination = {
                            'mutationAmplitude': mutationAmplitude,
                            'regularizationStrength': regularizationStrength,
                            'sizePopulation': sizePopulation,
                            'generationsNumber': geneticParameters['generationsNumber'],
                            'parentProportion': parentProportion,
                            'randomGeneAmplitude': randomGeneAmplitude
                        }
                        trainLoss, testLoss = Genetic(parametersCombination, trainCount, featureCount, trainFeatures, trainLabels, testFeatures, testLabels)
                        results.append({'trainLoss': trainLoss, 'testLoss': testLoss, 'lastTestLoss': testLoss[-1], 'parameters': parametersCombination})
    results = sorted(results, key=lambda k: k['lastTestLoss'])
    print(results[0])
    plt.plot(results[0]['trainLoss'], label='trainLoss')
    plt.plot(results[0]['testLoss'], label='testLoss')
    plt.xlabel('iterations')
    plt.ylabel('NRMSE')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    featureCount, testCount, trainCount, trainFeatures, trainLabels, testFeatures, testLabels = ReadData()
    variant = input()
    if variant == 'square':
        # best one is testNRMSE = 0.07219181907370517, regularization = 28.6
        LeastSquares(featureCount, trainCount, trainFeatures, trainLabels, testFeatures, testLabels)
    elif variant == 'gradient':
        # CalculateBestGradientParameters(featureCount, testCount, trainCount, trainFeatures, trainLabels, testFeatures, testLabels)
        parameters = {
            'learningRate': 1,
            'regularizationStrength': 0.001,
            'numIterations': 50,
            'crossValParameter': 10
        }
        bestTestLosses = []
        bestTrainLosses = []
        for maxIterations in range(parameters['numIterations']):
            print("Iteration {} out of {}".format(maxIterations + 1, parameters['numIterations']))
            trainLoss, testLoss = GradientDescent({
                    'learningRate': parameters['learningRate'],
                    'regularizationStrength': parameters['regularizationStrength'],
                    'crossValParameter': parameters['crossValParameter'],
                    'numIterations': maxIterations + 1
            }, featureCount, trainCount, trainFeatures, trainLabels, testFeatures, testLabels)
            bestTestLosses.append(min(testLoss))
            bestTrainLosses.append(min(trainLoss))
        plt.plot(bestTestLosses)
        plt.xlabel('maximum iterations')
        plt.ylabel('bestTestNRMSE')
        plt.show()
        plt.plot(bestTrainLosses)
        plt.xlabel('maximum iterations')
        plt.ylabel('bestTrainNRMSE')
        plt.show()
    elif variant == 'genetic':
        # CalculateBestGeneticParameters(trainCount, featureCount, trainFeatures, trainLabels, testFeatures, testLabels)
        parameters = {
            'mutationAmplitude': 0.1,
            'regularizationStrength': 1e-05,
            'sizePopulation': 50,
            'generationsNumber': 10,
            'parentProportion': 0.1,
            'randomGeneAmplitude': 1000
            }
        max_generations = parameters['generationsNumber']
        trainLoss, testLoss = Genetic({
            'mutationAmplitude': parameters['mutationAmplitude'],
            'regularizationStrength': parameters['regularizationStrength'],
            'sizePopulation': parameters['sizePopulation'],
            'generationsNumber': max_generations + 1,
            'parentProportion': parameters['parentProportion'],
            'randomGeneAmplitude': parameters['randomGeneAmplitude']
        }, trainCount, featureCount, trainFeatures, trainLabels, testFeatures, testLabels)
        plt.plot(trainLoss, label='trainLoss')
        plt.plot(testLoss, label='testLoss')
        plt.xlabel('maximum generations')
        plt.ylabel('NRMSE')
        plt.legend()
        plt.show()
