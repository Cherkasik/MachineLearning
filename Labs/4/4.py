import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, accuracy_score
import matplotlib.pyplot as plt

def LoadData():
    dataset = []
    subdirectories = [f.path for f in os.scandir('./data') if f.is_dir()]
    for subdirectory in subdirectories:
        files = [f.path for f in os.scandir(subdirectory) if f.is_file()]
        messages = []
        for filePath in files:
            file = open(filePath)
            features = np.array(file.read())
            message = np.array([features, 1 if 'legit' in filePath else 0])
            messages.append(message)
            file.close()
        dataset.append(messages)
    return np.array(dataset)

def SplitData(testPartNum, dataset):
    trainMessages = []
    for partNumber in range(len(dataset)):
        if partNumber != testPartNum:
            for message in dataset[partNumber]:
                trainMessages.append(message)
    trainMessages = np.array(trainMessages)
    return trainMessages[:, 0], trainMessages[:, 1], dataset[testPartNum][:, 0], dataset[testPartNum][:, 1]

def GetFeaturesLabels(dataset):
    messages = []
    for i in dataset:
        for message in i:
            messages.append(message)
    dataset = np.array(messages)
    features = dataset[:, 0]
    labels = dataset[:, 1]
    return dataset, features, labels

def MeanAccuracyPlotter(dataset, minSpamPropotion, vectorizer):
    spamProportion = 0.5
    accuracies = []
    spamProportions = []
    while spamProportion >= minSpamPropotion:
        accuracies.append(MeanAccuracy(dataset, vectorizer, spamProportion))
        spamProportions.append(spamProportion)
        spamProportion -= 0.005
    plt.plot(spamProportions, accuracies)
    print(accuracies, spamProportions)
    plt.ylabel('Accuracy')
    plt.xlabel('Spam proportion')
    plt.show()

def MeanAccuracy(dataset, vectorizer, spamProportion=0,):
    accuracies = []
    for partNumber in range(len(dataset)):
        model = MultinomialNB() if spamProportion == 0 else MultinomialNB(fit_prior=True, class_prior=[spamProportion, 1 - spamProportion])
        trainFeatures, trainLabels, testFeatures, testLabels = SplitData(partNumber, dataset)
        trainWordCounts = vectorizer.fit_transform(trainFeatures)
        testWordCounts = vectorizer.transform(testFeatures)
        model.fit(trainWordCounts, trainLabels)
        accuracies.append(accuracy_score(testLabels, model.predict(testWordCounts)))
    return sum(accuracies)/len(accuracies)

def CountFalsePositives(dataset, minSpamPropotion):
    dataset, features, labels = GetFeaturesLabels(dataset)
    # Learn the vocabulary dictionary and return term-document matrix
    trainWordCounts = vectorizer.fit_transform(features)
    # Transform documents to document-term matrix
    testWordCounts = vectorizer.transform(features)
    model = MultinomialNB(fit_prior=True, class_prior=[minSpamPropotion, 1 - minSpamPropotion])
    model.fit(trainWordCounts, labels)
    falsePositiveCount = sum([1 for actual, predicted in zip(labels, model.predict(testWordCounts)) if (actual != predicted) and (actual == '1')])
    print(falsePositiveCount)

def ROC(dataset, vectorizer):
    dataset, features, labels = GetFeaturesLabels(dataset)
    trainWordCounts = vectorizer.fit_transform(features)
    testWordCounts = vectorizer.transform(features)
    model = MultinomialNB()
    model.fit(trainWordCounts, labels)
    falsePositiveRate, truePositiveRate, _ = roc_curve(labels, model.predict_proba(testWordCounts)[:, 0], pos_label='0')
    plt.plot(falsePositiveRate, truePositiveRate)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

if __name__ == '__main__':
    minSpamPropotion = 10 ** -85
    dataset = LoadData()
    vectorizer = CountVectorizer()
    # CountFalsePositives(dataset, minSpamPropotion)
    # MeanAccuracyPlotter(dataset, minSpamPropotion, vectorizer)
    ROC(dataset, vectorizer)