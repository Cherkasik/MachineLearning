import mnist
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

def RunNN(trainImages, trainLabels, testImages, testLabels):
    nnModel = nn.Sequential(
        nn.Conv1d(1, 8, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Softmax2d()
    )
    nnModel.type(torch.FloatTensor)
    loss = nn.CrossEntropyLoss().type(torch.FloatTensor)
    optimizer = optim.Adam(nnModel.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

    trainImages = Variable(torch.from_numpy(trainImages))
    trainLabels = Variable(torch.from_numpy(trainLabels))

    def Train(model, loss, optimizer, epochsNumber, trainImages, trainLabels):
        lossHistory = []
        trainHistory = []
        for epoch in range(epochsNumber):
            print('--- Epoch %d ---' % (epoch + 1))
            model.train()
            lossAccum = 0
            correct = 0
            total = 0
            for iStep, (x, y) in enumerate(zip(trainImages, trainLabels)):
                prediction = model(x)
                lossValue = loss(prediction, y)
                optimizer.zero_grad()
                lossValue.backward()
                optimizer.step()
                _, indices = torch.max(prediction, 1)
                correct += torch.sum(indices == y)
                total += y.shape[0]
                lossAccum += lossValue
            averageLoss = lossAccum / (iStep + 1)
            trainAccuracy = float(correct) / total
            lossHistory.append(float(averageLoss))
            trainHistory.append(trainAccuracy)
            print("Average loss: %f, Train accuracy: %f" % (averageLoss, trainAccuracy))
        return lossHistory, trainHistory
    
    Train(nnModel, loss, optimizer, 3, trainImages, trainLabels)

if __name__ == '__main__':
    dataset = input()
    if dataset == 'mnist':
        RunNN(mnist.train_images()[:1000], mnist.train_labels()[:1000], mnist.test_images()[:1000], mnist.test_labels()[:1000])
    if dataset == 'fminst':
        (trainImages, trainLabels), (testImages, testLabels) = tf.keras.datasets.fashion_mnist.load_data()
        RunNN(trainImages[:1000], trainLabels[:1000], testImages[:1000], testLabels[:1000])