from torchvision import datasets, transforms
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def RunNN(trainImages, trainLabels, testImages, testLabels):
    nnModel = nn.Sequential(
        nn.Conv1d(1, 10, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Softmax2d()
    )
    nnModel.type(torch.FloatTensor)
    loss = nn.CrossEntropyLoss().type(torch.FloatTensor)
    optimizer = optim.Adam(nnModel.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

    def Train(model, loss, optimizer, epochsNumber, trainImages, trainLabels):
        lossHistory = []
        trainHistory = []
        for epoch in range(epochsNumber):
            print('--- Epoch %d ---' % (epoch + 1))
            model.train()
            lossAccum = 0
            correct = 0
            total = 0
            for i in range(1000):
                prediction = model(trainImages[i])
                lossValue = loss(prediction, trainLabels[i])
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
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == 'mnist':
        train = datasets.MNIST('.', download=True, train=True, transform=transform)
        test = datasets.MNIST('.', download=True, train=False, transform=transform)
        trainLoader = torch.utils.data.DataLoader(train, batch_size=1000)
        testLoader = torch.utils.data.DataLoader(test, batch_size=1000)
        trainIter = iter(trainLoader)
        trainImages, trainLabels = trainIter.next()
        testIter = iter(testLoader)
        testImages, testLabels = testIter.next()
        RunNN(trainImages, trainLabels, testImages, testLabels)
    if dataset == 'fmnist':
        train = datasets.FashionMNIST('.', download=True, train=True, transform=transform)
        test = dataset.FashionMNIST('.', download=True, train=False, transform=transform)
        trainLoader = torch.utils.data.DataLoader(train, batch_size=1000, shuffle=True)
        testLoader = torch.utils.data.DataLoader(test, batch_size=1000, shuffle=True)
        trainIter = iter(trainLoader)
        testIter = iter(testLoader)
        RunNN(trainIter, testIter)