import mnist
import tensorflow as tf
import numpy as np
from pandas import DataFrame

class AdamOptimizer:
    def __init__(self, weights, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0
        self.weights = weights

    # алгоритм коррекции весов в зависимости от градиента
    def BackwardPass(self, gradient):
        self.t = self.t + 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        mHat = self.m / (1 - self.beta1 ** self.t)
        vHat = self.v / (1 - self.beta2 ** self.t)
        self.weights = self.weights - self.alpha * (mHat / (np.sqrt(vHat) - self.epsilon))
        return self.weights

class Conv3x3:
    def __init__(self, filtersNumber):
        self.filtersNumber = filtersNumber
        self.filters = np.random.randn(filtersNumber, 3, 3) / 9
        self.adam = AdamOptimizer(self.filters)

    def IterateRegions(self, image):
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                imRegion = image[i:(i + 3), j:(j + 3)]
                yield imRegion, i, j

    def ForwardPropagation(self, input):
        self.lastInput = input
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.filtersNumber))
        for imRegion, i, j in self.IterateRegions(input):
            output[i, j] = np.sum(imRegion * self.filters, axis=(1, 2))
        return output

    # обратное распространение ошибки. Коррекция параметров в зависимости от промахов
    def BackPropagation(self, inputGradients):
        filters = np.zeros(self.filters.shape)
        for imRegion, i, j in self.IterateRegions(self.lastInput):
            for f in range(self.filtersNumber):
                filters[f] += inputGradients[i, j, f] * imRegion
        self.filters = self.adam.BackwardPass(filters)
        return None

class MaxPool2:
    def IterateRegions(self, image):
        h, w, _ = image.shape
        newH = h // 2
        newW = w // 2

        for i in range(newH):
            for j in range(newW):
                imRegion = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield imRegion, i, j

    def ForwardPropagation(self, input):
        self.lastInput = input
        h, w, filtersNumber = input.shape
        output = np.zeros((h // 2, w // 2, filtersNumber))
        for imRegion, i, j in self.IterateRegions(input):
            output[i, j] = np.amax(imRegion, axis=(0, 1))
        return output

    def BackPropagation(self, inputGradients):
        dLdInput = np.zeros(self.lastInput.shape)
        for imRegion, i, j in self.IterateRegions(self.lastInput):
            h, w, f = imRegion.shape
            amax = np.amax(imRegion, axis=(0, 1))
            for k in range(h):
                for l in range(w):
                    for m in range(f):
                        if imRegion[k, l, m] == amax[m]:
                            dLdInput[i * 2 + k, j * 2 + l, m] = inputGradients[i, j, m]
        return dLdInput

# получает пиксели на вход, примеряет к ним преобразование из весов, далее экспонента и взятие ее процентного вклада
class Softmax:
    def __init__(self, inputLen, nodes):
        self.weights = np.random.randn(inputLen, nodes) / inputLen
        # добавочные веса
        self.biases = np.zeros(nodes)
        self.adamWeights = AdamOptimizer(self.weights)
        self.adamBiases = AdamOptimizer(self.biases)

    def ForwardPropagation(self, input):
        self.lastInputShape = input.shape
        input = input.flatten()
        self.lastInput = input
        totals = np.dot(input, self.weights) + self.biases
        self.lastTotals = totals
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def BackPropagation(self, inputGradients):
        for i, gradient in enumerate(inputGradients):
            if gradient == 0:
                continue
            totalExp = np.exp(self.lastTotals)
            sumExp = np.sum(totalExp)
            dOutdt = -totalExp[i] * totalExp / (sumExp ** 2)
            dOutdt[i] = totalExp[i] * (sumExp - totalExp[i]) / (sumExp ** 2)
            dLdt = gradient * dOutdt
            dLdw = self.lastInput[np.newaxis].T @ dLdt[np.newaxis]
            dLdInputs = self.weights @ dLdt
            self.weights = self.adamWeights.BackwardPass(dLdw)
            self.biases = self.adamBiases.BackwardPass(dLdt)
            return dLdInputs.reshape(self.lastInputShape)

def RunNN(trainImages, trainLabels, testImages, testLabels):
    filtersAmount = 8
    nodesAmount = 10  # количество классов
    initialSize = 26  # размер изображения
    # 28x28x1 (исходный размер) -> 26x26x8
    conv = Conv3x3(filtersAmount)
    # 26x26x8 -> 13x13x8
    pool = MaxPool2()
    softmax = Softmax((initialSize // 2) * (initialSize // 2) * filtersAmount, nodesAmount)

    def ForwardPropagation(image, label):
        # нормировка данных в изображении
        out = conv.ForwardPropagation((image / 255) - 0.5)
        out = pool.ForwardPropagation(out)
        out = softmax.ForwardPropagation(out)
        loss = -np.log(out[label])
        acc = 1 if np.argmax(out) == label else 0
        return out, loss, acc

    def Train(im, label):
        out, loss, acc = ForwardPropagation(im, label)
        gradient = np.zeros(nodesAmount)
        gradient[label] = -1 / out[label]
        gradient = softmax.BackPropagation(gradient)
        gradient = pool.BackPropagation(gradient)
        gradient = conv.BackPropagation(gradient)
        return loss, acc

    for epoch in range(3):
        print('--- Epoch %d ---' % (epoch + 1))
        permutation = np.random.permutation(len(trainImages))
        trainImages = trainImages[permutation]
        trainLabels = trainLabels[permutation]

        loss = 0
        numCorrect = 0
        for i, (im, label) in enumerate(zip(trainImages, trainLabels)):
            if i % 100 == 99:
                print(
                    '[Step %d]: Average Loss %.3f | Accuracy: %d%%' %
                    (i + 1, loss / 100, numCorrect)
                )
                loss = 0
                numCorrect = 0
            l, isRight = Train(im, label)
            loss += l
            numCorrect += isRight

    loss = 0
    numCorrect = 0
    predictions = []

    def GetLabel(probs):
        label = 0
        labelProb = probs[0]
        for i in range(1, len(probs)):
            if probs[i] > labelProb:
                label = i
                labelProb = probs[i]
        return label

    for im, label in zip(testImages, testLabels):
        out, l, isRight = ForwardPropagation(im, label)
        predictions.append(GetLabel(out))
        loss += l
        numCorrect += isRight

    numTests = len(testImages)
    print('test loss:', loss / numTests)
    print('test accuracy:', numCorrect / numTests)
    error_rate = 1 - numCorrect / numTests
    print('error rate:', error_rate)

    cm = [[0 for j in range(nodesAmount)] for i in range(nodesAmount)]
    for p, l in zip(predictions, testLabels):
        cm[l][p] += 1

    print(DataFrame(cm))


if __name__ == '__main__':
    dataset = input()
    if dataset == 'mnist':
        RunNN(mnist.train_images()[:1000], mnist.train_labels()[:1000], mnist.test_images()[:1000], mnist.test_labels()[:1000])
    if dataset == 'fminst':
        (trainImages, trainLabels), (testImages, testLabels) = tf.keras.datasets.fashion_mnist.load_data()
        RunNN(trainImages[:1000], trainLabels[:1000], testImages[:1000], testLabels[:1000])