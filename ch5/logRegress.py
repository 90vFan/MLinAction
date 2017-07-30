# -*- coding: utf-8 -*-
# @Author: ubuntu
# @Date:   2017-07-22 00:26:45
# @Last Modified by:   ubuntu
# @Last Modified time: 2017-07-22 22:41:41

import numpy as np
from matplotlib import pyplot as plt


def loadDataSet():
    '''[summary]

    load data set line by line

    Returns:
        [description]
        [type]
    '''
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # X0   X1   X2
        # 1.0              dataMat 100x3 mxn
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # label  Y         labelMat 1x100 1xm
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    '''sigmoid function

    sigmoid_X = 1.0 / (1 + exp(-X))

    Args:
        inX: [description]

    Returns:
        [description]
        number
    '''
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    '''Gradient Ascent Function

    [description]

    Args:
        dataMatIn: [description]
        classLabels: [description]

    Returns:
        [description]
        [type]
    '''
    # 100x3
    dataMatirx = np.mat(dataMatIn)
    # 100x1
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatirx)
    # initiate variament
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        # hypothesis = g(theta' * X)  100x1
        h = sigmoid(dataMatirx * weights)
        # error = Y - hypothesis
        error = (labelMat - h)
        # Gradient ascent function 梯度上升
        # weights: Thteta(3x1)   3x100 * 100x1
        # Theta := Theta  + alpha * X' * err
        weights = weights + alpha * dataMatirx.transpose() * error
    return weights


def stocGradAscent0(dataMatrix, classLabels):
    '''随机梯度上升算法

    [description]

    Args:
        dataMatrix: [description]
        classLabels: [description]

    Returns:
        [description]
        [type]
    '''
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        # 1x3 * 3x1
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    '''改进的随机梯度上升算法

    [description]

    Args:
        dataMatrix: [description]
        classLabels: [description]
        numIter: [description] (default: {150})

    Returns:
        [description]
        [type]
    '''
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    # j: iterate times
    for j in range(numIter):
        dataIndex = range(m)
        # i: taining set i
        for i in range(m):
            # change alpha by i,j
            #      => decrease as iterate times increase, but always >0.01
            # rand set randIndex
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def plotBestFit(weights):
    '''画出数据集和最佳拟合曲线

    [description]

    Args:
        weights: [description]
    '''
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x1 = np.arange(-3.0, 3.0, 0.1)
    x2 = (-weights[0] - weights[1] * x1) / weights[2]
    print(x1, x2)
    ax.plot(x1, x2)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classifyVector(inX, weights):
    '''calculate sigmoid value of X and weights

    probability > 0.5 => class 1.0

    Args:
        inX: [description]
        weights: [description]

    Returns:
        [description]
        number
    '''
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    '''load horse colic data

    parse data to array
    get traning set weights by gradient ascent function

    Returns:
        [description]
        [type]
    '''
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')

    # get train weights from traning set
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        # lineArr: array of current line features
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # train weights
    trainWeights = stocGradAscent1(np.array(trainingSet),
                                   trainingLabels, 1000)

    # get test set hypothesis(probability)
    # compare with Y, get error rate
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.
        currLine = line.strip().split('\t')
        lineArr = []
        # line array length =21
        # label = currLine[21]
        # 20 features + 1 label
        for i in range(21):
            lineArr.append(float(currLine[i]))

        # calculate test set error count
        #  classifyVector return prob h != label Y
        if int(classifyVector(np.array(lineArr), trainWeights)) != \
                int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print "the error of this test is: %f" % errorRate
    return errorRate


def multiTest():
    '''iterate colicTest() 10 times to get average error rate

    [description]
    '''
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: \
        %f" % (numTests, errorSum / float(numTests))


if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    # weights = gradAscent(dataArr, labelMat)
    # print(weights.getA())
    # plotBestFit(weights.getA())

    # weights = stocGradAscent1(np.array(dataArr), labelMat, 300)
    # plotBestFit(weights)

    multiTest()
