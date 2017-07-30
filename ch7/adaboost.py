# -*- coding: utf-8 -*-
# @Author: ubuntu
# @Date:   2017-07-23 13:32:00
# @Last Modified by:   ubuntu
# @Last Modified time: 2017-07-24 20:59:44

import numpy as np
import matplotlib.pyplot as plt


def loadSmipleData():
    datMat = np.matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

    return datMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''classify data matrix

    [description]

    Args:
        dataMatrix: [description]
        dimen: [description]
        threshVal: [description]
        threshIneq: [description]

    Returns:
        [description]
        [type]
    '''
    # set retArray all one, length = dataMatrix[0]
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        # set all dataMatrix[:, dimen] <= threshVal conlum value to be -1
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0

    return retArray


def buildStump(dataArr, classLabels, D):
    '''单层决策生成函数
    加权错误率
    best strump decide by weight error which should be minimum

    Args:
        dataArr: [description]
        classLabels: [description]
        D: [description]

    Returns:
        bestStump: 最佳最小错误率单层决策树
        minError:　　最小错误率
        bestClassEst:  estimate class vector
        [type]
    '''
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0

    bestStump = {}
    bestClassEst = np.mat(np.zeros((m, 1)))
    minError = np.inf

    # iterate through all features
    for i in range(n):
        # get feature i value min, max
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        # get feature range steps
        stepSize = (rangeMax - rangeMin) / numSteps

        # iterate through featrue values, beyond threshold 1 step
        for j in range(-1, int(numSteps) + 1):
            # less than or greater than
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = \
                    stumpClassify(dataMatrix, i, threshVal, inequal)

                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                # D: weight vector
                weightedError = D.T * errArr
                # print('split: dim　%d, thresh %.2f, thresh inequal: %sm, \
                #       the weighted error is %.3f'
                #       % (i, threshVal, inequal, weightedError))

                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataArr, classLabels, numInt=80):
    '''AdaBoost algorithm

    DS: decision stump 单层决策树

    Args:
        dataArr: 数据集
        classLabels: 类别标签
        numInt: 迭代次数 (default: {40})

    Returns:
        weakClassArr: 弱分类器数组
        aggClassEst:  aggragate estimate of class
        [type]
    '''
    weakClassArr = []
    m = np.shape(dataArr)[0]
    # initiate weight vector D
    D = np.mat(np.ones((m, 1)) / m)
    # column vector, record class evaluation of each data point
    aggClassEst = np.mat(np.zeros((m, 1)))

    for i in range(numInt):
        # build a stump, return 最优单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print 'D:', D.T
        print 'error: ', error

        # get alpha                                确保没有零溢出
        # 最优单层决策树　???
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print 'classEst: ', classEst.T

        # calculate new weight vector D in next iteration
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()

        # aggragate eatimation
        aggClassEst += alpha * classEst
        print 'aggClassEst: ', aggClassEst.T

        # aggragate errors
        aggErrors = np.multiply(np.sign(aggClassEst) !=
                                np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print 'total error: ', errorRate, '\n'
        if errorRate == 0.0:
            break

    return weakClassArr, aggClassEst


def addaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    # iterate all weak classifier
    # get eatimata of classes by stumpClassify
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])

        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print aggClassEst
    return np.sign(aggClassEst)


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    # get number of fields
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def plotROC(predStrengths, classLabels):
    '''[summary]

    [description]

    Args:
        predStrengths: [description]
        classLabels: [description]
    '''
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClass = sum(np.array(classLabels) == 1.0)
    yStep = 1 / float(numPosClass)
    xStep = 1 / float(len(classLabels) - numPosClass)
    # argsort predStrengths to get array of sort index of each element
    sortedIndicies = predStrengths.argsort()
    print('argsort: ', predStrengths, sortedIndicies.tolist()[0])

    fig = plt.figure()
    fig.clf()

    ax = plt.subplot(111)
    print('classLabels: ', classLabels)
    for index in sortedIndicies.tolist()[0]:
        # get index in sortedIndicies
        # get classLabels[index] which
        #   index is sored in order of data matrix
        #   and will plot in order of Y value
        # decrease one Y step if get a label == 1.0 in y axix,
        #   x axis would not change
        # decrease one X step if get a label != 1.0 in x axis,
        #   y axis would not change
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            # AUC: Area Under Curve y length sum
            ySum += cur[1]

        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        # update cur location
        cur = (cur[0] - delX, cur[1] - delY)

    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')

    ax.axis([0, 1, 0, 1])
    plt.show()

    print 'the　Area Under the Curve is: ', ySum * xStep


if __name__ == '__main__':
    dataMat, classLabels = loadSmipleData()
    D = np.mat(np.ones((5, 1)) / 5)

    # bestStump, minError, bestClassEst = buildStump(dataMat, classLabels, D)
    # print(bestStump, minError, bestClassEst)

    weakClassArr = adaBoostTrainDS(dataMat, classLabels, numInt=10)
    # print(weakClassArr)

    # classifierArr = adaBoostTrainDS(dataMat, classLabels, 30)
    # print(classifierArr)
    # sign = addaClassify([0, 0], classifierArr)
    # print(sign)

    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 10)
    # print(classifierArray, aggClassEst)
    plotROC(aggClassEst.T, labelArr)

    # testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    # prediction10 = addaClassify(testArr, classifierArray)

    # errArr = np.mat(np.ones((67, 1)))
    # errSum = errArr[prediction10 != np.mat(testLabelArr).T].sum()
    # print(errSum / 67)
