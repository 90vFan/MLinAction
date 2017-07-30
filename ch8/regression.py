# -*- coding: utf-8 -*-
# @Author: ubuntu
# @Date:   2017-07-24 22:43:39
# @Last Modified by:   ubuntu
# @Last Modified time: 2017-07-27 01:08:31

import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import json
import urllib2


def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        currLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(currLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(currLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    '''普通线性回归
    计算最佳拟合曲线　ws

    ws = 1/(X'X) * X' * y

    Args:
        xArr: [description]
        yArr: [description]

    Returns:
        ws:  regression coefficient　回归系数
        [type]
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    # X'*X
    xTx = xMat.T * xMat

    # linalg: line algebra 线性代数  determinant行列式
    # check if det
    if np.linalg.det(xTx) == 0.0:
        print 'This matrix is singular, cannot do inverse'
        return

    # ws = 1/(X'X) * X' * y
    ws = xTx.I * (xMat.T * yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    '''LWLR 局部加权线性回归函数
    计算最佳拟合曲线　ws

    Gaussian kernel function:
        distance from point x_i to all points x
        divide -2*k^2
        apply to exp
    w(i, i) = exp(|x_i - x| / (-2 * k^2))

    Args:
        testPoint: [description]
        xArr: [description]
        yArr: [description]
        k: 控制衰减速度 (default: {1.0})

    Returns:
        testPoint * ws: estimate y
        [type]
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    m = np.shape(xMat)[0]
    # 对角矩阵 mxm
    weights = np.mat(np.eye((m)))

    # Gaussian kernel
    for j in range(m):
        # distance to xMat row set j
        diffMat = np.mat(testPoint) - xMat[j, :]
        # weights 随着样本点与待预测点距离递增，以指数级衰减　　　　
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print 'This matrix is sinular, cannot do inverse'
        return
    # regression coefficient
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    '''test of lwlr for testArr

    [description]

    Args:
        testArr: [description]
        xArr: [description]
        yArr: [description]
        k: [description] (default: {1.0})

    Returns:
        [description]
        [type]
    '''
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def plotCurve(xArr, yArr, ws):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    # yHat = xMat * ws

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # mat.flatten => matrix to one row matrix
    # matrix.A    => matrix to array
    ax.scatter(xMat[:, 1].flatten().A[0],
               yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    print('xmat: ', xCopy[0:2])
    xCopy.sort(0)
    print('xmat.sort: ', xCopy[0:2])
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)

    # print(yHat[0:2], yMat[0][0])
    # 相关系数
    relate = np.corrcoef(yHat.T, yMat)
    print('related corrcoef: ', relate)

    plt.show()


def plotRegress(xArr, yArr, yHat):
    xMat = np.mat(xArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])

    ax.scatter(xMat[:, 1].flatten().A[0],
               np.mat(yArr).T.flatten().A[0],
               s=2, c='red')
    plt.show()


def wsTest():
    xArr, yArr = loadDataSet('ex0.txt')
    print('xArr: ', xArr[0:2])
    print('yArr: ', yArr[0:2])

    # standard regress
    ws = standRegres(xArr, yArr)
    print('ws: ', ws)

    plotCurve(xArr, yArr, ws)

    # lwlr regress
    est = lwlr(xArr[0], xArr, yArr, 1.0)
    print('y0:', yArr[0])
    print('est0: ', est)

    yHat = lwlrTest(xArr, xArr, yArr, 0.03)
    print(yHat)

    plotRegress(xArr, yArr, yHat)


def rssError(yArr, yHatArr):
    err = ((yArr - yHatArr) ** 2).sum()
    print(err)
    return err


def abaloneTest():
    abX, abY = loadDataSet('abalone.txt')
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)

    print('lwlr training set error:')
    rssError(abY[0:99], yHat01.T)
    rssError(abY[0:99], yHat1.T)
    rssError(abY[0:99], yHat10.T)

    # 56.7886874305
    # 429.89056187
    # 549.118170883

    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)

    print('lwlr test set error:')
    rssError(abY[100:199], yHat01.T)
    rssError(abY[100:199], yHat1.T)
    rssError(abY[100:199], yHat10.T)
    # lwlr regress error
    # 57913.5155016
    # 573.526144189  k=10
    # 517.571190538

    # when k=10, get the best minimum train and test error

    ws = standRegres(abX[0:99], abY[0:99])
    yHat = np.mat(abX[100:199]) * ws
    print('standar regress test error: ')
    rssError(abY[100:199], yHat.T.A)
    # standard regress error
    # 518.636315325  k=10

    ws = ridgeRegrss(np.mat(abX[0:99]), np.mat(abY[0:99]).T)
    yHat = np.mat(abX[100:199]) * ws
    print('ridge regress test error: ')
    rssError(abY[100:199], yHat.T.A)
    # ridge　regress
    # 429.629980174

    ridgeWeights = ridgeTest(abX, abY)

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()


def ridgeRegrss(xMat, yMat, lamb=0.2):
    '''ridge regress 岭回归

    [description]

    Args:
        xMat: [description]
        yMat: [description]
        lamb: [description] (default: {0.2})

    Returns:
        ws
        [type]
    '''
    xTx = xMat.T * xMat
    # xTx + I * lambda
    denom = xTx + np.eye(np.shape(xMat)[1]) * lamb
    if np.linalg.det(denom) == 0.0:
        print 'This matrix is singular, cannot do inverse'
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    '''[summary]

    [description]

    Args:
        xArr: [description]
        yArr: [description]

    Returns:
        [description]
        [type]
    '''
    # data normalize 标准化
    #   x - x Means / x Variable
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar

    # 30 different lambda
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegrss(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


def regularize(xMat):
    '''regularize by columns

    [description]

    Args:
        xMat: [description]

    Returns:
        [description]
        [type]
    '''
    inMat = xMat.copy()
    # calc mean then subtract it off
    inMeans = np.mean(inMat, 0)
    # calc variance of Xi then divide by it
    inVar = np.var(inMat, 0)
    inMat = (inMat - inMeans) / inVar
    return inMat


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    '''前向逐步线性回归算法　step by step linear regress

    [description]

    Args:
        xArr: input data set X
        yArr: output label Y
        eps: tune step of each iteration (default: {0.01})
        numIt: number of iteration (default: {100})

    Returns:
        [description]
        [type]
    '''
    # normalize
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))

    # initiate ws, wsTest, wsMax
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()

    # iterate numIt times, get lowest error
    for i in range(numIt):
        print ws.T

        lowestError = np.inf
        # iterate through all features
        for j in range(n):
            # increase/decrease feature
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += (eps * sign)
                yTest = xMat * wsTest
                # error
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest

        ws = wsMax.copy()
        returnMat[i, :] = ws.T

    return returnMat


def stageWiseTest():
    '''compare stage wise return and stand regress return

    [description]
    '''
    xArr, yArr = loadDataSet('abalone.txt')
    retMat = stageWise(xArr, yArr, 0.001, 5000)
    print(retMat)

    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xMat = regularize(xMat)
    yM = np.mean(yMat, 0)
    yMat = yMat - yM
    weights = standRegres(xMat, yMat.T)
    print(weights.T)


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (
        myAPIstr, setNum)
    # get API json data, and loads it to be an object
    print(searchURL)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())

    print(retDict)
    for i in range(len(retDict['item'])):
        try:
            currItem = retDict['item'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0

            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print '%d\t%d\t%d\t%f\t%f' % \
                        (yr, numPce, newFlag, origPrc, sellingPrice)
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except Exception:
            print 'problem with item %d' % i


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    # searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    # searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    # searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    # searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    # searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


if __name__ == '__main__':
    # wsTest()

    # abaloneTest()

    # stageWiseTest()

    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
