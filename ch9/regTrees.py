# -*- coding: utf-8 -*-
# @Author: ubuntu
# @Date:   2017-07-27 23:34:13
# @Last Modified by:   SailByCode
# @Last Modified time: 2017-07-30 23:20:27

import numpy as np
import matplotlib.pyplot as plt


class treeNode():
    def __init__(self, feat, val, right, left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left


def loadDataSet(fileName):
    '''general function to parse tab -delimited floats

    [description]

    Args:
        fileName: [description]

    Returns:
        [description]
        [type]
    '''
    dataMat = []  # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # map all elements to float()
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    '''二分法 binary split 
    split dataSet by split value of feature X

     > value or <= value

    Args:
        dataSet: [description]
        feature: [description]
        value: [description]

    Returns:
        [description]
        [type]
    '''
    # left subtree / right subtree
    # split by value
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    # print(dataSet[:, feature] <= value)
    # [[ True]
    # [False]
    # [ True]
    # [ True]]
    # print(np.nonzero(dataSet[:, feature] <= value)[0])
    # [0 2 3]
    return mat0, mat1


def regLeaf(dataSet):
    '''returns the mean value of leaf point dataset

    [description]

    Args:
        dataSet: [description]

    Returns:
        [description]
        [type]
    '''
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    '''总方差　　　variance * num of set

    [description]

    Args:
        dataSet: [description]

    Returns:
        [description]
        [type]
    '''
    tolVar = np.var(dataSet[:, -1]) * np.shape(dataSet)[0]
    return tolVar


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(0.1, 4)):
    '''[summary]

    choose best split 

    Args:
        dataSet: [description]
        leafType: [description] (default: {regLeaf})
        errType: [description] (default: {regErr})
        ops: [description] (default: {(1, 4)})

    Returns:
        [description]
        [type]
    '''
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # print(feat, val)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val

    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    # print('retTree: ', retTree)
    return retTree


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    '''[summary]

    1. summarize left different features
        =1 return
    2. check splited dataset total variance improvement
        if > threshhod(tolS) split
        else  create leaf point directly
    3. check splited sub dataset shape 
        if < tolN not split 
    4. else return split feature and feature value

    Args:
        dataSet: [description]
        leafType: [description] (default: {regLeaf})
        errType: [description] (default: {regErr})
        ops: [description] (default: {(1, 4)})
                tolS: 容许的误差下降值
                tolN： minimal set number to split
    Returns:
        none, leaf point mean value
        切分特征，切分值
        index of best feature to split, value of best feature
        [type]
    '''
    tolS = ops[0]
    tolN = ops[1]

    # if all the target variables are the same value: quit and return value
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # exit cond 1
        # print('no different')
        return None, leafType(dataSet)
    # caculate size and error of current feature
    m, n = np.shape(dataSet)
    # the choice of the best feature is driven by Reduction in RSS error from
    # mean
    S = errType(dataSet)
    # print('S', S)

    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    # iterate all features
    for featIndex in range(n - 1):
        # print(dataSet[:, featIndex].flatten().tolist()[0])
        # iterate all split value in dataset of featrue INDEX
        for splitVal in set(dataSet[:, featIndex].flatten().tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            # total variance
            newS = errType(mat0) + errType(mat1)
            # get best total variance bestS
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # print(bestIndex, bestValue, bestS)
    # if the decrease (S-bestS) is less than a threshold don't do the split
    #  split only if new split decrease total variance more than threshold
    if (S - bestS) < tolS:
        # print('small improvement of variance')
        return None, leafType(dataSet)  # exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # if splited dataset is in small shape, return none,leftype
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):  # exit cond 3
        # print(np.shape(mat0))
        # print(np.shape(mat1))
        # print('small shape')
        return None, leafType(dataSet)
    return bestIndex, bestValue  # returns the best feature to split on
    # and the value used for that split


def isTree(obj):
    '''check if this is a Tree or leaf point

    [description]

    Args:
        obj: [description]

    Returns:
        [description]
        [type]
    '''
    return isinstance(obj, dict)


def getMean(tree):
    '''recursive tree
    塌陷处理，返回平均值
    if find two leaf point, return mean value

    Args:
        tree: [description]
    '''
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    # if we have no test data collapse the tree
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    # if the branches are not trees try to prune them
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # if they are now both leafs, see if we can merge them　!!!
    #   leaf point, tree['left'] is value of split feature
    # error: sum of square of (testSet[:, -1] value - tree mean value or tree
    # split value)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) +\
            np.sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean
        else:
            return tree
    else:
        return tree


def linearSolve(dataSet):
    '''format dataSet to be linear function 
    helper function used in two places

    [description]

    Args:
        dataSet: [description]

    Returns:
        [description]
        [type]

    Raises:
        X Y: format dataSet to be X and Y
        ws:  linear coefficient
    '''
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))  # create a copy of data with 1 in 0th postion
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]  # and strip out Y
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):
    '''generate learf point of a linear model
    create linear model and return coeficients

    [description]

    Args:
        dataSet: [description]

    Returns:
        [description]
        [type]
    '''
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    '''total varience

    [description]

    Args:
        dataSet: [description]

    Returns:
        [description]
        [type]
    '''
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return np.sum(np.power(Y - yHat, 2))


def scatterDataSet(dataSet):
    # print(myMat3[:, 0].T.tolist()[0])
    dataMat = np.mat(dataSet)
    plt.scatter(dataMat[:, 0].T.tolist()[0], dataMat[:, 1].T.tolist()[0])
    plt.show()


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat


if __name__ == '__main__':
    # testMat = np.mat(np.eye(4))
    # print(testMat)

    # mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    # print(mat0)
    # print(mat1)

    myDat = loadDataSet('ex00.txt')
    myMat = np.mat(myDat)
    # scatterDataSet(myDat)

    # retTree = createTree(myMat)
    # print(retTree)

    myDat2 = loadDataSet('ex2.txt')
    myMat2 = np.mat(myDat2)
    # scatterDataSet(myDat2)

    # myTree = createTree(myMat2, ops=(0, 1))
    # print(myTree)

    myDat3 = loadDataSet('exp2.txt')
    myMat3 = np.mat(myDat3)

    # myTree3 = createTree(myMat3, modelLeaf, modelErr, (1, 10))
    # print(myTree3)

    trainMat = np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))

    regTree = createTree(trainMat, ops=(1, 20))
    yHat = createForeCast(regTree, testMat[:, 0])
    regCoef = np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
    print(regCoef)

    modelTree = createTree(trainMat, modelLeaf, modelErr, (1,20))
    yHat = createForeCast(modelTree, testMat[:, 0], modelTreeEval)
    modelCoef = np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
    print(modelCoef)

    ws, X, Y = linearSolve(trainMat)
    yHat = testMat[:, 0] * ws[1, 0] + ws[0, 0]
    linearCoef = np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
    print(linearCoef)
