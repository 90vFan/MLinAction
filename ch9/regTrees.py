# -*- coding: utf-8 -*-
# @Author: ubuntu
# @Date:   2017-07-27 23:34:13
# @Last Modified by:   ubuntu
# @Last Modified time: 2017-07-28 00:28:52

import numpy as np


class treeNode():
    def __init__(self, feat, val, right, left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left


def regLeaf(dataSet):  # returns the value used for each leaf
    return mean(dataSet[:, -1])


def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        currLine = line.strip().split('\t')
        # map each poiont to float
        fltLine = np.map(float, currLine)
        dataMat.append(fltLine)

    return dataMat


def binSplitDataSet(dataSet, feature, value):
    '''二分法 binary split 

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
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :][0]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :][0]
    return mat0, mat1


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val

    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)

    return retTree


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    # if all the target variables are the same value: quit and return value
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # exit cond 1
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    # the choice of the best feature is driven by Reduction in RSS error from
    # mean
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS:
        return None, leafType(dataSet)  # exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # exit cond 3
        return None, leafType(dataSet)
    return bestIndex, bestValue  # returns the best feature to split on
    # and the value used for that split


if __name__ == '__main__':
    testMat = np.mat(np.eye(4))
    print(testMat)

    mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    print(mat0)
    print(mat1)
