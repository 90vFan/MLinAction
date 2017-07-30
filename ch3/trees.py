# -*- coding: utf-8 -*-
# @Author: ubuntu
# @Date:   2017-07-14 22:54:13
# @Last Modified by:   ubuntu
# @Last Modified time: 2017-07-20 19:29:41

from math import log
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import operator
import treePlotter
import pickle


def calcShannonEnt(dataSet):
    '''caculate shannont entropy

    Arguments:
        dataSet {array} -- [description]

    Returns:
        shannonEnt -- [description]
    '''
    numEntries = len(dataSet)
    # dict of all labels {label: count}
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    # prob:      prabablity of label
    # shannoEnt: shannon entropy
    shannonEnt = 0.0
    probArr = np.array([[0, 0]])
    for key in labelCounts:
        # label key occur probability
        prob = float(labelCounts[key]) / numEntries
        # shannon entropy
        shannonEnt -= prob * log(prob, 2)
        probArr = np.append(probArr, [[prob, log(prob, 2)]], axis=0)

    # plt.plot(probArr, 'ro')
    # plt.show()
    print('shannonEnt:', shannonEnt)
    return shannonEnt


def createDataSet():
    #         浮游  脚蹼
    #   surfasing  flliper
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
               # [0, 0, 'maybe']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    '''[summary]

    split DataSet by value
    return subDataSet with specified value

    Arguments:
        dataSet {[type]} -- [description]
        axis {[type]} -- [description]
        value {[type]} -- [description]

    Returns:
        [type] -- [description]
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reduceFeatVec)

    # print(retDataSet)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """return best feature colomn location
       with minimum shannon entropy

    [description]

    Arguments:
        dataSet {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    # features count num
    numFeatures = len(dataSet[0]) - 1
    # original shannon entorpy
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # iterate all features
    for i in range(numFeatures):
        # new list of feature i
        featList = [example[i] for example in dataSet]
        # python set, all values of set is different
        # set of unique value in list
        # 获取列表唯一值
        uniqueVals = set(featList)
        newEntropy = 0.0
        # get new shannon entropy of subDataSet of value
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # infoGain 信息增益 熵减小，数据无序度减小
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
        print('infoGain', i, infoGain)

    print("bestFeature:", bestFeature)
    return bestFeature


def majorityCnt(classList):
    '''返回出现次数最多分类名称

    [description]

    Arguments:
        classList {[type]} -- [description]

    Returns:
        [type] -- [description]
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(
            classCount.iteritems(),
            key=operator.itergetter(1),
            reverse=True)

    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    '''[summary]

    return {bestFeatureLabel: {leaf point}}
    {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}

    Arguments:
        dataSet {[type]} -- [description]
        labels {[type]} -- [description]

    Returns:
        [type] -- [description]
    '''
    classList = [example[-1] for example in dataSet]
    # all class are the same, stop split
    # return all the same class
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # return most frequently occured class
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # best feature axis
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    # delete labels[bestFeat]{no surfacing}, return left {flliper}
    del(labels[bestFeat])
    # list of all features  [1, 1, 0]
    featValues = [example[bestFeat] for example in dataSet]
    #  set([0, 1])
    uniqueVals = set(featValues)
    # print(featValues, uniqueVals)

    # iterate all values in unique values set
    # best feature column locate at bestFeat
    # all values in column {bestFeat}
    # return leaf point/sub object
    #    {'flippers': {0: 'no', 1: 'yes'}}
    for value in uniqueVals:
        # copy labels to subLabels
        subLabels = labels[:]
        # 递归函数
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value),
            subLabels)

    print('myTree')
    print(myTree)
    return myTree


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")

    createPlot.ax1.annotate(nodeTxt,
                            xy=parentPt,
                            xycoords='axes fraction',
                            xytext=centerPt,
                            textcoords='axes fraction',
                            va="center",
                            ha="center",
                            bbox=nodeType,
                            arrowprops=arrow_args)


def createPlot():
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")

    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('decision point', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('leaf point', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]

    return classLabel


def storeTree(inputTree, filename):
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)


def predLenses():
    fr = open('lenses.txt', 'r')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # print(lenses)
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print('lensesTree', lensesTree)
    treePlotter.createPlot(lensesTree)


if __name__ == "__main__":
    myData, labels = createDataSet()
    # calcShannonEnt(myData)

    # splitDataSet(myData, 0, 0)

    # chooseBestFeatureToSplit(myData)

    # createTree(myData, labels)

    # createPlot()

    # myTree = treePlotter.retrieveTree(0)
    # classLabel = classify(myTree, labels, [1, 0])
    # print(classLabel)

    # storeTree(myTree, 'classifierStorage.txt')
    # gt = grabTree('classifierStorage.txt')
    # print(gt)

    predLenses()
