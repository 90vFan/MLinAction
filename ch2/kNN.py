# -*- coding: utf-8 -*-
# @Author: ubuntu
# @Date:   2017-07-13 22:53:59
# @Last Modified by:   ubuntu
# @Last Modified time: 2017-07-13 22:54:20


from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    '''
    calculate distances from inX to each element of dataSet with label
    get the most near k elements
    get their labels
    which would be inX's label

    Arguments:
        inX {[type]} -- [description]
        dataSet {[type]} -- [description]
        labels {[type]} -- [description]
        k {[type]} -- [description]

    Returns:
        [type] -- [description]
    '''

    dataSetSize = dataSet.shape[0]
    # [4,2]  4
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # [[0 1]
    #  [0 1]  [inX] - dataSet #group
    #  [0 1]
    #  [0 1]]
    sqDiffMat = diffMat ** 2
    # square
    sqDistance = sqDiffMat.sum(axis=1)
    # sum of sqDiffMat each row
    distances = sqDistance**0.5
    # sqrt
    sortedDistIndicies = distances.argsort()
    # index sort [3,1,2,0]
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # {'A': 1, 'B': 2} count A 1  count B 2
    # print("classCount", classCount)
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    # [('B', 2), ('A', 1)]
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)  # file rows
    returnMat = zeros((numberOfLines, 3))  # zeros[setSize, features]
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide

    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10  # hold out 10%
    datingDataMat, datingLabels = file2matrix(
        'datingTestSet2.txt')  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :],
                                     normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" \
            % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))
    print errorCount


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input(
        "percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges,
                                 normMat, datingLabels, 3)
    print("You will probably like this person: ",
          resultList[classifierResult - 1])


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, (32 * i) + j] = int(lineStr[j])

    # print(returnVect)
    return returnVect


def handwritingClassTest():
    hwLabels = []
    # load the training set
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" \
            % (classifierResult, classNumStr)
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount / float(mTest))


if __name__ == "__main__":
    # group, labels = createDataSet()
    # result = classify0([0, 0], group, labels, 3)
    # datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
    #            15.0 * array(datingLabels), 15.0 * array(datingLabels))
    # # plt.show()

    # ormDataSet, ranges, minVals = autoNorm(datingDataMat)
    # print(ormDataSet, ranges, minVals)

    # datingClassTest()

    # classifyPerson()

    # image2vector('testDigits/0_13.txt')

    handwritingClassTest()
