# -*- coding: utf-8 -*-
# @Author: ubuntu
# @Date:   2017-07-20 21:32:35
# @Last Modified by:   ubuntu
# @Last Modified time: 2017-07-21 01:03:28


from numpy import *
import re
import feedparser


def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    '''[summary]

    create volcabulary list
    :param dataSet: [description]
    :type dataSet: [type]
    :returns: list of unique vocabulary set
    ['cute', 'love', 'help', ..., 'mr', 'steak', 'my']
    :rtype: {[type]}
    '''
    vocabSet = set([])
    for document in dataSet:
        # 并集 | OR return unique volcabularies
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    '''set of words model

    convert document to be a vocabulary list
    :param vocabList: list of all vocabulary
    :type vocabList: [type]
    :param inputSet: set of document
    :type inputSet: [type]
    :returns: vocabulary words list
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    :rtype: {[type]}
    '''
    # vocabulary vector of [0]
    returnVec = [0] * len(vocabList)
    # return vocabVec of [0, 1, 0, 1] which 1 means word occurrence
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)

    return returnVec


def bagOfWords2VecMN(vocabList, inputSet):
    '''bag of words model

    convert document to be vocabulary list of words occurrence counter

    Args:
        vocabList: [description]
        inputSet: [description]

    Returns:
        [description]
        [type]
    '''
    returnVec = [0] * len(vocabList)
    # return words vector of [0, 1, 3, 0] which each num means word
    # occurrence times
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    '''[summary]

    naive beyes classifier training function

    Args:
        trainMatrix:   matrix of document vocabulary list
        trainCategory: labels of all documents

    Returns:
        p0Vect:  vector of probability 0
        p1Vect:  vector of probability 1
        pAusive: probability of all class 1(abusive)  == 0.5
        [type]
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # initiate probabilty
    # pXNum: vector of documents of class 0/1
    # pXDenom: sum of words in this class
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0

    print(trainMatrix)
    # iterate train matirx of document vocabulary list
    # trainMatrix: matrix of words vector
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # vector: probablity of document class
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)

    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''[summary]

    calculate probabilty of words vector
    compare p1 p0 to classify as 1/0

    Args:
        vec2Classify: [description]
        p0Vec: [description]
        p1Vec: [description]
        pClass1: [description]

    Returns:
        class 0/1
        number
    '''
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)

    if p1 > p0:
        return 1
    else:
        return 0


def textParse(bigString):
    '''parse text to list

    input is big string,
    output is word list
    '''
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    # read and parse text
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    # 50 emails total
    trainingSet = range(50)
    # create test set, randomly choose 10 training set as test set
    # delete test set in training set
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    # train the classifier use left training set (get probs) trainNB0
    # matrix of words vector
    trainMat = []
    # list of class
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    # caculate error rate
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != \
                classList[docIndex]:
            errorCount += 1
            print "classification error", docList[docIndex]
    print 'the error rate is: ', float(errorCount) / len(testSet)
    # return vocabList,fullText


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print(myVocabList)
    vocabVec = setOfWords2Vec(myVocabList, listOPosts[0])
    print(vocabVec)

    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    print(p0V, p1V, pAb)

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classfied as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classfied as: ', classifyNB(thisDoc, p0V, p1V, pAb)


def calcMostFreq(vocabList, fullText):
    '''calculate most occurrence frequency

    [description]

    Args:
        vocabList: [description]
        fullText: [description]

    Returns:
        [description]
        [type]
    '''
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1),
                        reverse=True)
    return sortedFreq[:30]
    # return []


def localWords(feed1, feed0):
    '''[summary]

    remove most frequent words

    Args:
        feed1: [description]
        feed0: [description]

    Returns:
        [description]
        [type]
    '''
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    # visit one rss source one time
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # create vocabulary list with document list
    vocabList = createVocabList(docList)
    # remove top 30 words
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = range(2 * minLen)
    testSet = []
    # create test set
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != \
                classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount) / len(testSet)
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []

    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))

    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**" * 10
    for item in sortedSF:
        print item[0]

    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "Ny**" * 10
    for item in sortedNY:
        print item[0]


def getTopWords1(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]


if __name__ == '__main__':
    # testingNB()

    # spamTest()

    ny = feedparser.parse(
        'https://newyork.craigslist.org/search/stp?format=rss')
    sf = feedparser.parse('https://sfbay.craigslist.org/search/stp?format=rss')
    # vocabList, pSF, pNY = localWords(ny, sf)

    getTopWords(ny, sf)
