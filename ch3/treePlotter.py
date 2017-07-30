'''
Created on Oct 14, 2010

@author: Peter Harrington
'''
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def getNumLeafs(myTree):
    '''[summary]

    get leafs count
    leafs point: {0:'no'}
    count leafs which is not dict

    Arguments:
        myTree {[type]} -- [description]

    Returns:
        [type] -- [description]
    '''
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # test to see if the nodes are dictonaires, if not they are leaf nodes
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    '''[summary]

    count tree depth
    count when target is a dict

    Arguments:
        myTree {[type]} -- [description]

    Returns:
        [type] -- [description]
    '''
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # test to see if the nodes are dictonaires, if not they are leaf nodes
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(
        nodeTxt, xy=parentPt, xycoords='axes fraction',
        xytext=centerPt, textcoords='axes fraction',
        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center",
                        ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    '''[summary]

    ((0.5, 1.0), 'no surfacing')
    (0.125, 0.6666666666666667, 'no')
    ((0.625, 0.6666666666666667), 'flippers')
    ((0.5, 0.3333333333333334), 'head')
    (0.375, 1.1102230246251565e-16, 'no')
    (0.625, 1.1102230246251565e-16, 'yes')
    (0.875, 0.3333333333333334, 'no')

    Arguments:
        myTree {[type]} -- [description]
        parentPt {[type]} -- [description]
        nodeTxt {[type]} -- [description]
    '''
    # if the first key tells you what feat was split on
    #
    # this determines the x width of this tree
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    # the text label for this node should be this
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) /
              2.0 / plotTree.totalW, plotTree.yOff)

    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    print(cntrPt, firstStr)

    # myTree["no surfacing"]
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD

    for key in secondDict.keys():
        # test to see if the nodes are dictonaires, if not they are leaf nodes
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff,
                                       plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
            print(plotTree.xOff, plotTree.yOff, secondDict[key])
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD
# if you do get a dictonary you know it's a tree, and the first element
# will be another dict


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

# def createPlot():
#    fig = plt.figure(1, facecolor='white')
#    fig.clf()
#    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
#    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#    plt.show()


def retrieveTree(i):
    listOfTrees = \
        [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
         {'no surfacing':
            {0: 'no', 1: {'flippers': {
                0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
         ]
    return listOfTrees[i]

# createPlot(thisTree)


if __name__ == '__main__':
    myTree = retrieveTree(1)
    print(getNumLeafs(myTree))
    print(getTreeDepth(myTree))

    createPlot(myTree)
