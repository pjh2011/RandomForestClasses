import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


class DecisionTreeNode(object):

    def __init__(self, indices):
        self.indices = indices
        self.leftChild = None
        self.rightChild = None
        self.leaf = False

    def setSplit(self, splitFeat, splitVal, informationGain, nObsAtSplit):
        self.splitFeat = splitFeat
        self.splitVal = splitVal
        self.informationGain = informationGain
        self.nObsAtSplit = nObsAtSplit

    def setLeft(self, leftChild):
        self.leftChild = leftChild

    def setRight(self, rightChild):
        self.rightChild = rightChild

    def setLeaf(self, classes):
        self.leaf = True

        counts = Counter(classes)

        self.maxClass = counts.most_common(1)[0][0]
        self.percentMax = 1.0 * \
            counts.most_common(1)[0][1] / sum(counts.values())


class DecisionTree(object):

    def __init__(self, minIG=None, nPointsLeaf=1,
                 nFeaturesToSplit=None):
        self.minIG = minIG
        self.nPointsLeaf = nPointsLeaf
        self.root = None
        self.nFeaturesToSplit = nFeaturesToSplit

    def fit(self, XTrain, yTrain):
        '''
        fit takes in an array of training data with n observations and k
        features and a 1-d array of labels for the training data. It fits a
        decision tree to the training data.

        Input: XTrain as a numpy array shape (n, k) where n is the number of
                observations and k is the number of features
             : yTrain as a 1-d numpy array with n elements

        Output: None
        '''
        self.XTrain = XTrain
        self.yTrain = yTrain
        self.nFeatures = self.XTrain.shape[1]
        self.nObs = self.XTrain.shape[0]

        self.splitCandidates = {}

        for i in xrange(self.nFeatures):
            x = self.XTrain[:, i]
            xUnique = np.unique(x)
            XUniqueAndSorted = np.sort(xUnique)
            self.splitCandidates[i] = XUniqueAndSorted

        self.root = DecisionTreeNode(indices=np.array(range(self.nObs)))
        queue = [self.root]

        while len(queue) > 0:
            currNode = queue.pop(0)
            indices = currNode.indices
            split = self.chooseSplit_(indices, self.nFeaturesToSplit)

            if split is not None:
                splitFeat, splitVal, informationGain, nObsAtSplit = split
                currNode.setSplit(splitFeat, splitVal, informationGain,
                                  nObsAtSplit)
                leftIndices = indices[np.where(
                    XTrain[indices, splitFeat] <= splitVal)[0]]
                rightIndices = indices[np.where(
                    XTrain[indices, splitFeat] > splitVal)[0]]

                leftChild = DecisionTreeNode(indices=leftIndices)
                rightChild = DecisionTreeNode(indices=rightIndices)

                currNode.setLeft(leftChild)
                currNode.setRight(rightChild)

                queue.append(leftChild)
                queue.append(rightChild)
            else:
                currNode.setLeaf(self.yTrain[indices])

    def gini_(self, classes):
        '''
        Calculates the Gini Impurity metric for a given set of class labels
        (https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)

        Input: classes as a numpy array of class labels
        Output: Gini impurity defined as SUM over i [p_i * (1 - p_i)]
        '''

        class_labels = np.unique(classes)
        n_data_points = len(classes)

        probs = [1.0 * np.sum(classes == i) /
                 n_data_points for i in class_labels]
        gini = 0

        for p_i in probs:
            gini += p_i * (1 - p_i)

        return gini

    def informationGain_(self, classesBefore, classesAfter):
        '''
        Calculates the information gain from splitting a set of class labels
        into two different sets. Information gain is the decrease in Gini
        Impurity achieved by splitting the class labels.

        Input: classesBefore is a numpy array of the classes, before splitting
             : classesAfter is a list of two numpy arrays. Each numpy array
                contains the class labels of one split of the original class
                labels
        Output: information gain, defined as the Gini impurity of the class
                labels prior to splitting, minus the weighted average of Gini
                impurities of the splits
        '''

        splitA = classesAfter[0]
        splitB = classesAfter[1]

        nPreSplit = len(classesBefore)
        nSplitA = len(splitA)
        nSplitB = len(splitB)

        weightSplitA = 1.0 * nSplitA / nPreSplit
        weightSplitB = 1.0 * nSplitB / nPreSplit

        giniPreSplit = self.gini_(classesBefore)
        giniSplitA = self.gini_(splitA)
        giniSplitB = self.gini_(splitB)

        giniSplits = weightSplitA * giniSplitA + weightSplitB * giniSplitB

        informationGain = giniPreSplit - giniSplits

        return informationGain

    def chooseSplit_(self, indices, nFeaturesToSplit=None):
        '''
        Input: numpy array of indices to consider for a given split
        Output: list  - First element: is the feature to split on
                      - Second: is the value on which to split
                      - Third: is Information Gain at split
                      - Fourth: is number of data points at the split
        '''
        XAtSplit = self.XTrain[indices, :]
        yAtSplit = self.yTrain[indices]

        maxInformationGain = 0
        splitFeat = None
        splitVal = None

        if len(indices) <= self.nPointsLeaf:
            return None

        if nFeaturesToSplit is None:
            featuresToTry = range(self.nFeatures)
        else:
            nFeatures = self.XTrain.shape[1]
            featuresToTry = np.random.choice(range(nFeatures),
                                             size=nFeaturesToSplit,
                                             replace=False)

        # loop over all features being checked for splits
        for feat in featuresToTry:
            # array of unique, sorted values to try splitting on
            splitsToTry = self.splitCandidates[feat]

            # loop over all candidate splits for that feature
            for split in splitsToTry:

                indicesLeft = np.where(XAtSplit[:, feat] <= split)[0]
                indicesRight = np.where(XAtSplit[:, feat] > split)[0]

                classes = yAtSplit
                classesLeft = yAtSplit[indicesLeft]
                classesRight = yAtSplit[indicesRight]

                informationGain = self.informationGain_(classes, [classesLeft,
                                                                  classesRight]
                                                        )

                if informationGain > maxInformationGain:
                    maxInformationGain = informationGain
                    splitFeat = feat
                    splitVal = split

        if maxInformationGain == 0:
            return None

        return [splitFeat, splitVal, maxInformationGain, len(indices)]

    def predict(self, X):
        '''
        Predicts class labels for a given input dataset

        Input: X as numpy array with (n, k) shape where n
                is the number of observations and k is the
                number of features
        Output: predictions as n x 2 numpy array --
                first column are the predicted class labels
                    corresponding to each row of X
                second column is the degree of certainty of the
                    prediction, defined as the percentage of the
                    labels in the leaf that match the most common
                    class
        '''
        preds = np.zeros([X.shape[0], 2])

        for i, x in enumerate(X):
            preds[i, :] = self.predictOne(x)

        return preds

    def predictOne(self, x):
        '''
        Predicts class label for one input row

        Input: x as a (1,k) numpy array where k is the number
            of features
        Output:
            predictions as a (2, ) numpy array
            first column is the predicted label
            second column is the degree of certainty of the
                prediction, defined as the percentage of the
                labels in the leaf that match the most common
                class
        '''

        currNode = self.root

        while not currNode.leaf:
            splitFeat = currNode.splitFeat
            splitVal = currNode.splitVal

            if x[splitFeat] <= splitVal:
                currNode = currNode.leftChild
            else:
                currNode = currNode.rightChild

        return np.array([currNode.maxClass, currNode.percentMax])

if __name__ == "__main__":
    x = np.arange(0, 1000) / 1000.
    x2 = x.copy()
    y = x.copy() + np.random.randn(1000) / 30
    y2 = x.copy() + 0.25 + np.random.randn(1000) / 30

    X = np.vstack([np.hstack([x, x2]), np.hstack([y, y2])]).T
    y = np.array([0] * 1000 + [1] * 1000)

    dt = DecisionTree()
    dt.fit(X, y)
