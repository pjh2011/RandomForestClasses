import numpy as np
from decisionTreeClass import DecisionTree
from collections import Counter, defaultdict
import time
from plottingFunctions import plotDecisionBoundary


class RandomForest(object):

    def __init__(self, nTrees=10, maxDepth=None, minIG=None, nPointsLeaf=2,
                 nFeaturesToSplit='sqrt'):
        self.nTrees = nTrees
        self.maxDepth = maxDepth
        self.minIG = minIG
        self.nPointsLeaf = nPointsLeaf
        self.nFeaturesToSplit = nFeaturesToSplit

    def fit(self, XTrain, yTrain):
        self.XTrain = XTrain
        self.yTrain = yTrain

        self.nFeatures = self.XTrain.shape[1]

        if self.nFeaturesToSplit == 'sqrt':
            self.nFeaturesToSplit = int(np.sqrt(self.nFeatures))

        self.trees = []
        self.predsOOB = defaultdict(list)

        for i in xrange(self.nTrees):
            print i
            tree = DecisionTree(nFeaturesToSplit=self.nFeaturesToSplit)
            indicesSample, indicesOOB = self.bootstrapSample()

            t = time.time()
            tree.fit(self.XTrain[indicesSample, :], self.yTrain[indicesSample])
            print time.time() - t

            self.predictOOB_(indicesOOB, tree)
            self.trees.append(tree)

    def bootstrapSample(self):
        '''
        Returans a tuple with two numpy arrays:
            first element: the indices for a bootstrapped sample
            second element: the indices of the out of bag points
        '''
        self.nObs = self.XTrain.shape[0]
        indicesAll = np.arange(self.nObs)

        indicesSample = np.random.choice(indicesAll, self.nObs, replace=True)

        mask = np.zeros(self.XTrain.shape[0], dtype=bool)
        mask[np.unique(indicesSample)] = True

        indicesOOB = indicesAll[~mask]

        return (indicesSample, indicesOOB)

    def predictOne(self, x):
        treePreds = np.zeros(self.nTrees)

        for i in xrange(self.nTrees):
            treePreds[i] = self.trees[i].predictOne(x)[0]

        votes = Counter(treePreds)

        return votes.most_common(1)[0][0]

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

    def predictOOB_(self, indicesOOB, tree):
        XOOB = self.XTrain[indicesOOB, :]

        predsOOB = tree.predict(XOOB)[:, 0]

        for i in xrange(len(indicesOOB)):
            self.predsOOB[indicesOOB[i]].append(predsOOB[i])

    def calcOOBError(self):
        pass

# random forest advantages:
# oob error
# adding more estimators doesn't cause overfitting
# variable importance

# downside: want balanced classes since the optimization
# criteria is accuracy

if __name__ == "__main__":
    x = np.arange(0, 1000) / 1000.
    x2 = x.copy()
    y = x.copy() + np.random.randn(1000) / 30
    y2 = x.copy() + 0.25 + np.random.randn(1000) / 30

    X = np.vstack([np.hstack([x, x2]), np.hstack([y, y2])]).T
    y = np.array([0] * 1000 + [1] * 1000)

    rf = RandomForest()
    rf.fit(X, y)

    plotDecisionBoundary(X, y, rf)
