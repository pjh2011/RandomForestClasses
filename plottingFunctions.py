import numpy as np
import matplotlib.pyplot as plt


def plotDecisionBoundary(X, y, predictor, nPoints=100):
    '''
    Takes in a set of 2-d observations X with labels y, and
        an instance machine learning algorithm class that has a
        predict() method. Creates a grid of points and uses
        the given machine learning algorithm to predict
        labels on that grid. Plots the grid predictions as a plot
        background of decision "zones", then overlays the true
        data X, with labels y. Labels are denoted by colors.
    Inputs  : X as  as an (n, 2) numpy array with n=number of
                observations and columns as (x, y) to plot
            : y as (n,) numpy array with each row being the
                label for the corresponding row of X
            : predictor as an object of a class with a predict
                method, which takes in a (2,) numpy array and
                predicts a class label
            : nPoints as an int specifying the number of points
                along each axis used to plot decision zones
    Output  : none

    '''
    # evenly sampled points
    xMin, xMax = X[:, 0].min(), X[:, 0].max()
    yMin, yMax = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(xMin, xMax, nPoints),
                         np.linspace(yMin, yMax, nPoints))
    plt.xlim(xMin, xMax)
    plt.ylim(yMin, yMax)

    # plot background colors
    ax = plt.gca()
    Z = predictor.predict(np.vstack([xx.ravel(), yy.ravel()]).T)
    Z = Z[:, 0]
    Z = Z.reshape(xx.shape)
    cs = ax.contourf(xx, yy, Z, cmap='RdBu', alpha=0.25)
    cs2 = ax.contour(xx, yy, Z, cmap='RdBu', alpha=0.25)

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.get_cmap('RdBu'), alpha=0.75)
    plt.show()
