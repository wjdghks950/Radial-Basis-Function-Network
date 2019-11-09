import os
import numpy as np
import matplotlib.pyplot as plt

'''
K-Means Clustering Algorithm From the following URL:

    https://pythonmachinelearning.pro/using-neural-networks-for-regression-radial-basis-function-networks/
'''

def kMeans(x, k):
    clusters = np.random.choice(np.squeeze(x), size=k)
    prevClusters = clusters.copy()
    stds = np.zeros(k)
    converged = False

    iteration = 0
    while not converged:
        distances = np.squeeze(np.abs(x[:, np.newaxis] - clusters[np.newaxis, :])) # Calculate distance between all points and centroids
        closestCluster = np.argmin(distances, axis=1)
        for i in range(k):
            pointsForCluster = x[closestCluster == i]
            if len(pointsForCluster) > 0:
                clusters[i] = np.mean(pointsForCluster, axis=0) # Calculate new mean

        diff = np.linalg.norm(clusters - prevClusters)
        converged = diff < 1e-6
        prevClusters = clusters.copy()

    distances = np.squeeze(np.abs(x[:,np.newaxis] - clusters[np.newaxis, :]))
    closestCluster = np.argmin(distances, axis=1)

    clustersWithNoPoints = []
    for i in range(k):
        pointsForCluster = x[closestCluster == i]
        if len(pointsForCluster) < 2:
            clustersWithNoPoints.append(i)
            continue
        else:
            stds[i] = np.std(x[closestCluster == i])

    if len(clustersWithNoPoints) > 0:
        pointsToAverage = []
        for i in range(k):
            if i not in clustersWithNoPoints:
                pointsToAverage.append(x[closestCluster == i])
        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))

    return clusters, stds