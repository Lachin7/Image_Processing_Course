import math

import matplotlib.pyplot as plt
import numpy as np
import random


def readPoints():
    # read the points from file:
    with open("../images/Points.txt", "r") as file:
        n = int(file.readline())
        x, y = np.zeros((n, 1), dtype=float), np.zeros((n, 1), dtype=float)
        lines = file.readlines()[0:]
        for i in range(len(lines)):
            values = lines[i].split()
            x[i], y[i] = float(values[0]), float(values[1])

    file.close()
    plt.plot(x, y, 'r.')
    plt.savefig("../results/res01.jpg")
    return x, y, n


# implementing k-mean method here:

def calculate_distance(x1, y1, x2, y2):
    return np.power(x1 - x2, 2) + np.power(y1 - y2, 2)


def assign_cluster(x1, y1, k, clusters_x, clusters_y):
    min_distance, cluster = None, 0
    for i in range(k):
        distance = calculate_distance(x1, y1, clusters_x[i], clusters_y[i])
        if min_distance is None or distance < min_distance:
            min_distance, cluster = distance, i

    return cluster


def convert_to_scalar(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


def K_means(x, y, n, img_num, convert=False):
    # first choose k random points:
    k = 2
    centers_change = True
    clusters_index = random.sample(range(0, n), k)
    clusters_x, clusters_y = x[clusters_index], y[clusters_index]
    # then iterate until the centers get fixed
    while centers_change:
        clusters = np.ndarray((n, 1), dtype=int)
        for i in range(n):
            clusters[i] = assign_cluster(x[i], y[i], k, clusters_x, clusters_y)
        prev_centers_x, prev_centers_y = clusters_x.copy(), clusters_y.copy()
        if np.array_equal(prev_centers_x, clusters_x) and np.array_equal(prev_centers_y, clusters_y):
            centers_change = False
            # check if it was in the polar space
            if convert:
                x, y = convert_to_scalar(x, y)
            for cluster_num in range(k):
                xx = x[clusters == cluster_num]
                yy = y[clusters == cluster_num]
                clusters_x[cluster_num] = np.mean(xx, axis=0)
                clusters_y[cluster_num] = np.mean(yy, axis=0)
                plt.plot(xx, yy, '.')
        plt.savefig("../results/res0" + str(img_num) + ".jpg")
        plt.show()


x, y, n = readPoints()
K_means(x, y, n, 2)
K_means(x, y, n, 3)


r, theta = np.sqrt(np.power(x, 2) + np.power(y, 2)), np.arctan2(y, x)
K_means(r, theta, n, 4, convert=True)
