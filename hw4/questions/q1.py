import matplotlib.pyplot as plt
import numpy as np
import random

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


# implemented k-mean method here:

def calculate_distance(x1, y1, x2, y2):
    return np.power(x1 - x2, 2) + np.power(y1 - y2, 2)


def define_cluster(x1, y1):
    min_distance, cluster = None, 0
    for i in range(k):
        distance = calculate_distance(x1, y1, clusters_x[i], clusters_y[i])
        if min_distance is None or distance < min_distance:
            min_distance, cluster = distance, i

    return cluster


# first choose k random points:
k = 10
clusters_index = random.sample(range(0, n), k)
clusters_x, clusters_y = x[clusters_index], y[clusters_index]
for iteration in range(3):
    clusters = np.ndarray((n, 1), dtype=int)
    for i in range(n):
        clusters[i] = define_cluster(x[i], y[i])

    for cluster_num in range(k):
        xx = x[clusters == cluster_num]
        yy = y[clusters == cluster_num]
        clusters_x[cluster_num] = np.mean(xx, axis=0)
        clusters_y[cluster_num] = np.mean(yy, axis=0)
        plt.plot(xx, yy, '.')
    plt.show()
plt.savefig("../results/res02.jpg")

