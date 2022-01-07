import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import cv2

image = cv2.imread("../images/park.jpg")
resized_img = cv2.resize(image, (0, 0), fx=0.04, fy=0.04)
# kernel = np.ones((3, 3), np.float32) / 3 ** 2
# resized_img = cv2.filter2D(resized_img, -1, kernel)
lol = cv2.resize(resized_img, (0, 0), fx=25, fy=25)
cv2.imwrite("lol.jpg", lol)
#
# def euclid_distance(x, xi):
#     return np.sqrt(np.sum((x - xi)**2))
#
# def neighbourhood_points(X, x_centroid, distance = 5):
#     eligible_X = []
#     for x in X:
#         distance_between = euclid_distance(x, x_centroid)
#         # print('Evaluating: [%s vs %s] yield dist=%.2f' % (x, x_centroid, distance_between))
#         if distance_between <= distance:
#             eligible_X.append(x)
#     return eligible_X
#
# def gaussian_kernel(distance, bandwidth):
#     val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((distance / bandwidth))**2)
#     return val

look_distance = 10 # How far to look for neighbours.
kernel_bandwidth = 6  # Kernel parameter.

# X = np.copy(resized_img)
# print('Initial X: ', X)

# past_X = []
# n_iterations = 20
# for it in range(n_iterations):
#     # print('Iteration [%d]' % (it))
#
#     for i, x in enumerate(X):
#         ### Step 1. For each datapoint x ∈ X, find the neighbouring points N(x) of x.
#         neighbours = neighbourhood_points(X, x, look_distance)
#         # print('[%s] has neighbours [%d]' % (x, len(neighbours)))
#
#         ### Step 2. For each datapoint x ∈ X, calculate the mean shift m(x).
#         numerator = 0
#         denominator = 0
#         for neighbour in neighbours:
#             distance = euclid_distance(neighbour, x)
#             weight = gaussian_kernel(distance, kernel_bandwidth)
#             numerator += (weight * neighbour)
#             denominator += weight
#
#         new_x = numerator / denominator
#
#         ### Step 3. For each datapoint x ∈ X, update x ← m(x).
#         X[i] = new_x
#
#     # print('New X: ', X)
#     past_X.append(np.copy(X))
#
# resized_img = cv2.resize(past_X[19], (0, 0), fx=25, fy=25)
# cv2.imwrite("nam.jpg", resized_img)

#
# X = np.array([[1, 2],
#               [1.5, 1.8],
#               [5, 8],
#               [8, 8],
#               [1, 0.6],
#               [9, 11],
#               [8, 2],
#               [10, 2],
#               [9, 3], ])

#
# colors = 10 * ["g", "r", "c", "b", "k"]
#

class Mean_Shift:
    def __init__(self, radius=7):
        self.radius = radius

    def fit(self, data):
        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]

        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    if np.linalg.norm(featureset - centroid) < self.radius:
                        in_bandwidth.append(featureset)

                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            prev_centroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break

            if optimized:
                break

        self.centroids = centroids

#
clf = Mean_Shift()
clf.fit(resized_img)
resized_img = cv2.resize(clf.centroids, (0, 0), fx=25, fy=25)
cv2.imwrite("nam.jpg", resized_img)
#
# centroids = clf.centroids
#
# plt.scatter(X[:, 0], X[:, 1], s=150)
#
# for c in centroids:
#     plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)
#
# plt.show()
