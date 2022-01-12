import math

import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries


class Center:
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label


image = cv2.imread("../images/slic.jpg")
scale = 0.125
image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
height, width, _ = image.shape
image_rgb = image.copy()
image = cv2.medianBlur(image, 5)
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)


def generate_gradient():
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    edges_x = cv2.filter2D(image, cv2.CV_64F, kernelx)
    edges_y = cv2.filter2D(image, cv2.CV_64F, kernely)
    return np.mean(np.sqrt(edges_x ** 2 + edges_y ** 2), axis=2)


def generate_initial_centers():
    centers_indices_x = np.arange(start=0, stop=width, step=s)
    centers_indices_y = np.arange(start=0, stop=height, step=s)

    centers = []
    label = 0
    for y in centers_indices_y:
        for x in centers_indices_x:
            min_val, center = math.inf, Center(x, y, label)
            for j in range(max(0, y - 5), min(height, y + 6)):
                for i in range(max(0, x - 5), min(width, x + 6)):
                    if gradient[j, i] < min_val:
                        min_val, center = gradient[j, i], Center(i, j, label)
            centers.append(center)
            label += 1

    return centers


def distance(xk, xn, yk, yn, img):
    d_lab = np.linalg.norm(img[yk, xk, :] - img[yn, xn, :])
    d_xy = (xk - xn) ** 2 + (yk - yn) ** 2
    return d_lab + alpha * d_xy


def assign_centers():
    # in the third dimension, the first one stores the min value of the match found and the second one labels it
    matches, labels = np.full((height, width), np.inf), np.full((height, width), -11)
    for center in centers:
        y, x = center.y, center.x
        y_min, y_max, x_min, x_max = max(0, y - s), min(height, y + s + 1), max(0, x - s), min(width, x + s + 1)
        for j in range(y_min, y_max):
            for i in range(x_min, x_max):
                dist = distance(i, x, j, y, image)
                if matches[j, i] > dist:
                    matches[j, i] = dist
                    labels[j, i] = center.label
    return labels


def generate_new_centers():
    for center in centers:
        cluster = np.argwhere(labels == center.label)
        if len(cluster) > 0:
            center.x, center.y = int(np.mean(cluster[:, 1])), int(np.mean(cluster[:, 0]))


# defining the constants:
k = 2048
s = math.floor(math.sqrt((height * width) / k))
alpha = 0.05

gradient = generate_gradient()
centers = generate_initial_centers()
for iteration in range(5):
    labels = assign_centers()
    generate_new_centers()
    if iteration == 4:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        image = mark_boundaries(image_rgb, labels, color=(0, 0, 0))
        image = cv2.resize(image, (0, 0), fx=1 / scale, fy=1 / scale)
        plt.imsave('../results/res09.jpg', image)

