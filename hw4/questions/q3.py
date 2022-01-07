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
image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

# defining the constants:
k = 64
s = math.floor(math.sqrt((height * width) / k))
alpha = 0


def generate_gradient():
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    edges_x = cv2.filter2D(image, cv2.CV_8U, kernelx)
    edges_y = cv2.filter2D(image, cv2.CV_8U, kernely)
    return np.mean(np.sqrt(edges_x ** 2 + edges_y ** 2), axis=2)

    # cv2.imshow('Gradients_X 1', edges_x)
    # cv2.imshow('Gradients_Y 1', edges_y)
    # sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    # sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # cv2.imshow('Gradients_X 2', sobel_x)
    # cv2.imshow('Gradients_Y 2', sobel_y)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def generate_initial_centers():
    centers_indices_x = np.arange(start=0, stop=width, step=s)
    centers_indices_y = np.arange(start=0, stop=height, step=s)

    centers = []
    label = 0
    for y in centers_indices_y:
        for x in centers_indices_x:
            min_val, center = math.inf, Center(x, y, label)
            for j in range(max(0, y - 2), min(height, y + 3)):
                for i in range(max(0, x - 2), min(width, x + 3)):
                    if gradient[j, i] < min_val:
                        min_val, center = gradient[j, i], Center(i, j, label)
            centers.append(center)
            label += 1

    return centers


def distance(xk, xn, yk, yn, img):
    d_lab = np.linalg.norm(img[yk, xk, :] - img[yn, xn, :])
    # d_lab = (img[yk, xk, 0] - img[yn, xn, 0]) ** 2 + (img[yk, xk, 1] - img[yn, xn, 1]) ** 2 + (
    #             img[yk, xk, 2] - img[yn, xn, 2]) ** 2
    d_xy = (xk - xn) ** 2 + (yk - yn) ** 2
    return d_lab + alpha * d_xy


def assign_centers():
    # in the third dimension, the first one stores the min value of the match found and the second one labels it
    matches, labels = np.full((height, width), np.inf), np.full((height, width), -11)
    for center in centers:
        y, x = center.y, center.x
        y_min, y_max, x_min, x_max = max(0, y - s), min(height, y + s + 1), max(0, x - s), min(width, x + s + 1)
        # d_lab = np.linalg.norm(image_lab[y_min:y_max, x_min:x_max, :] - image_lab[y, x, :], axis=2)
        # m = np.arange(y_min, y_max).reshape((-1, 1))
        # m1 = m * np.ones((y_max - y_min, x_max - x_min))
        # n = np.arange(x_min, x_max).reshape((-1, 1))
        # n1 = n * np.ones((y_max - y_min, x_max - x_min))
        # d_xy = np.linalg.norm(np.ndarray([y, x]) - np.stack([m1, n1], axis=2), axis=2)
        #
        # D = d_xy + d_lab * alpha
        # condition = D < matches[y_min:y_max, x_min:x_max]
        # mask = np.where(condition)
        # labels[mask], matches[mask] = center.label, D[mask]
        for j in range(y_min, y_max):
            for i in range(x_min, x_max):
                dist = distance(i, x, j, y, image_lab)
                if matches[j, i] > dist:
                    matches[j, i] = dist
                    labels[j, i] = center.label
        # centers.append(center)
    return labels


def generate_new_centers():
    for center in centers:
        cluster = np.argwhere(labels == center.label)
        center.x, center.y = int(np.mean(cluster[:, 0])), int(np.mean(cluster[:, 1]))
    # for count in range(len(centers)):
    #     cluster = image[labels == count]
    #     if len(cluster) > 0:
    #         tmp = np.mean(cluster, axis=0)
    #         centers[count] = tmp
        # mean = np.mean(cluster, axis=0)
        # center = mean


gradient = generate_gradient()
centers = generate_initial_centers()
for iteration in range(4):
    labels = assign_centers()
    generate_new_centers()
    if iteration == 3:
        res = image.copy()

        for c in range(len(centers)):
            res[labels == c] = centers[c]

        res = res[:, :, 2:].astype(np.uint8)
        res = cv2.cvtColor(res, cv2.COLOR_Lab2RGB)

        plt.imsave('res08.jpg', mark_boundaries(image, labels.astype(int)))
