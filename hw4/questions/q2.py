import cv2
import numpy as np

image = cv2.imread("../images/lol.jpg")
scale =1
resized_img = cv2.resize(image, (0, 0), fx=scale, fy=scale)
# this code works for small images
def mean_shift(img):
    centroids = img.copy()
    for iteration in range(3):
        for centroid in centroids:
            labels = np.full((img.shape[0], img.shape[1]), -11)
            neighbors = []
            for x in range(img.shape[1]):
                for y in range(img.shape[0]):
                    pxl = img[y, x, :]
                    if np.linalg.norm(pxl - centroid) < 10:
                        neighbors.append(pxl)
                        labels[y, x] = 1
            if len(neighbors) > 0:
                new_centroid = np.average(neighbors, axis=0)
                img[labels == 1] = new_centroid

        centroids = np.unique(img)


    resized_img = cv2.resize(img, (0, 0), fx=1 / scale, fy=1 / scale)
    cv2.imwrite("nam.jpg", resized_img)

