import numpy as np
import cv2


def logTransformation(image, alpha):
    c = 255 / np.log(1 + alpha * 255)
    result = (c * (np.log(1 + alpha * image))).astype('uint8')
    return result


def equalizeHistogram(image):
    b, g, r = cv2.split(image)
    height, width, channels = image.shape

    B_hist, _ = np.histogram(image[:, :, 0], bins=256, range=[0, 256])
    G_hist, _ = np.histogram(image[:, :, 1], bins=256, range=[0, 256])
    R_hist, _ = np.histogram(image[:, :, 2], bins=256, range=[0, 256])

    B_cumSum = (np.cumsum(B_hist) * 255 / (height * width)).astype('uint8')
    G_cumSum = (np.cumsum(G_hist) * 255 / (height * width)).astype('uint8')
    R_cumSum = (np.cumsum(R_hist) * 255 / (height * width)).astype('uint8')
    print(B_cumSum.shape)

    return cv2.merge((B_cumSum[b], G_cumSum[g], R_cumSum[r]))


path = "../images/Enhance1.JPG"
img = cv2.imread(path)
logTransformed = logTransformation(img, 0.15)
equalized = equalizeHistogram(logTransformed)
cv2.imwrite("../results/res01.jpg", equalized)
cv2.imshow('image', equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
