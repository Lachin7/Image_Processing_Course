import numpy as np
import cv2


def logTransformation(image, alpha):
    c = 255 / np.log(1 + alpha * 255)
    result = (c * (np.log(1 + alpha * image))).astype('uint8')
    return result


path = "../images/Enhance2.JPG"
img = cv2.imread(path)
logTransformed = logTransformation(img, 0.04)
cv2.imwrite("../results/res02.jpg", logTransformed)
cv2.imshow('image', logTransformed)
cv2.waitKey(0)
cv2.destroyAllWindows()

# def logTransformation(image, alpha):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
#     c = 255 / np.log(1 + alpha * 255)
#     h = (c * (np.log(1 + alpha * h))).astype('uint8')
#     hsv[:, :, 0] = h
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     # return result
