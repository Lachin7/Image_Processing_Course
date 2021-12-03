import numpy as np
import cv2
import time


def cv2Filter(k, img):
    n = 2 * k + 1
    height, width, _ = img.shape
    t0 = time.time()
    kernel = np.ones((n, n), np.float32) / n ** 2
    img = cv2.filter2D(img, -1, kernel)
    img = img[k: height - k, k: width - k, :]
    t1 = time.time()
    cv2.imwrite("res07.jpg", img)
    return (t1 - t0), img


def dumbFilter(k, img):
    n = 2 * k + 1
    height, width, channels = img.shape
    t0 = time.time()
    result = np.zeros((height, width, 3))
    print(result.shape)
    for channel in range(3):
        for i in range(k, height - k - 1):
            for j in range(k, width - k - 1):
                result[i, j, channel] = np.sum(img[i - k:i + k + 1, j - k:j + k + 1, channel]) / n ** 2

    t1 = time.time()
    result = result[k: height - k, k: width - k, :]
    print(result.shape)
    cv2.imwrite("res08.jpg", result)
    return (t1 - t0), result


def improvedFilter(k, img):
    n = 2 * k + 1
    height, width, _ = img.shape
    t0 = time.time()
    result = []
    for channel in range(3):
        sumArr = np.zeros((height - 2 * k, width - 2 * k))
        for i in range(n):
            for j in range(n):
                sumArr += img[i:height - (2 * k - i), j:width - (2 * k - j), 2 - channel] / n ** 2
        sumArr = sumArr.astype(np.uint8)
        result.insert(0, sumArr)

    blurredImage = np.dstack(result)
    t1 = time.time()
    cv2.imwrite("../results/res09.jpg", blurredImage)
    return (t1 - t0), blurredImage


image = cv2.imread("../images/Pink.jpg")
t1, img1 = cv2Filter(2, image)
t2, img2 = dumbFilter(2, image)
t3, img3 = improvedFilter(2, image)
print('cv2 time: ' + str(t1) + 'dumb filter time: ' + str(t2) + 'improved filter time: ' + str(t3))

# cv2.imshow('image', dumbFilter(4, image))

# def sumOf(img, i, j, k, channel):
#     result = 0
#     for x in range(-k, k + 1):
#         for y in range(-k, k + 1):
#             result += img[i + k, j + k, channel]
#
#     return result
