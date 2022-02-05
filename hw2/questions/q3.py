import numpy as np
import cv2


def makeArray(x1, y1, x2, y2, x3, y3):
    height = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    width = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    return np.zeros((int(width), int(height), 3)), width, height


def warp(h, src, dest):
    dh, dw, dc = dest.shape
    hInv = np.linalg.inv(h)
    for j in range(dh):
        for i in range(dw):
            v = np.array([[i], [j], [1]])
            # [[l], [k], [t]] = np.matmul(hInv, v)
            [[l], [k], [t]] = hInv @ v
            l, k = l / t, k / t
            # if 0 <= l < (sh - 1) and 0 <= k < (sw - 1):
            y, x = np.floor(l).astype(int), np.floor(k).astype(int)
            b, a = l - y, k - x
            x1, y1 = x + 1, y + 1
            res = (1 - b) * (1 - a) * src[x, y] + a * (1 - b) * src[x1, y] + b * (1 - a) * src[x, y1] + a * b * src[
                x1, y1]
            dest[j, i] = res
    return dest


books = cv2.imread('../images/books.jpg')


def findTheBook(x1, y1, x2, y2, x3, y3, x4, y4, name):
    book, width, height = makeArray(x1, y1, x2, y2, x4, y4)
    srcPoints = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=float)
    desPoints = np.array([[0, 0], [height - 1, 0], [0, width - 1], [height - 1, width - 1]], dtype=float)
    h, status = cv2.findHomography(srcPoints, desPoints)
    print(h)
    result = warp(h, books, book)
    cv2.imwrite(name + '.jpg', result)


findTheBook(666, 215, 601, 395, 384, 113, 320, 289, '../results/res16')
findTheBook(358, 741, 158, 710, 408, 472, 208, 431, '../results/res17')
findTheBook(809, 972, 610, 1102, 621, 675, 424, 801, '../results/res18')
