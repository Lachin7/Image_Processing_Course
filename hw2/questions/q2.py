import numpy as np
import cv2


def templateMatching(img, temp):
    height, width = img.shape
    h, w = temp.shape

    img = np.array(img, dtype="float")
    temp = np.array(temp, dtype="float")

    ncc = np.zeros((height - h, width - w))
    temp = temp - np.mean(temp)

    for y in range(0, height - h):
        for x in range(0, width - w):

            f = img[y: y + h, x: x + w]
            f = f - np.mean(f)

            correlation = np.sum(f * temp)
            norm = np.sqrt((np.sum(f ** 2))) * np.sqrt(np.sum(temp ** 2))

            if norm == 0:
                ncc[y, x] = 0
            else:
                ncc[y, x] = correlation / norm
    return ncc


def detectDistinctOnes(locx, locy):
    distinctOnes = [locx[0]]
    indices = [0]
    rx, ry = [], []
    for i in range(1, len(locx)):
        if locx[i] - distinctOnes[-1] > 2:
            distinctOnes.append(locx[i])
            indices.append(i)
    indices.append(len(locx) - 1)
    for i in range(1, len(indices)):
        print(indices[i])
        rx.append(locx[int((indices[i] + indices[i - 1]) / 2)])
        ry.append(int(np.average(locy[indices[i - 1]: indices[i]])))
    return rx, ry


image = cv2.imread('../images/Greek-ship.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray = cv2.resize(gray, None, fx=0.125, fy=0.125)

template = cv2.imread('../images/patch.png', 0)
template = cv2.resize(template, None, fx=0.125, fy=0.125)

h, w = template.shape
ncc = templateMatching(gray, template)

threshold = 0.38
(locY, locX) = np.where(ncc >= threshold)
sortedIndices = locX.argsort()
locX = locX[sortedIndices[::-1]]
locX = np.flip(locX)
locY = locY[sortedIndices[::-1]]
locY = np.flip(locY)
locX, locY = detectDistinctOnes(locX, locY)

for pt in zip(*(locY, locX)[::-1]):
    cv2.rectangle(image, (pt[0] * 8, pt[1] * 8), (pt[0] * 8 + w * 8, pt[1] * 8 + h * 8), (0, 0, 255), 2)

cv2.imwrite('../results/res15.jpg', image)
