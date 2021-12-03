import numpy as np
import cv2
from matplotlib import pyplot, pyplot as plt


def matchChannels(cdf1, cdf2, channel1):
    lut = np.zeros(256)  # lookuptable = lut
    vr = 0
    for v1 in range(256):
        for v2 in range(256):
            if cdf2[v2] >= cdf1[v1]:
                vr = v2
                break
        lut[v1] = vr

    return changePixels(channel1,lut)


def changePixels(ch, lut):
    result = np.zeros(ch.shape)
    for i in range(256):
        result[ch == i] = lut[i]
    return result


def getCdf(img, channel):
    hist, _ = np.histogram(img[:, :, channel], bins=256, range=[0, 256])
    cdf = np.cumsum(hist)
    return cdf / float(cdf.max())


def matchHistograms(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    matchedBlueChannel = matchChannels(getCdf(img1, 0), getCdf(img2, 0), img1[:, :, 0])
    matchedGreenChannel = matchChannels(getCdf(img1, 1), getCdf(img2, 1), img1[:, :, 1])
    matchedRedChannel = matchChannels(getCdf(img1, 2), getCdf(img2, 2), img1[:, :, 2])

    result = cv2.merge([matchedBlueChannel, matchedGreenChannel, matchedRedChannel])
    result = cv2.convertScaleAbs(result)

    resPdf, resBin = np.histogram(result, bins=256, range=[0, 256])
    img1Pdf, img1Bin = np.histogram(img2, bins=256, range=[0, 256])

    plt.plot(resBin[0:-1], resPdf, color="blue")
    plt.plot(img1Bin[0:-1], img1Pdf, color="green")
    plt.plot(resBin[0:-1], np.cumsum(resPdf), color="yellow")
    plt.plot(img1Bin[0:-1], np.cumsum(img1Pdf), color="red")
    plt.show()

    cv2.imwrite("../results/res11.jpg", result)
    cv2.imshow('image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


matchHistograms("../images/Dark.jpg", "../images/Pink.jpg")
