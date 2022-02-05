import numpy as np
import cv2


def separateChannels(img):
    height = int(img.shape[0] / 3)

    B = img[:height]
    G = img[height:2 * height]
    R = img[2 * height:3 * height]

    return np.dstack([B, G, R])


def computeOffset(ch1, ch2, offset, i):
    ch1, ch2 = shift(ch1, ch2, offset)
    ch1, ch2 = cv2.resize(ch1, (0, 0), fx=0.5 ** i, fy=0.5 ** i), cv2.resize(ch2, (0, 0), fx=0.5 ** i, fy=0.5 ** i)
    offset = (2**i) * align(ch1, ch2)
    return offset


def colorize(img):
    img = separateChannels(img).astype(np.int16)
    b, g, r = cv2.split(img)
    height, width, _ = img.shape
    # we fix the channel g and align channels b and r with it:
    offset1, offset2 = np.zeros(2, dtype=np.int16), np.zeros(2, dtype=np.int16)
    depth = int(np.log2(width / 100))
    # image pyramid here:
    for i in range(depth, -1, -1):
        offset1 += computeOffset(g, b, offset1, i)
        offset2 += computeOffset(g, r, offset2, i)
        print("the offset needed for b to move: " + str(offset1) + "  the offset needed for r to move: " + str(
            offset2) + " depth: " + str(i))
        xb, yb = offset1
        xr, yr = offset2
        b = np.roll(b, (xb, yb), axis=(1, 0))
        r = np.roll(r, (xr, yr), axis=(1, 0))

        img = np.dstack((b, g, r)).astype("uint8")
        img = cropEdges(img)
        cv2.imwrite("../results/q3_subResults/train" + str(i) + ".jpg", img)
        cv2.imshow(str(i), img)


def cropEdges(img):
    height, width, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    edge1, edge2, edge3, edge4 = 0, 0, 0, 0
    for i in range(int(height / 8)):
        if (np.std(h[i, :])) < 37:
            edge1 = i
        if (np.std(h[height - i - 1, :])) < 37:
            edge2 = height - i - 1
    img = img[edge1: edge2, :, :]

    for i in range(int(width / 8)):
        if (np.std(h[:, i])) < 38:
            edge3 = i
        if (np.std(h[:, width - i - 1])) < 38:
            edge4 = width - i - 1
    img = img[:, edge3: edge4, :]
    return img


def calculateSSD(shifted1, shifted2):
    return np.average((shifted1 - shifted2) ** 2)


def shift(ch1, ch2, offset):
    x, y = offset
    height, width = ch2.shape
    return ch1[max(y, 0): height + min(y, 0), max(x, 0): width + min(x, 0)], \
           ch2[-min(y, 0): height - max(y, 0), -min(x, 0): width - max(x, 0)]


def align(channel1, channel2):
    opt_offset = np.zeros(2, dtype=np.int16)
    minValue = None
    for x in range(-7, 7):
        for y in range(-7, 7):
            s, m = shift(channel1, channel2, (x, y))
            ssd = calculateSSD(np.asarray(s, dtype="int16"), np.asarray(m, dtype="int16"))
            if minValue is None or ssd < minValue:
                minValue = ssd
                opt_offset[0], opt_offset[1] = x, y
    return opt_offset


""" your image path here : """
colorize(cv2.imread("../images/train.tif", 0))
cv2.waitKey(0)
cv2.destroyAllWindows()

""" other different approaches to separate the channels: """
# def separateChannels(img):
#     # height, width = img.shape
#     # # h_wwm is height without white margin
#     # h_wwm = height - height * 0.05
#     # h_each = int(h_wwm / 3)
#     # # w_wwm is width without white margin
#     # w_wwm = width - width * 0.05
#     #
#     # # w0 is starting width index
#     # w0 = int(width * 0.025)
#     # # wn is the ending width index
#     # wn = int(w_wwm + width * 0.025)
#     # # e is the average whitespace on the top
#     # e = int(height * 0.025)
#     #
#     # b_channel = img[e: h_each + e, w0: wn]
#     # g_channel = img[h_each + e: h_each * 2 + e, w0: wn]
#     # r_channel = img[h_each * 2 + e: h_each * 3 + e, w0: wn]
#
#     stacked = np.dstack((b_channel, g_channel, r_channel))
#     return stacked

# def separate(img):
#     height, width = img.shape
#     i, j, k, l = 0, width-1, 0, height-1
#     rows = len(img)
#     cols = len(img[0])
#
#     while img[int(height/2), i] > 240:
#         i += 1
#     while img[int(height/2), j] > 240:
#         j -= 1
#     while img[k, int(width/2)] > 240:
#         k += 1
#     while img[l, int(width/2)] > 240:
#         l -= 1
#
#     img = img[k:l, i:j]
#     height, width = img.shape
#     i, j, k, l = 0, width - 1, 0, height - 1
#
#     while img[ int(height/2), i] < 11:
#         i += 1
#     while img[int(height/2), j] < 11:
#         j -= 1
#     while img[k,int(width/2)] < 11:
#         k += 1
#     while img[l,int(width/2)] < 11:
#         l -= 1
#
#     img = img[k:l, i:j]
#
#     height = int(height / 3)
#
#     B = img[:height]
#     G = img[height:2 * height]
#     R = img[2 * height:3 * height]
#
#     return np.dstack([B, G, R])
#
# paths = ["../images/emir.tif", "../images/train.tif", "../images/mosque.tif"]
# for i in range(3):
#     image = cv2.imread(paths[i], 0)
#     colorize(image)
