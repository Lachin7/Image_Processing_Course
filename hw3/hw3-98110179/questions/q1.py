import math

import numpy as np
import cv2
from matplotlib import pyplot as plt

def detectDistinctOnesReverse(B400, A, B, distance):
    distinctOnes = [B400[0]]
    indices = [0]
    ra, rb = [], []
    for i in range(1, len(B400)):
        if B400[i] - distinctOnes[-1] > distance:
            distinctOnes.append(B400[i])
            indices.append(i)
    indices.append(len(B400) - 1)
    for i in range(1, len(indices)):
        # if i == 1:
        #     ra.append(B[indices[i+1]-1])
        # else:
            ra.append(B[int((indices[i] + indices[i - 1]) / 2)])
            rb.append(A[indices[i]])
    return np.asarray(ra), np.asarray(rb)
        # rb.append(int(np.average(locy[indices[i - 1]: indices[i]])))
        # if i == len(indices) -1:
        #     ra.append(B[len(B) - 1])
        #     rb.append(A[len(A) - 1])
        # else:
        #     ra.append(B[indices[i+1]-1])
        #     rb.append(A[indices[i+1]-1])

def detectDistinctOnes(B400, A, B, distance):
    distinctOnes = [B400[0]]
    indices = [0]
    ra, rb = [], []
    for i in range(1, len(B400)):
        if B400[i] - distinctOnes[-1] > distance:
            distinctOnes.append(B400[i])
            indices.append(i)
    indices.append(len(B400) - 1)
    for i in range(1, len(indices)):
        ra.append(B[int((indices[i] + indices[i - 1]) / 2)])
        rb.append(A[int((indices[i] + indices[i - 1]) / 2)])
        # if i == len(indices) -1:
        #     ra.append(B[len(B) - 1])
        #     rb.append(A[len(A) - 1])
        # else:
        #     ra.append(B[indices[i+1]-1])
        #     rb.append(A[indices[i+1]-1])


        # if i == len(indices) -1:
        #     rb.append(A[len(A) - 1])
        # else:
        #     rb.append(A[indices[i+1]-1])
        # if i == len(indices) - 1:
        #    rb.append(np.average(A[indices[i]:]))
        # else:
        #     # rb.append(np.average(A[indices[i]: indices[i + 1]]))


    return np.asarray(ra), np.asarray(rb)


def detectLines(img, i):
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    kernel = np.ones((3, 3), np.float32) / 3 ** 2
    blurred_img = cv2.filter2D(img, -1, kernel)
    edges = cv2.Canny(image=blurred_img, threshold1=340, threshold2=350)
    cv2.imwrite("../results/res0" + str(i) + ".jpg", edges)
    plt.imshow(edges)
    plt.show()

    height, width = edges.shape
    theta_size, rho_size = 180, int(np.ceil(np.sqrt(height ** 2 + width ** 2)))
    Acc = np.zeros((rho_size, theta_size))
    for (y, x) in list(zip(*np.where(edges == 255))):
        for t in range(0, theta_size):
            xt, yt = x, y
            cos = np.cos(np.deg2rad(2 * t))
            sin = np.sin(np.deg2rad(2 * t))
            rho = int(xt * cos + yt * sin)
            Acc[rho, t] += 1

    cv2.imwrite("../results/res0" + str(i + 2) + "-hough-space.jpg", Acc)
    plt.imshow(Acc)
    threshold, cons = 50, 2
    (R, T) = np.where(Acc > threshold)
    C, S = np.cos(np.deg2rad(2 * T)), np.sin(np.deg2rad(2 * T))
    B, A = np.asarray(R / S, dtype=np.float16), np.asarray(-C / S, dtype=np.float16)

    for (b, a) in list(zip(*(B, A))):
            x0, y0, x1, y1 = int(50), int(a * 50 + b), int(700), int(a * 700 + b)
            cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
    cv2.imwrite("../results/res0" + str(i + 4) + "-lines.jpg", img)


    B400 = A*300 + B
    sortedIndices = B400.argsort()
    # sortedIndices = sortedIndices[::-1]
    B4001 = np.flip(B400[sortedIndices[::-1]])
    A = np.flip(A[sortedIndices[::-1]])
    B = np.flip(B[sortedIndices[::-1]])
    B, A = detectDistinctOnes(B4001, A, B, 18)
    for (b, a) in list(zip(*(B, A))):
        if 0.5 < a or a < -0.5:
            x0, y0, x1, y1 = int(50), int(a * 50 + b), int(700), int(a * 700 + b)
            cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1)

    res = cv2.resize(img, None, fx=2, fy=2)
    cv2.imwrite("../results/res0" + str(i + 6) + "-chess.jpg", res)

    for (b, a) in list(zip(*(B, A))):
        if 0.5 < a or a < -0.5:
            for (b, a) in list(zip(*(B, A))):
                if 0.5 < a or a < -0.5:
                    x0, y0, x1, y1 = int(50), int(a * 50 + b), int(700), int(a * 700 + b)
                    cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
            x0, y0, x1, y1 = int(50), int(a * 50 + b), int(700), int(a * 700 + b)
            cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
    # B1, A1, B2, A2 = [], [], [], []
    # for (b, a) in list(zip(*(B, A))):
    #     if a > 0.55:
    #         A1.append(a)
    #         B1.append(b)
    #     elif a < -0.55:
    #         A2.append(a)
    #         B2.append(b)
    # B1, A1, B2, A2 = np.asarray(B1), np.asarray(A1), np.asarray(B2), np.asarray(A2),
    #
    # B4001 = A1*300 + B1
    # sortedIndices = B4001.argsort()
    # # sortedIndices = sortedIndices[::-1]
    # B4001 = np.flip(B4001[sortedIndices[::-1]])
    # A1 = np.flip(A1[sortedIndices[::-1]])
    # B1 = np.flip(B1[sortedIndices[::-1]])
    # B1, A1 = detectDistinctOnes(B4001, A1, B1, 20)
    # for (b, a) in list(zip(*(B1, A1))):
    #     if 0.5 < a or a < -0.5:
    #         x0, y0, x1, y1 = int(50), int(a * 50 + b), int(700), int(a * 700 + b)
    #         cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
    #
    # B4001 = A1 * 300 + B1
    # sortedIndices = B4001.argsort()
    #     # sortedIndices = sortedIndices[::-1]
    # B4001 = np.flip(B4001[sortedIndices[::-1]])
    # A1 = np.flip(A1[sortedIndices[::-1]])
    # B1 = np.flip(B1[sortedIndices[::-1]])
    # B1, A1 = detectDistinctOnes(B4001, A1, B1, 20)
    # for (b, a) in list(zip(*(B1, A1))):
    #     if 0.5 < a or a < -0.5:
    #         x0, y0, x1, y1 = int(50), int(a * 50 + b), int(700), int(a * 700 + b)
    #         cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
    #
    # B4002 = A2 * 300 + B2
    # sortedIndices = B4002.argsort()
    #     # sortedIndices = sortedIndices[::-1]
    # B4002 = np.flip(B4002[sortedIndices[::-1]])
    # A2 = np.flip(A2[sortedIndices[::-1]])
    # B2 = np.flip(B2[sortedIndices[::-1]])
    # B2, A2 = detectDistinctOnesReverse(B4002, A2, B2, 15)
    # for (b, a) in list(zip(*(B2, A2))):
    #     if 0.5 < a or a < -0.5:
    #         x0, y0, x1, y1 = int(50), int(a * 50 + b), int(700), int(a * 700 + b)
    #         cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
        # cr, sr = cos * rho, sin * rho
        # x0, y0, x1, y1 = int(cr - 1000 * sin), int(sr + 1000 * cos), int(cr + 1000 * sin), int(sr - 1000 * cos)
        # cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
        # x1, y1, x2, y2, cons = B[rho, t, 0], B[rho, t, 1], B[rho, t, 2], B[rho, t, 3], 2
        # a = np.cos(np.deg2rad(2*t))
        # b = np.sin(np.deg2rad(t*2))
        # x0 = a * rho
        # y0 = b * rho
        # x1 = int(x0 + 1000 * (-b))
        # y1 = int(y0 + 1000 * (a))
        # x2 = int(x0 - 1000 * (-b))
        # y2 = int(y0 - 1000 * (a))

        # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    plt.imshow(img)

    plt.show()


detectLines(cv2.imread('../images/im01.jpg', 1), 1)
detectLines(cv2.imread('../images/im02.jpg', 1), 2)

# theta = t * 360 / 5
# cos = math.cos(math.radians(theta))
# sin = math.sin(math.radians(theta))
# x1, x2 = 0, height - 1
# if sin == 0:
#     sin = 0.0001
# y1, y2 = int((rho - x1 * cos) / sin), int((rho - x2 * cos) / sin)
