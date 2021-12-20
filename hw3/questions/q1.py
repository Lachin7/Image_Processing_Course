import math

import numpy as np
import cv2
from matplotlib import pyplot as plt


def detectLines(img, i):
    kernel = np.ones((3, 3), np.float32) / 3 ** 2
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    blurred_img = cv2.filter2D(img, -1, kernel)
    edges = cv2.Canny(image=blurred_img, threshold1=350, threshold2=380)
    cv2.imwrite("../results/res0" + str(i) + ".jpg", edges)
    plt.imshow(edges)
    plt.show()

    height, width = edges.shape
    # nco = 10  # normalized coefficient to find the required interval for rho
    theta_size, rho_size = 180, int(np.ceil(np.sqrt(height ** 2 + width ** 2)))
    A = np.zeros((rho_size, theta_size))
    B = np.zeros((rho_size, theta_size, 4), dtype="uint16")
    for (y, x) in list(zip(*np.where(edges == 255))):
        for t in range(0, theta_size):
            # xt, yt = (2 * (x - x / 2)) / width, (2 * (-y + y / 2)) / height
            xt, yt = x, y

            # cos = math.cos(math.radians(t))
            # sin = math.sin(math.radians(t))
            cos = np.cos(np.deg2rad(2*t))
            sin = np.sin(np.deg2rad(2*t))
            rho = int(xt * cos + yt * sin)
            A[rho, t] += 1
            if B[rho, t, 0] == 0:
                B[rho, t, 0], B[rho, t, 1] = x, y
            else:
                B[rho, t, 2], B[rho, t, 3] = x, y

    cv2.imwrite("../results/res0" + str(i + 2) + "-hough-space.jpg", A)
    plt.imshow(A)
    threshold = 80
    for (rho, t) in list(zip(*np.where(A > threshold))):
        cos, sin = np.cos(np.deg2rad(2*t)), np.sin(np.deg2rad(2*t))
        x0 = cos * rho
        y0 = sin * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # x1, y1, x2, y2, cons = B[rho, t, 0], B[rho, t, 1], B[rho, t, 2], B[rho, t, 3], 2
        # cv2.line(img, (x1 * cons, x2 * cons), (y1 * cons, y2 * cons), (0, 0, 255), 3)

    plt.imshow(img)
    cv2.imwrite("../results/res0" + str(i + 4) + "-lines.jpg", img)
    plt.show()


detectLines(cv2.imread('../images/im01.jpg'), 1)
# theta = t * 360 / 5
# cos = math.cos(math.radians(theta))
# sin = math.sin(math.radians(theta))
# x1, x2 = 0, height - 1
# if sin == 0:
#     sin = 0.0001
# y1, y2 = int((rho - x1 * cos) / sin), int((rho - x2 * cos) / sin)
