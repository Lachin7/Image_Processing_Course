import math

import cv2
import numpy as np
from scipy.spatial import Delaunay

img_1 = cv2.imread("../images/res01.jpg")
img_2 = cv2.imread("../images/res02.jpg")
h1, w1, _ = img_1.shape


def readPoints():  # read the points from file:
    with open("../images/Points.txt", "r") as file:
        n = int(file.readline())
        pts_1, pts_2 = np.ndarray((int(n / 2), 2), dtype=float), np.ndarray((int(n / 2), 2), dtype=float)
        lines = file.readlines()[0:]
        for i in range(n):
            values = lines[i].split()
            j = math.floor(i / 2)
            if i % 2 == 0:
                pts_1[j, 0], pts_1[j, 1] = float(values[1]), float(values[0])
            else:
                pts_2[j, 0], pts_2[j, 1] = float(values[1]), float(values[0])

    file.close()
    return pts_1, pts_2, int(n / 2)


def interpolate():
    tris_1 = (pts_1[tri.simplices]).astype(int)
    tris_2 = (pts_2[tri.simplices]).astype(int)
    tris_res = (interpolation_pts[tri.simplices]).astype(int)

    for i in range(tris_1.shape[0]):
        tri_1, tri_2, tri_res = tris_1[i, :], tris_2[i, :], tris_res[i, :]

        rec_1, rec_2, (x, y, w, h) = cv2.boundingRect(tri_1), cv2.boundingRect(tri_2), cv2.boundingRect(tri_res)

        tri_1[:, 0], tri_1[:, 1] = tri_1[:, 0] - rec_1[0], tri_1[:, 1] - rec_1[1]
        tri_2[:, 0], tri_2[:, 1] = tri_2[:, 0] - rec_2[0], tri_2[:, 1] - rec_2[1]
        tri_res[:, 0], tri_res[:, 1] = tri_res[:, 0] - x, tri_res[:, 1] - y

        img_rec_1 = img_1[rec_1[1]: rec_1[1] + rec_1[3], rec_1[0]: rec_1[0] + rec_1[2]]
        img_rec_2 = img_2[rec_2[1]: rec_2[1] + rec_2[3], rec_2[0]: rec_2[0] + rec_2[2]]
        mask = np.zeros((h, w, 3))
        cv2.fillConvexPoly(mask, tri_res.astype(np.int32), (1.0, 1.0, 1.0), 16, 0)

        # find the affine transform:
        size = (w, h)
        warp_mat_1 = cv2.getAffineTransform(np.float32(tri_1), np.float32(tri_res))
        warped1 = cv2.warpAffine(img_rec_1, warp_mat_1, size, None, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)

        warp_mat_2 = cv2.getAffineTransform(np.float32(tri_2), np.float32(tri_res))
        warped2 = cv2.warpAffine(img_rec_2, warp_mat_2, size, None, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)

        interpolated_warp = (1 - t) * warped1 + t * warped2

        res[y:y + h, x:x + w] = res[y:y + h, x:x + w] * (1 - mask) + interpolated_warp * mask


pts_1, pts_2, n = readPoints()
tri = Delaunay(pts_1)

out = cv2.VideoWriter('../results/morph.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (w1, h1))
img_1, img_2 = img_1.astype(float), img_2.astype(float)
results = []
for frame in range(1, 46):
    t = (frame - 1) / 44
    interpolation_pts = (1 - t) * pts_1 + t * pts_2
    res = np.zeros((h1, w1, 3))
    interpolate()
    out.write(res.astype(np.uint8))
    results.append(res.astype(np.uint8))

cv2.imwrite("../results/res03.jpg", results[14])
cv2.imwrite("../results/res04.jpg", results[29])

results.reverse()
for i in range(0, 45):
    out.write(results[i])
out.release()
