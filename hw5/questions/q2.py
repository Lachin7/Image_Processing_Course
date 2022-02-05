import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

img_src = cv2.imread("../images/res05.jpg")
img_tar = cv2.imread("../images/res06.jpg")
mask = cv2.imread("../images/dog_mask.jpg", 0)

n, m, _ = img_tar.shape
translation_mat = np.float32([[1, 0, 290], [0, 1, 402]])
img_src = cv2.warpAffine(img_src, translation_mat, (m, n))
mask = cv2.warpAffine(mask, translation_mat, (m, n))
mask[mask < 0.1] = 0


def laplacian_mat():  # generate the laplacian matrix
    data, I = np.ones((m * m)), np.eye(m)
    sparse_matrix = sp.spdiags(np.array([data, -2 * data, data]), np.array([-1, 0, 1]), n, n)
    return -1 * (sp.kron(I, sparse_matrix) + sp.kron(sparse_matrix, I)).tolil()


A = laplacian_mat()
lap = A.tocsc()

# since the area does not necessarily have a rectangular shape, we should set the ones outside this region to identity.
for y in range(1, n - 1):
    for x in range(1, m - 1):
        if mask[y, x] == 0:
            z = x + y * m
            A[z, z], A[z, z + m], A[z, z - m], A[z, z + 1], A[z, z - 1] = 1, 0, 0, 0, 0
A = A.tocsc()

src_reshaped, tar_reshaped = img_src.copy().reshape((m * n, 3)), img_tar.copy().reshape((m * n, 3))
mask = mask.flatten()
b = lap.dot(src_reshaped)
b[mask == 0] = tar_reshaped[mask == 0]
res = spsolve(A, b).reshape((n, m, 3))
res[res > 255], res[res < 0] = 255, 0

cv2.imwrite("../results/res07.jpg", res.astype('uint8'))

# mask_flat = mask.flatten()
# for channel in range(3):
#     source_flat = img_src[0:n, 0:m, channel].flatten()
#     target_flat = img_tar[0:n, 0:m, channel].flatten()
#
#     mat_b = lap.dot(source_flat)
#
#     mat_b[mask_flat == 0] = target_flat[mask_flat == 0]
#
#     x = spsolve(A, mat_b)
#
#     x = x.reshape((n, m))
#
#     x[x > 255] = 255
#     x[x < 0] = 0
#     x = x.astype('uint8')
#
#     img_tar[0:n, 0:m, channel] = x
