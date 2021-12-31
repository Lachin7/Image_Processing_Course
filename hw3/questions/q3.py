import random

import numpy as np
import cv2


def SSD(shifted1, shifted2):
    if shifted1.shape != shifted2.shape:
        return float("inf")
    return np.average((shifted1 - shifted2) ** 2)


def apply_offsets(image, A, B, offset, mask_x0, mask_y0, p_half):
    for i in range(offset.shape[1]):
        for j in range(offset.shape[0]):
            image[mask_y0 + j, mask_x0 + i, :] = B[offset[j, i, 0], offset[j, i, 1], :]
            arr = np.ones((3, 3, 3))

# a weighted one :
            p_half, ps = 1, 3
            gaussianKernel = np.zeros((ps, ps))
            for x in range(-p_half, p_half+1):
                for y in range(-p_half, p_half+1):
                    gaussianKernel[x + 1, y + 1] = np.exp(- (np.power(x, 2) + np.power(y, 2)) / (2 * 1 * 1)) / (
                            2 * np.pi * 1 * 1)
            gaussianKernel = np.dstack((gaussianKernel, gaussianKernel, gaussianKernel))
            for k in range(-p_half, p_half + 1):
                for l in range(-p_half, p_half + 1):
                    if -1 < j - l < offset.shape[0] and -1 < i - k < offset.shape[1]:
                        y, x = offset[j - l, i - k, 0] + l, offset[j - l, i - k, 1] + k
                        if -1 < y < B.shape[0] and -1 < x < B.shape[1]:
                            arr[l, k, :] = B[y, x, :]
                        else:
                            arr[l, k, :] = B[offset[j, i, 0], offset[j, i, 1], :]
                    else:
                        arr[l, k, :] = image[j-l + mask_y0, i-k+mask_x0, :]
            image[mask_y0 + j, mask_x0 + i, :] = np.average(arr, weights=gaussianKernel, axis=(0, 1))

    #
    #
    # #     average, num = B[offset[j, i, 0], offset[j, i, 1], :], 1
    #         image[mask_y0 + j, mask_x0 + i, :] = average / num


#
# if i % 2 == 0:
#     rand_y, rand_x = random.randint(-p_half, p_half + 1), random.randint(-p_half, p_half + 1)
#     if -1 < j - rand_y < offset.shape[0] and -1 < i - rand_x < offset.shape[1]:
#         y, x = offset[j - rand_y, i - rand_x, 0] + rand_y, offset[j - rand_y, i - rand_x, 1] + rand_x
#         if -1 < y and x < B.shape[0]:
#             average = B[y, x, :]
#             num = 1
# else:

def getPatchCenteredIn(center_y, center_x, ps_half, arr):
    return arr[center_y - ps_half: center_y + ps_half + 1, center_x - ps_half: center_x + ps_half + 1, :]


def odd_iteration(image, A, B, offset, mask_x0, mask_y0):
    p_half, patch_size = 2, 5
    for j in range(0, offset.shape[0]):
        for i in range(0, offset.shape[1]):
            src_patch = getPatchCenteredIn(j + mask_y0, i + mask_x0, p_half, image)
            rj, ri = offset[j, i, 0], offset[j, i, 1]
            ref_patch = getPatchCenteredIn(rj, ri, p_half, B)

            if j != 0 and i == 0:
                n2j, n2i = offset[j - 1, i, 0], offset[j - 1, i, 1]
                neighbor2 = getPatchCenteredIn(n2j + 1, n2i, p_half, B)
                ref_ssd, n2_ssd = SSD(src_patch, ref_patch), SSD(src_patch, neighbor2)
                min_ssd = min(ref_ssd, n2_ssd)
                if n2_ssd == min_ssd:
                    offset[j, i, 0], offset[j, i, 1] = n2j + 1, n2i
                random_search(B, offset, patch_size, p_half, src_patch, min_ssd, i, j)
            elif j == 0 and i != 0:
                n1j, n1i = offset[j, i - 1, 0], offset[j, i - 1, 1]
                neighbor1 = getPatchCenteredIn(n1j, n1i + 1, p_half, B)
                ref_ssd, n1_ssd = SSD(src_patch, ref_patch), SSD(src_patch, neighbor1)
                min_ssd = min(ref_ssd, n1_ssd)
                if n1_ssd == min_ssd:
                    offset[j, i, 0], offset[j, i, 1] = n1j, n1i + 1
                random_search(B, offset, patch_size, p_half, src_patch, min_ssd, i, j)
            elif j != 0 or i != 0:
                n1j, n1i, n2j, n2i = offset[j, i - 1, 0], offset[j, i - 1, 1], offset[j - 1, i, 0], offset[
                    j - 1, i, 1]
                neighbor1 = getPatchCenteredIn(n1j, n1i + 1, p_half, B)
                neighbor2 = getPatchCenteredIn(n2j + 1, n2i, p_half, B)
                ref_ssd, n1_ssd, n2_ssd = SSD(src_patch, ref_patch), SSD(src_patch, neighbor1), SSD(src_patch,
                                                                                                    neighbor2)
                min_ssd = min(ref_ssd, n1_ssd, n2_ssd)
                if n1_ssd == min_ssd:
                    offset[j, i, 0], offset[j, i, 1] = n1j + 1, n1i
                elif n2_ssd == min_ssd:
                    offset[j, i, 0], offset[j, i, 1] = n2j, n2i + 1
                random_search(B, offset, patch_size, p_half, src_patch, min_ssd, i, j)

    apply_offsets(image, A, B, offset, mask_x0, mask_y0, p_half)


def even_iteration(image, A, B, offset, mask_x0, mask_y0):
    A_h, A_w, _ = A.shape
    p_half, patch_size = 2, 5
    for j in range(offset.shape[0] - 1, -1):
        for i in range(offset.shape[1] - 1, -1):
            src_patch = getPatchCenteredIn(j + mask_y0, i + mask_x0, p_half, image)
            rj, ri = offset[j, i, 0], offset[j, i, 1]
            ref_patch = getPatchCenteredIn(rj, ri, p_half, B)

            if j != A_h - 1 and i == A_w - 1:
                n2j, n2i = offset[j + 1, i, 0], offset[j + 1, i, 1]
                neighbor2 = getPatchCenteredIn(n2j - 1, n2i, p_half, B)
                ref_ssd, n2_ssd = SSD(src_patch, ref_patch), SSD(src_patch, neighbor2)
                min_ssd = min(ref_ssd, n2_ssd)
                if n2_ssd == min_ssd:
                    offset[j, i, 0], offset[j, i, 1] = n2j - 1, n2i
                random_search(B, offset, patch_size, p_half, src_patch, min_ssd, i, j)
            elif j == A_h - 1 and i != A_w - 1:
                n1j, n1i = offset[j, i + 1, 0], offset[j, i + 1, 1]
                neighbor1 = getPatchCenteredIn(n1j, n1i - 1, p_half, B)
                ref_ssd, n1_ssd = SSD(src_patch, ref_patch), SSD(src_patch, neighbor1)
                min_ssd = min(ref_ssd, n1_ssd)
                if n1_ssd == min_ssd:
                    offset[j, i, 0], offset[j, i, 1] = n1j, n1i - 1
                random_search(B, offset, patch_size, p_half, src_patch, min_ssd, i, j)
            elif j != 0 or i != 0:
                n1j, n1i, n2j, n2i = offset[j, i - 1, 0], offset[j, i - 1, 1], offset[j - 1, i, 0], offset[
                    j - 1, i, 1]
                neighbor1 = getPatchCenteredIn(n1j, n1i - 1, p_half, B)
                neighbor2 = getPatchCenteredIn(n2j - 1, n2i, p_half, B)
                ref_ssd, n1_ssd, n2_ssd = SSD(src_patch, ref_patch), SSD(src_patch, neighbor1), SSD(src_patch,
                                                                                                    neighbor2)
                min_ssd = min(ref_ssd, n1_ssd, n2_ssd)
                if n1_ssd == min_ssd:
                    offset[j, i, 0], offset[j, i, 1] = n1j - 1, n1i
                elif n2_ssd == min_ssd:
                    offset[j, i, 0], offset[j, i, 1] = n2j, n2i - 1
                random_search(B, offset, patch_size, p_half, src_patch, min_ssd, i, j)
    apply_offsets(image, A, B, offset, mask_x0, mask_y0, p_half)


def random_search(B, offset, patch_size, ps_half, src_patch, min_ssd, i, j):
    w, alpha = B.shape[0], 1 / 2
    while w > 2 * patch_size + 1:
        x_rand, y_rand = np.random.randint(int(w / 10), w - int(w / 10)), np.random.randint(int(w / 10),
                                                                                            w - int(w / 10))
        random_patch = getPatchCenteredIn(y_rand, x_rand, ps_half, B)
        if SSD(src_patch, random_patch) < min_ssd:
            offset[j, i, 0], offset[j, i, 1] = y_rand, x_rand
        w = np.floor(alpha * w)


def iterate(image, A, B, offset, mask_x0, mask_y0):
    for i in range(1, 5):
        if i % 2 == 1:
            odd_iteration(image, A, B, offset, mask_x0, mask_y0)
        else:
            # image, A, B, offset = np.rot90(image, 2), np.rot90(A, 2), np.rot90(B, 2), np.rot90(offset, 2)
            # odd_iteration(image, A, B, offset,
            #               image.shape[1] - mask_x0 - A.shape[1], image.shape[0] - mask_y0 - A.shape[0], )
            # image, A, B, offset = np.rot90(image, -2), np.rot90(A, -2), np.rot90(B, 2), np.rot90(offset, 2)
            even_iteration(image, A, B, offset, mask_x0, mask_y0)
    cv2.imwrite('../results/res3/iteration' + str(i) + '.jpg', image)


def convertToDepth(mask_x0, mask_y0, mask_x1, mask_y1, ref_x0, ref_y0, ref_x1, ref_y1, i):
    mask_x0, mask_y0 = int(mask_x0 * 0.5 ** i), int(mask_y0 * 0.5 ** i)
    mask_x1, mask_y1 = int(mask_x1 * 0.5 ** i), int(mask_y1 * 0.5 ** i)
    ref_x0, ref_y0 = int(ref_x0 * 0.5 ** i), int(ref_y0 * 0.5 ** i)
    ref_x1, ref_y1 = int(ref_x1 * 0.5 ** i), int(ref_y1 * 0.5 ** i)
    return mask_x0, mask_y0, mask_x1, mask_y1, ref_x0, ref_y0, ref_x1, ref_y1


def inpaint(img, mask_x0, mask_y0, mask_x1, mask_y1, ref_x0, ref_y0, ref_x1, ref_y1):
    depth = 3
    # image pyramid here:
    for i in range(depth, -1, -1):
        image = cv2.resize(img, (0, 0), fx=0.5 ** i, fy=0.5 ** i)
        mx0, my0, mx1, my1, rx0, ry0, rx1, ry1 = convertToDepth(mask_x0, mask_y0, mask_x1, mask_y1, ref_x0, ref_y0,
                                                                ref_x1, ref_y1, i)
        A = image[my0:my1, mx0:mx1, :]
        A_h, A_w, _ = A.shape
        B = image[ry0: ry1, rx0: rx1, :]
        if i == depth:
            offset = np.random.randint(5, B.shape[0] - 5, (A_h, A_w, 2))
        else:
            offset = np.repeat(offset, repeats=2, axis=0)
            offset = np.repeat(offset, repeats=2, axis=1)
            offset = offset * 2
        iterate(image, A, B, offset, mx0, my0)
        cv2.imwrite('../results/res3/depth' + str(i) + '.jpg', image)
    iterate(img, A, B, offset, mask_x0, mask_y0)


img = cv2.imread('../images/im04.jpg')
# inpaint(img, 730, 680, 980, 1210, 590, 1175, 1745, 2330)
birds = cv2.imread('../images/im03.jpg')
#p1: mask_x0, mask_y0, mask_x1, mask_y1, ref_x0, ref_y0, ref_x1, ref_y1
# inpaint(birds, mask_y0= 70, mask_y1=159, mask_x0=329, mask_x1=530, ref_x0=251, ref_x1=512, ref_y0=171, ref_y1=324)
# inpaint(birds, mask_y0= 102, mask_y1=155, mask_x0=327, mask_x1=388, ref_x0=333, ref_x1=423, ref_y0=211, ref_y1=296)
# inpaint(birds, mask_y0= 600, mask_y1=725, mask_x0=1130, mask_x1=1250, ref_x0=1236, ref_x1=1411, ref_y0=738, ref_y1=937)
inpaint(birds, mask_y0= 750, mask_y1=930, mask_x0=785, mask_x1=960, ref_x0=511, ref_x1=641, ref_y0=750, ref_y1=890)


cv2.imwrite('../results/res16.jpg', birds)

# plt.imshow(mask)
# plt.show


# def even_iteration(image, A, B, offset, mask_x0, mask_y0):
#     A_h, A_w, _ = A.shape
#     p_half, patch_size = 5, 11
#     for j in range(A_h - 1, 0, -1):
#         for i in range(A_w - 1, 0, -1):
#                 src_patch = image[j - p_half + mask_y0: j + p_half + mask_y0, i - p_half + mask_x0: i + p_half + mask_x0, :]
#                 rj, ri = offset[j, i, 0], offset[j, i, 1]
#                 ref_patch = B[rj - p_half: rj + p_half, ri - p_half: ri + p_half, :]
#                 if j + 1!= A_h and i
#                 n1j, n1i, n2j, n2i = offset[j, i + 1, 0], offset[j, i + 1, 1], offset[j + 1, i, 0], offset[j + 1, i, 1]
#                 neighbor1 = B[n1j - p_half: n1j + p_half, n1i - p_half - 1: n1i + p_half - 1, :]
#                 neighbor2 = B[n2j - p_half - 1: n2j + p_half - 1, n2i - p_half: n2i + p_half, :]
#                 ref_ssd, n1_ssd, n2_ssd = SSD(src_patch, ref_patch), SSD(src_patch, neighbor1), SSD(src_patch,
#                                                                                                     neighbor2)
#                 min_ssd = min(ref_ssd, n1_ssd, n2_ssd)
#                 if n1_ssd == min_ssd:
#                     offset[j, i, 0], offset[j, i, 1] = n1j, n1i - 1
#                 elif n2_ssd == min_ssd:
#                     offset[j, i, 0], offset[j, i, 1] = n2j - 1, n2i
#                 random_search(B, offset, patch_size, p_half, src_patch, min_ssd, i, j)


# marked_image = cv2.imread('../images/im04-marked.jpg')
# mask = np.where(np.abs(image - marked_image) > 0, 255, 0)
# image = cv2.imread('../images/im04.jpg')
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
# mask_area = np.logical_and(146 < hsv[:, :, 0], hsv[:, :, 0] < 152)
# mask = np.zeros(image.shape)
# mask[mask_area] = 255
# def iterate(image, A, B, offset, m_x0, m_y0, even):
#     A_h, A_w, _ = A.shape
#     p_half, patch_size = 5, 11
#     j_start, j_end, i_start, i_end, step = 0, A_h - 1, 0, A_w - 1, 1
#     if not even:
#         j_start, j_end, i_start, i_end, step = A_h - 1, 0, A_w - 1, 0, -1
#     for j in range(j_start, j_end, step):
#         for i in range(i_start, i_end, step):
#             src_patch = image[j - p_half + m_y0: j + p_half + m_y0, i - p_half + m_x0: i + p_half + m_x0, :]
#             rj, ri = offset[j, i, 0], offset[j, i, 1]
#             ref_patch = B[rj - p_half: rj + p_half, ri - p_half: ri + p_half, :]
#             if j != 0 and i == 0:
#                 n2j, n2i = offset[j - 1, i, 0], offset[j - 1, i, 1]
#                 neighbor2 = B[n2j - p_half + 1: n2j + p_half + 1, n2i - p_half: n2i + p_half, :]
#                 ref_ssd, n2_ssd = SSD(src_patch, ref_patch), SSD(src_patch, neighbor2)
#                 min_ssd = min(ref_ssd, n2_ssd)
#                 if n2_ssd == min_ssd:
#                     offset[j, i, 0], offset[j, i, 1] = n2j + 1, n2i
#                 random_search(B, offset, patch_size, p_half, src_patch, min_ssd, i, j)
#             elif j == 0 and i != 0:
#                 n1j, n1i = offset[j, i - 1, 0], offset[j, i - 1, 1]
#                 neighbor1 = B[n1j - p_half: n1j + p_half, n1i - p_half + 1: n1i + p_half + 1, :]
#                 ref_ssd, n1_ssd = SSD(src_patch, ref_patch), SSD(src_patch, neighbor1)
#                 min_ssd = min(ref_ssd, n1_ssd)
#                 if n1_ssd == min_ssd:
#                     offset[j, i, 0], offset[j, i, 1] = n1j, n1i + 1
#                 random_search(B, offset, patch_size, p_half, src_patch, min_ssd, i, j)
#             elif j != 0 or i != 0:
#                     n1j, n1i, n2j, n2i = offset[j, i - 1, 0], offset[j, i - 1, 1], offset[j - 1, i, 0], offset[
#                         j - 1, i, 1]
#                     neighbor1 = B[n1j - p_half: n1j + p_half, n1i - p_half + 1: n1i + p_half + 1, :]
#                     neighbor2 = B[n2j - p_half + 1: n2j + p_half + 1, n2i - p_half: n2i + p_half, :]
#                     # print(src_patch.shape, ref_patch.shape, neighbor1.shape, neighbor2.shape)
#                     # print(n1j, n1i, n2j, n2i)
#                     ref_ssd, n1_ssd, n2_ssd = SSD(src_patch, ref_patch), SSD(src_patch, neighbor1), SSD(src_patch,
#                                                                                                         neighbor2)
#                     min_ssd = min(ref_ssd, n1_ssd, n2_ssd)
#                     if n1_ssd == min_ssd:
#                         offset[j, i, 0], offset[j, i, 1] = n1j, n1i + 1
#                     elif n2_ssd == min_ssd:
#                         offset[j, i, 0], offset[j, i, 1] = n2j + 1, n2i
#                     random_search(B, offset, patch_size, p_half, src_patch, min_ssd, i, j)
