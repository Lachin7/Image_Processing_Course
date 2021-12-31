import numpy as np
import cv2
import random


def matchPatch(texture, patch, strip_size, patch_size, direction=0):
    texture, patch = texture.astype(np.float32), patch.astype(np.float32)
    mask, limit_y, limit_x = np.zeros(patch.shape, dtype=np.float32), None, None
    if direction == 0:
        result, limit_x = cv2.matchTemplate(texture, patch[:, -strip_size:, :], 1), texture.shape[1] - patch_size
    elif direction == 1:
        result, limit_y = cv2.matchTemplate(texture, patch[-strip_size:, :, :], 1), texture.shape[0] - patch_size
    else:
        mask[0:strip_size, 0:strip_size, :] = 1
        result, limit_y = cv2.matchTemplate(texture, patch, 1, mask=mask), texture.shape[0]
    ry, rx = findRandomMin(result, 5, limit_y, limit_x)
    return texture[ry:ry + patch_size, rx:rx + patch_size, :]


def findRandomMin(result, num, limit_y, limit_x):
    idx = result.ravel().argsort()[:num]
    locations = np.stack(np.unravel_index(idx, result.shape)).T
    if limit_y is not None:
        indices = np.where(locations[:, 0] < limit_y)
    elif limit_x is not None:
        indices = np.where(locations[:, 1] < limit_x)
    yLoc, xLoc = np.zeros(indices[0].size, int), np.zeros(indices[0].size, int)
    for i in range(yLoc.shape[0]):
        yLoc[i], xLoc[i] = locations[indices[0][i], 0], locations[indices[0][i], 1]
    if yLoc.shape[0] == 0:
        return findRandomMin(result, num + 1, limit_y, limit_x)
    else:
        r = np.random.randint(0, yLoc.shape[0])
        return yLoc[r], xLoc[r]


def findMinCutPatch(patch1, patch2, strip_size, direction=0):
    if direction == 1:
        strip1, strip2 = patch1[-strip_size:, :, :], patch2[:strip_size, :, :]
        mask = makeMinCutHorizontallyMask(strip1, strip2)
        return strip1 * mask + strip2 * (1 - mask)
    elif direction == 0:
        strip1, strip2 = patch1[:, -strip_size:, :], patch2[:, :strip_size, :]
        mask = np.rot90(makeMinCutHorizontallyMask(np.rot90(strip1, -1), np.rot90(strip2, -1)), 1)
        return strip1 * mask + strip2 * (1 - mask)
    else:
        mask_h, mask_v = np.zeros(patch2.shape), np.zeros(patch2.shape)
        strip1, strip2 = patch1[:strip_size, :, :], patch2[:strip_size, :, :]
        mask_h[:strip_size, :, :] = makeMinCutHorizontallyMask(strip1, strip2)
        strip3, strip4 = patch1[:, :strip_size, :], patch2[:, :strip_size, :]
        mask_v[:, :strip_size, :] = np.rot90(makeMinCutHorizontallyMask(np.rot90(strip3, -1), np.rot90(strip4, -1)), 1)
        mask = np.logical_or(mask_v, mask_h)
        return patch1 * mask + patch2 * (1 - mask)


def makeMinCutHorizontallyMask(strip1, strip2):
    difference = np.sum(np.abs(strip1 - strip2), axis=2)

    cost, path = np.zeros(difference.shape, dtype=np.int16), np.zeros(difference.shape, dtype=np.int16)
    cost[:, 0] = difference[:, 0]

    for j in range(1, difference.shape[1]):
        for i in range(difference.shape[0]):
            min_val, res_neighbor = 100000000, 0
            for neighbor in -1, 0, 1:
                if 0 <= i - neighbor < difference.shape[0]:
                    if cost[i - neighbor, j - 1] < min_val:
                        min_val, res_neighbor = cost[i - neighbor, j - 1], neighbor
            cost[i, j] = min_val + difference[i, j]
            path[i, j - 1] = i - res_neighbor
    min_i = (np.where(cost[:, cost.shape[1] - 1] == np.amin(cost[:, cost.shape[1] - 1])))[0][0]
    mask = np.zeros(strip1.shape)

    for col in range(path.shape[1] - 1, 0, -1):
        mask[:min_i, col, :] = 1
        min_i = path[min_i, col - 1]
    return mask


def synthesize(image, res_name):
    # first choose a random patch:
    texture = image
    height, width, _ = texture.shape
    patch_size, strip_size, diff = 100, 20, 80
    h_start_index, w_start_index = random.randint(0, height - patch_size), random.randint(0, width - patch_size)
    patch = texture[h_start_index: h_start_index + patch_size, w_start_index: w_start_index + patch_size, :]

    result = np.zeros((2500, 2500, 3))
    result[0: patch_size, 0: patch_size, :] = patch

    for i in range(0, 31):
        for j in range(0, 31):
            if i == 0 and j != 0:
                adjacent_v_patch = result[0: patch_size, j * diff + strip_size - patch_size: j * diff + strip_size, :]
                new_patch = matchPatch(texture, adjacent_v_patch, strip_size, patch_size, direction=0)
                min_cut_strip = findMinCutPatch(adjacent_v_patch, new_patch, strip_size, direction=0)
                result[0:patch_size, j * diff: j * diff + strip_size, :] = min_cut_strip
                result[0:patch_size, j * diff + strip_size: j * diff + patch_size, :] = new_patch[:, strip_size:, :]
            if i != 0 and j == 0:
                adjacent_h_patch = result[i * diff + strip_size - patch_size: i * diff + strip_size, 0: patch_size, :]
                new_patch = matchPatch(texture, adjacent_h_patch, strip_size, patch_size, direction=1)
                min_cut_strip = findMinCutPatch(adjacent_h_patch, new_patch, strip_size, direction=1)
                result[i * diff: i * diff + strip_size, 0:patch_size, :] = min_cut_strip
                result[i * diff + strip_size: i * diff + patch_size, 0:patch_size, :] = new_patch[strip_size:, :, :]
            elif i != 0 and j != 0:
                adjacent_tc_patch = result[diff * i: diff * i + patch_size, diff * j: diff * j + patch_size, :]
                new_patch = matchPatch(texture, adjacent_tc_patch, strip_size, patch_size, direction=2)
                min_cut_patch = findMinCutPatch(adjacent_tc_patch, new_patch, strip_size, direction=2)
                result[diff * i: diff * i + patch_size, diff * j: diff * j + patch_size, :] = min_cut_patch

    res = np.zeros((2500, 2500 + texture.shape[1] + 300, 3))
    res[:, :2500, :] = result
    res[300:texture.shape[0] + 300, 2500 + 100 : 2500 + 100 + texture.shape[1] , :] = texture
    cv2.imwrite("../results/" + res_name + ".jpg", res)


# synthesize(cv2.imread('../images/Textures/texture01.jpg'), 'res11')
# synthesize(cv2.imread('../images/Textures/texture02.png'), 'res12')
synthesize(cv2.imread('../images/Textures/texture20.jpg'), 'res15')
synthesize(cv2.imread('../images/Textures/texture21.jpeg'), 'res16')

# adjacent_v_patch = result[diff*i: diff*i + patch_size, j * diff + strip_size - patch_size: j * diff + strip_size, :]
# adjacent_h_patch = result[i * diff + strip_size - patch_size: i * diff + strip_size, diff*j: diff*j + patch_size, :]
# result[i * diff: i*diff + strip_size, j * diff + (j - 1) * strip_size: j * (diff + strip_size), :] = min_cut_strip
# result[i*diff + strip_size: i*diff + patch_size, j * (diff + strip_size): j * (diff + strip_size) + diff, :] = new_patch[:, strip_size:, :]
# min_cut_strip = findMinCutPatchHorizontally(patch, new_patch, strip_size)
# result[i * diff + (i - 1) * strip_size: i * (diff + strip_size), j * diff: j*diff + strip_size, :] = min_cut_strip
# result[i * (diff + strip_size): i * (diff + strip_size) + diff, j*diff + strip_size: j*diff + patch_size, :] = new_patch[strip_size:, :, :]

# def matchTemplate(texture, patch, patch2, strip_size, direction=0):
#     height, width, _ = texture.shape
#     h, w, _ = patch.shape
#     # strip_h, strip_w = strip_size_ratio * h, strip_size_ratio * w
#     ssd = np.zeros((height - h, width - w))
#
#     for y in range(0, height - h):
#         for x in range(0, width - w):
#             if direction == 0:
#                 err = np.mean(np.power(texture[y: y + h, x: x + strip_size] - patch[:, -strip_size:], 2))
#             elif direction == 1:
#                 err = np.mean(np.power(texture[y: y + strip_size, x: x + w] - patch[-strip_size:, :], 2))
#             else:
#                 err = np.mean(np.power(texture[y: y + strip_size, x: x + w] - patch[-strip_size:, :], 2)) + np.mean(
#                     np.power(texture[y: y + h, x: x + strip_size] - patch2[:, -strip_size:], 2))
#             if err > 0:
#                 ssd[y, x] = err
#     minVal = np.min(ssd)
#     tolerance = 0.5
#     y, x = np.where(ssd < (1.0 + tolerance) * (minVal))
#     c = np.random.randint(0, len(y))
#     y, x = y[c], x[c]
#     return texture[y:y + h, x:x + w]
