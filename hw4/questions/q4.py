import cv2
import numpy as np
import skimage

image = cv2.imread("../images/birds.jpg")
scale = 0.25
resized_img = cv2.resize(image, (0, 0), fx=scale, fy=scale)
height, width, _ = resized_img.shape
# apply the felzenszwalb method to do image segmentation
# scale=800, sigma=0.6, min_size=100 # scale=50, sigma=0.5, min_size=100
segmented_img = skimage.segmentation.felzenszwalb(resized_img, scale=800, sigma=0.6, min_size=100)
segments_num = np.max(segmented_img) + 1
hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)


def make_feature_vector(segment_num):
    segment = (segmented_img == segment_num)
    hsv_av = (np.average(hsv_img[segment]) - np.average(hsv_img)) / np.std(hsv_img)
    rgb_av = (np.average(resized_img[segment]) - np.average(resized_img)) / np.std(resized_img)
    y_av = (np.average(np.argwhere(segment)[:, 0]) - height / 2) / height
    g_av = (np.average(resized_img[segment][:, 1]) - np.average(resized_img[:, :, 1])) / np.std(resized_img[:, :, 1])
    s_av = (np.average(hsv_img[segment][:, 1]) - np.average(hsv_img[:, :, 1])) / np.std(hsv_img[:, :, 1])
    location = np.argwhere(segment)

    size1 = ((location[:, 0].max() - location[:, 0].min()) * (
            location[:, 1].max() - location[:, 1].min())) / 7500
    size2 = np.count_nonzero(segment) / 1000
    return [hsv_av, rgb_av, 3 * y_av, g_av, s_av, size1, 2 * size2]


def difference(v1, v2):
    return np.average(np.power(np.subtract(v1, v2), 2))


result = np.zeros(resized_img.shape, dtype='uint8')
bird_sample = (int(2167 * scale), int(1078 * scale))

target_num = segmented_img[bird_sample]
sample_bird_feature = make_feature_vector(target_num)
threshold = 0.09
diffs = np.empty(0, dtype=float)
for k in range(segments_num):
    feature = make_feature_vector(k)
    diff = difference(feature, sample_bird_feature)
    diffs = np.append(diffs, diff)
    if diff < threshold:
        segment = segmented_img == k
        where = np.argwhere(segment)
        result[segmented_img == k] = resized_img[segmented_img == k]
        # cv2.rectangle(result, (where[:, 1].min(), where[:, 0].min()), (where[:, 1].max(), where[:, 0].max()),
        #               (0, 0, 255), 2)
        # cv2.putText(result, str(diff), (where[:, 1].min(), where[:, 0].min()), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
        #             (36, 255, 12), 1)
        # print(str(feature) + "  " + str(diff))

result = cv2.resize(result, (0, 0), fx=1 / scale, fy=1 / scale)

cv2.imwrite("../results/res10.jpg", result)
