import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage
from sklearn import cluster

image = cv2.imread("../images/birds.jpg")
scale = 0.25
# plt.imshow(image)
# plt.show()
resized_img = cv2.resize(image, (0, 0), fx=scale, fy=scale)

height, width, _ = resized_img.shape
# apply the felzenszwalb method to do image segmentation
segmented_img = skimage.segmentation.felzenszwalb(resized_img, scale=800, sigma=0.6, min_size=100)
segments_num = np.max(segmented_img) + 1

label_image = np.zeros_like(resized_img)
for t in np.unique(segmented_img):
    label_image[segmented_img == t] = np.random.randint(0, 255, 3)
plt.figure(figsize=(10, 10))
plt.imsave("labels.jpg", label_image)

hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)

def make_feature_vector(segment_num):
    feature_vector = []
    segment = (segmented_img == segment_num)
    hsv_av = (np.average(hsv_img[segment]) - np.average(hsv_img))/np.std(hsv_img)
    rgb_av = (np.average(resized_img[segment]) - np.average(resized_img))/np.std(resized_img)
    y_av = (np.average(np.argwhere(segment)[:, 0]) - height/2)/height
    b_av = (np.average(resized_img[segment][:, 1]) - np.average(resized_img[:, :, 1]))/np.std(resized_img[:, :, 1])
    s_av = (np.average(hsv_img[segment][:, 1]) - np.average(hsv_img[:, :, 1]))/np.std(hsv_img[:, :, 1])
    location = np.argwhere(segment)

    size1 = ((location[..., 0].max() - location[..., 0].min()) * (location[..., 1].max() - location[..., 1].min()))/7500
    size2 = np.count_nonzero(segment)/1000
    return [hsv_av, rgb_av, 3 * y_av, b_av, s_av, size1, 2 * size2]
   # size =
# [-0.246950113712763, -0.29119909536290234, 0.34397205829218347, -0.3036354606797444, -0.7439041915853016, 0.361]  0.0
# [-0.4470950362842415, -0.9364959689006672, 0.49982832773139596, -0.9528285506541389, -0.539029048249852, 0.401]  0.15763045236324816
# [-0.42972531870054304, -0.9700057814424317, 0.5344250951447354, -0.9812590546650902, -0.5275355366754871, 0.377]  0.17278379697067606
# [-0.46459482580273537, -1.0352541658984107, 0.5291604907773387, -1.0478961597004053, -0.6457335627066497, 0.352]  0.1998207307829648

# features = np.zeros((segments_num, 3), dtype=float)
# for i in range(segments_num):
#     features[i, :] = make_feature_vector(i)
# bandwidth = cluster.estimate_bandwidth(resized_img, quantile=0.3, n_samples=344448, n_jobs=-1) * 0.2
#clusters = cluster.MeanShift(bandwidth=2000).fit([make_feature_vector(i) for i in range(segments_num)])
def difference(v1, v2):
    return np.average(np.power(np.subtract(v1, v2), 2))


result = np.zeros(resized_img.shape, dtype='uint8')
# bird_point = (int(2042 * scale), int(1957 * scale))
bird_point = (int(2167 * scale), int(1078 * scale))

target_num = segmented_img[bird_point]
sample_bird_feature = make_feature_vector(target_num)
threshold = 0.08
diffs = np.empty(0, dtype=float)
for k in range(segments_num):
    feature = make_feature_vector(k)
    diff = difference(feature, sample_bird_feature)
    diffs = np.append(diffs, diff)
    if diff < threshold:
        segment = segmented_img == k
        where = np.argwhere(segment)
        result[segmented_img == k] = resized_img[segmented_img == k]
        # cv2.rectangle(result, (where[..., 1].min(), where[..., 0].min()), (where[..., 1].max(), where[..., 0].max()), (0, 0, 255), 2)
        # cv2.putText(result, str(diff), (where[..., 1].min(), where[..., 0].min()), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36,255,12), 1)
        # print(str(feature) + "  " + str(diff))

# print(np.sort(diffs))
#
# for k in range(segments_num):
#     if clusters.labels_[k] == clusters.labels_[segmented_img[bird_point[0], bird_point[1]]]:
#         result[segmented_img==k] = resized_img[segmented_img==k]
#
result = cv2.resize(result, (0, 0), fx=1/scale, fy=1/scale)

# some of the best results up to this point:
# scale=800, sigma=0.6, min_size=100
# scale=50, sigma=0.5, min_size=100
# def show_segments(img):
#     for i in
cv2.imwrite("q4.jpg", result)
# plt.imshow(result)
# plt.show()

