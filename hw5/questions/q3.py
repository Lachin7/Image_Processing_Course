import cv2
import matplotlib.pyplot as plt
import numpy as np

img_1 = cv2.imread("../images/res08.jpeg").astype(float)
img_2 = cv2.imread("../images/res09.jpeg").astype(float)
mask = cv2.imread("../images/cat_bird_mask.jpeg")
res = np.zeros(img_1.shape)


# laplacian stack:
def get_laplacian(k, img):
    blur = cv2.GaussianBlur(img, (k, k), 20)
    laplacian = img - blur
    return blur, laplacian


def apply_feathering(src, tar, maskk, k):
    maskk = cv2.GaussianBlur(maskk, (k, k), 20)
    return tar * maskk + src * (1 - maskk)


iter_num, blur_window = 7, 7
stack1, stack2 = [], []
for iteration in range(iter_num):
    img_1, lap1 = get_laplacian(blur_window, img_1)
    img_2, lap2 = get_laplacian(blur_window, img_2)
    stack1.append(lap1), stack2.append(lap2)
    plt.imshow(img_1.astype('uint8'))
    plt.show()
    plt.imshow(lap1, cmap='gray')
    plt.show()

stack1.append(img_1), stack2.append(img_2)
res = apply_feathering(stack1[iter_num], stack2[iter_num], mask, 19)

for iteration in range(iter_num - 1, -1, -1):
    res += apply_feathering(stack1[iteration], stack2[iteration], mask, 7)

res1 = res.copy()
res1[res1 > 255] = 255
# res1[res1 < 0] = 0
res1 = res1.astype('uint8')

res2 = res.copy()


def scale(array):
    min, max = array.min(), array.max()
    return ((array - min) * (1 / (max - min) * 255)).astype('uint8')


res2 = scale(res2)
res2[res2 == 0] = res1[res2 == 0]

cv2.imwrite("../results/res10.jpg", res2)
