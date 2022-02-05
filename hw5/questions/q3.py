import cv2
import matplotlib.pyplot as plt
import numpy as np

img_1 = cv2.imread("../images/res09.jpeg").astype(float)
img_2 = cv2.imread("../images/res08.jpeg").astype(float)
mask = cv2.imread("../images/cat_bird_mask.jpeg").astype(float)
res = np.zeros(img_1.shape)


# laplacian stack:
def get_laplacian(k, img):
    blur = cv2.blur(img, (k, k), borderType=cv2.BORDER_REFLECT101)
    laplacian = img - blur
    return blur, laplacian


def apply_feathering(src, tar, mask, k):
    mask = cv2.blur(mask, (k, k), borderType=cv2.BORDER_REFLECT101)
    plt.imshow(mask)
    plt.show()
    return src * mask + tar * (1 - mask)


iter_num, blur_window = 11, 30
stack1, stack2 = [], []
for iteration in range(iter_num):
    img_1, lap1 = get_laplacian(blur_window, img_1)
    img_2, lap2 = get_laplacian(blur_window, img_2)
    stack1.append(lap1), stack2.append(lap2)
    plt.imshow(img_1.astype('uint8'))
    plt.show()
    plt.imshow(lap1.astype('uint8'))
    plt.show()

stack1.append(img_1), stack2.append(img_2)
res = apply_feathering(stack1[iter_num], stack2[iter_num], mask, 38)
plt.imshow(res)

for iteration in range(iter_num - 1, -1, -1):
    res += apply_feathering(stack1[iteration], stack2[iteration], mask, 21)
    plt.imshow()

res[res > 255], res[res < 0] = 255, 0
cv2.imwrite("../results/res10.jpg", res.astype('uint8'))
