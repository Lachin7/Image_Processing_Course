import cv2

img_1 = cv2.imread("../images/res08.jpg")
img_2 = cv2.imread("../images/res09.jpg")
mask = cv2.imread("../images/mask.jpg", 0)


# laplacian stack:
def get_laplacian(k, img):
    blur = cv2.blur(img, (k, k), borderType=cv2.BORDER_REFLECT101)
    laplacian = img - blur
    return blur, laplacian


def apply_feathering(src, tar, mask, k):
    mask = cv2.blur(mask, (k, k), borderType=cv2.BORDER_REFLECT101)
    return src * mask + tar * (1 - mask)


stack1, stack2 = [img_1], [img_2]
for iteration in range(7):
    img_1, lap1 = get_laplacian(7, img_1)
    img_2, lap2 = get_laplacian(7, img_2)
    stack1.append(lap1), stack2.append(lap2)




