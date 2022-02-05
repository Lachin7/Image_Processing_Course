import cv2



def apply_blending(src_name, tar_name, mask_name):
    img_1 = cv2.imread("../images/" + src_name + ".jpg").astype(float)
    img_2 = cv2.imread("../images/" + tar_name + ".jpg").astype(float)
    mask = cv2.imread("../images/" + mask_name + ".jpg") / 255
    iter_num, blur_window, window_size_low, window_size_high = 7, 7, 7, 19
    stack1, stack2 = [], [] # build the stacks for these two images.
    for iteration in range(iter_num):
        img_1, lap1 = get_laplacian(blur_window, img_1)
        img_2, lap2 = get_laplacian(blur_window, img_2)
        stack1.append(lap1), stack2.append(lap2)

    stack1.append(img_1), stack2.append(img_2)
    res = apply_feathering(stack1[iter_num], stack2[iter_num], mask, window_size_high)

    for iteration in range(iter_num - 1, -1, -1):
        res += apply_feathering(stack1[iteration], stack2[iteration], mask, window_size_low)

    cv2.imwrite("../results/res10.jpg", scale(res))
    return scale(res)


# laplacian stack:
def get_laplacian(k, img):
    blur = cv2.GaussianBlur(img, (k, k), 20)
    laplacian = img - blur
    return blur, laplacian


def apply_feathering(src, tar, maskk, k):
    maskk = cv2.GaussianBlur(maskk, (k, k), 20)
    return tar * maskk + src * (1 - maskk)


def scale(array):
    min, max = array.min(), array.max()
    return ((array - min) * (1 / (max - min) * 255)).astype('uint8')


apply_blending("res08", "res09", "cat_bird_mask")
