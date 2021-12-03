import numpy as np
import cv2

path = "../images/Flowers.jpg"
img = cv2.imread(path)
r, g, b = cv2.split(img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
print(h.shape)

mask1= np.logical_and(h <= 178, h >= 138, s>=30)
# mask2 = np.logical_and(h >= 165, h <= 175)
# mask = np.logical_or(mask1, mask2)
hsv[mask1, 0] = 25
yellowImage = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

kernel = np.ones((11, 11), np.float32) / 11 ** 2
blurredImage = cv2.filter2D(img, -1, kernel)
blurHsv = cv2.cvtColor(blurredImage, cv2.COLOR_BGR2HSV)

h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

mask1 = np.logical_and(h <= 26, h >= 24)
# mask2 = np.logical_and(h >= 165, h <= 175)
# mask3 = np.logical_or(mask1, mask2)
mask = np.dstack((mask1, mask1, mask1))

result = np.where(mask, hsv, blurHsv)
result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

cv2.imwrite("res06.jpg", result)
cv2.imshow('image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
