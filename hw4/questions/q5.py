import cv2
import numpy as np

image = cv2.imread("../images/tasbih.jpg")

points_img = image.copy()
# collect the points from mouse clicks:
points = []


def collect_point(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(points_img, (x, y), 2, (255, 255, 0), -1)
        points.append([x, y])
        mouseX, mouseY = x, y


cv2.namedWindow('image')
cv2.setMouseCallback('image', collect_point)

while 1:
    cv2.imshow('image', points_img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
# print(points)
# print(np.roll(points, 1, axis=0))
def calculate_d():
    points_diff = np.roll(points, 1, axis=0) - points
    return np.mean(np.sqrt(np.power(points_diff[:, 0], 2)+np.power(points_diff[:, 1], 2)), axis=0)

def calculate_E_external():
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    gradient = -(gradient_x ** 2 + gradient_y ** 2)
    return gradient

# define some constants
alpha, beta, gamma = 1, 1, 1
iterations = 100

for interation in range(iterations):



# calculate E_external:

