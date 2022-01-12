import cv2
import numpy as np
import math
from scipy.interpolate import interp1d, UnivariateSpline

image = cv2.imread("../images/tasbih.jpg")
height, width, _ = image.shape
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

points = np.asarray(points)


def interpolate_points(points):
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]
    alpha = np.linspace(0, 1, 500)
    interpolator = interp1d(distance, points, kind='cubic', axis=0)
    return np.asarray(interpolator(alpha))


interpolate_points(points)


def calculate_d():
    points_diff = np.roll(points, 1, axis=0) - points
    return np.mean(np.sqrt(np.power(points_diff[:, 0], 2) + np.power(points_diff[:, 1], 2)), axis=0)


def calculate_E_external():
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    gradient = -(gradient_x ** 2 + gradient_y ** 2)
    return gradient


out = cv2.VideoWriter('contour.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))


def show_iteration():
    img_to_show = image.copy()
    cv2.drawContours(img_to_show, [interpolate_points(points).astype(int)], 0, (255, 255, 0), 1)
    out.write(img_to_show)
    if iteration == iterations - 1:
        cv2.imwrite('../results/res11.jpg', img_to_show)
        out.release()


# define some constants
alpha, gamma, center_closeness = 45, 3, 7
n = len(points)
iterations = 200
neighbors = np.asarray([[-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2], [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
                        [0, -2], [0, -1], [0, 0], [0, 1], [0, 2], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
                        [2, -2], [2, -1], [2, 0], [2, 1], [2, 2]])
neighborhood_size = 25

E_external = calculate_E_external()
for iteration in range(iterations):
    cost, path = np.zeros((n, neighborhood_size)), np.zeros((n, neighborhood_size), dtype=np.int16)
    center_x, center_y = np.average(points[:, 0]), np.average(points[:, 1])
    d = calculate_d()
    for i in range(n):
        for j in range(neighborhood_size):
            min_val, min_index = math.inf, 0
            current_v = points[i, :] + neighbors[j, :]
            for k in range(neighborhood_size):
                prev_v = points[i - 1, :] + neighbors[k, :]
                e = cost[i - 1, k] + (alpha * np.power(np.power(np.linalg.norm(current_v - prev_v), 2) - d, 2))
                if e < min_val:
                    min_val, min_index = e, k
            E_ex = E_external[int(current_v[0]), int(current_v[1]), 0] + E_external[
                int(current_v[0]), int(current_v[1]), 1] + E_external[
                       int(current_v[0]), int(current_v[1]), 2]

            cost[i, j] = min_val + (gamma * E_ex) + (
                        center_closeness * np.power(np.linalg.norm(current_v - [center_x, center_y]), 2))
            path[i, j] = min_index

    min_i = (np.where(cost[cost.shape[0] - 1, :] == np.amin(cost[cost.shape[0] - 1, :])))[0][0]
    print(min_i)
    mini = points[n - 1, :] + neighbors[min_i]
    for col in range(path.shape[0] - 1, 0, -1):
        min_i = path[col, min_i]
        points[col, :] = points[col - 1, :] + neighbors[min_i]

    points[0, :] = mini
    show_iteration()

# add more points
# def generate_more_points(points):
#     new_points = []
#     n = len(points)
#     for i in range(n):
#         new_points.append(points[i, :])
#         new_points.append(
#             [int((points[i, 0] + points[(i + 1) % n, 0]) / 2), int((points[i, 1] + points[(i + 1) % n, 1]) / 2)])
#     return new_points


# while len(points) < 300:
#     points = np.asarray(generate_more_points(points))
