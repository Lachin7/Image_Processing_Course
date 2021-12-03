import numpy as np
import cv2
from matplotlib import pyplot as plt

near = cv2.imread("../results/res19-near.jpeg")
near = near.astype(float)
height, width, c = near.shape

far = cv2.imread("../results/res20-far.jpg")
far = far.astype(float)

x1, y1, x2, y2 = 104, 159, 187, 145
x3, y3, x4, y4 = 117, 179, 186, 166

# Find the scale transformation which maps points from the near one to the far image.
a = (x4 - x3) / (x2 - x1)
b = (y4 - y3) / (y2 - y1)
h = np.array([[a, 0, 0], [0, b, 0], [0, 0, 1]], dtype=float)
[[nx1], [ny1], [t]] = np.dot(h, np.array([[x1], [y1], [1]]))
nx1, ny1 = int(nx1), int(ny1)
xdiff, ydiff = x3 - nx1, y3 - ny1
near = near[xdiff:, ydiff:, ]
far = cv2.warpPerspective(far, h, (height, width))
far = far[0: width - xdiff, 0: height - ydiff, ]
cv2.imwrite('../results/res21-near.jpg', near)
cv2.imwrite('../results/res22-far.jpg', far)
height, width = near.shape[0], near.shape[1]


def generateGaussianKernel(height, width, sigma, mu1, mu2, number):
    gaussianKernel = np.zeros((height, width))
    for x in range(height):
        for y in range(width):
            gaussianKernel[x, y] = np.exp(- (np.power(x - mu1, 2) + np.power(y - mu2, 2)) / (2 * sigma * sigma))
    plt.imshow(gaussianKernel)
    plt.colorbar()
    plt.savefig('../results/res'+str(number)+'-'+str(sigma)+'.jpg')
    plt.show()
    return gaussianKernel


def generateLaplacianFilter(height, width, sigma, mu1, mu2, number):
    laplacianKernel = 1 - generateGaussianKernel(height, width, sigma, mu1, mu2, number)
    plt.imshow(laplacianKernel)
    plt.colorbar()
    plt.savefig('../results/res'+str(number)+'-'+str(sigma)+'.jpg')
    plt.show()
    return laplacianKernel


def generateFFT(img, name):
    ImgFft = np.fft.fft2(img, axes=(0, 1))
    shiftedImg = np.fft.fftshift(ImgFft)
    amplitude = np.abs(shiftedImg)
    logAmplitude = 20 * np.log(amplitude)
    cv2.imwrite('../results/' + name + '.jpg', logAmplitude)
    return shiftedImg


def applyFilter(image, filter, name):
    result = []
    for i in range(0, 3):
        result.insert(0, filter * image[:, :, 2 - i])
    uimg = np.dstack(result)
    mag = np.log(np.abs(uimg) + 1)
    cv2.imwrite('../results/' + name + '.jpg', mag)
    return uimg


def scale(array):
    min, max = array.min(), array.max()
    return ((array - min) * (1 / (max - min) * 255)).astype('uint8')


nearFFT = generateFFT(near, 'res23-dft-near')
farFFT = generateFFT(far, 'res24-dft-far')

r, s = 55, 8
gaussianKernel = generateGaussianKernel(height, width, r, int(height / 2), int(width / 2),25)
laplacianKernel = generateLaplacianFilter(height, width, s, int(height / 2), int(width / 2),26)

lowPassed = applyFilter(nearFFT, laplacianKernel, 'res27-lowpassed')
highPassed = applyFilter(farFFT, gaussianKernel, 'res28-highpassed')

result = (np.array(lowPassed) + np.array(highPassed)) / 2.0
cv2.imwrite('../results/res29-hybrid.jpg', np.log(np.abs(result)))

ishifted = np.fft.ifftshift(result)
final = np.fft.ifft2(ishifted, axes=(0, 1))
final = (np.real(final))
cv2.imwrite('../results/res30-hybrid-near.jpg', scale(final))

far = cv2.resize(final, None, fx=0.2, fy=0.2)
cv2.imwrite('../results/res31-hybrid-far.jpg', scale(far))

# laplacianKernel = np.zeros((height, width))
# c = -1 / (np.pi * sigma ** 4)
# for x in range(height):
#     for y in range(width):
#         m = (np.power(x - mu1, 2) + np.power(y - mu2, 2)) / (2 * sigma * sigma)
#         laplacianKernel[x, y] = c * (1 - m) * np.exp(-m)
# plt.imshow(laplacianKernel)
# plt.colorbar()
# plt.show()
# return laplacianKernel
# , 'res25-highpass-' + str(r)
# , 'res26-lowpass-' + str(s)

# nearFFT[(near.shape[0] // 2) - 40: (near.shape[0] // 2) + 40, (near.shape[1] // 2) - 40: (near.shape[1] // 2) + 40] = 0
# imgMask = np.zeros((269, 268), np.uint8)
# imgMask[(far.shape[0] // 2) - 50: (far.shape[0] // 2) + 50, (far.shape[1] // 2) - 50: (far.shape[1] // 2) + 50] = 1
# result = []
# for i in range(0, 3):
#     result.insert(0, np.multiply(imgMask, farFFT[:, :, 2 - i]))
# farFFT = np.dstack(result)
# h, status = cv2.findHomography(srcPoints, desPoints)
# so = []
# for i in range(0, 3):
#         ishifted = np.fft.ifftshift(lowPassed[:, :, 2 - i])
#         final = np.fft.ifft2(ishifted)
#         so.insert(0, final)
# uimg = np.dstack(so)
# final = (np.real(uimg)).astype('uint8')
# cv2.imwrite('../results/lowwww.jpg', final)
