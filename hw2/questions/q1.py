import math

import numpy as np
import cv2
import matplotlib.pylab as plt

path = "../images/flowers.blur.png"
image = cv2.imread(path)
image = image.astype(float)
height, width, channels = image.shape


def generateGaussianKernel(height, width, sigma, mu1, mu2):
    gaussianKernel = np.zeros((height, width))
    for x in range(height):
        for y in range(width):
            gaussianKernel[x, y] = np.exp(- (np.power(x - mu1, 2) + np.power(y - mu2, 2)) / (2 * sigma * sigma)) / (
                    2 * np.pi * sigma * sigma)
    return gaussianKernel


def generateLaplacianFilter(height, width, sigma, mu1, mu2):
    laplacianKernel = np.zeros((height, width))
    c = -1 / (np.pi * sigma ** 4)
    for x in range(height):
        for y in range(width):
            m = (np.power(x - mu1, 2) + np.power(y - mu2, 2)) / (2 * sigma * sigma)
            laplacianKernel[x, y] = c * (1 - m) * np.exp(-m)
    return laplacianKernel


def scale(array):
    min, max = array.min(), array.max()
    return ((array - min) * (1 / (max - min) * 255)).astype('uint8')


"""first part:"""
sigma, k = 1, 2
gaussianKernel = generateGaussianKernel(2 * k + 1, 2 * k + 1, sigma, k, k)
plt.imshow(gaussianKernel)
plt.colorbar()
plt.savefig('../results/res01.jpg')
plt.show()

blurred = cv2.filter2D(image, -1, gaussianKernel)
cv2.imwrite('../results/res02.jpg', blurred)

unSharpMask = cv2.subtract(image, blurred)
cv2.imwrite('../results/res03.jpg', unSharpMask)

alpha = 4
sharpenedMask = cv2.add(image, cv2.multiply(unSharpMask, alpha))
cv2.imwrite('../results/res04.jpg', sharpenedMask)

"""second part:"""
sigma, k = 1, 3
laplacianKernel = generateLaplacianFilter(2 * k + 1, 2 * k + 1, sigma, k, k)
plt.imshow(laplacianKernel)
plt.colorbar()
plt.savefig('../results/res05.jpg')
plt.show()

unSharpMask = cv2.filter2D(image, -1, laplacianKernel)
cv2.imwrite('../results/res06.jpg', unSharpMask)

k = 4
sharpenedMask = cv2.subtract(image, cv2.multiply(unSharpMask, k))
cv2.imwrite('../results/res07.jpg', sharpenedMask)
#
""" third part:"""
imgFft = np.fft.fft2(image, axes=(0, 1))
shiftedImg = np.fft.fftshift(imgFft)
amplitude = np.abs(shiftedImg)
logAmplitude = np.log(amplitude)
cv2.imwrite('../results/res08.jpg', logAmplitude)

laplacianKernel = generateLaplacianFilter(height, width, 200, int(image.shape[0] / 2),
                                          int(image.shape[1] / 2))
H_hp = np.fft.fft2(laplacianKernel)
shiftedH_hp = np.fft.fftshift(H_hp)
mag = np.log(np.abs(shiftedH_hp))
plt.savefig('../results/res09.jpg')

k = 10
## calculate (1+k*H_hp).F
KH = np.array(1 + k * shiftedH_hp)
result = []
for i in range(0, 3):
    result.insert(0, np.multiply(KH, shiftedImg[:, :, 2 - i]))
res = np.dstack(result)

mag = np.log(np.abs(res))
cv2.imwrite('../results/res10.jpg', mag)

# final image
ishifted = np.fft.ifftshift(res)
final = np.fft.ifft2(ishifted, axes=(0, 1))
final = scale(np.real(final))

cv2.imwrite('../results/res11.jpg', final)


""" fourth part:"""
res12 = np.zeros((height, width, 3), dtype=np.complex_)
for c in range(3):
    for u in range(height):
        for v in range(width):
            res12[u, v, c] = 4 * np.power(math.pi, 2) * (np.power(height / 2 - u, 2) + np.power(width / 2 - v, 2)) * \
                             shiftedImg[
                                 u, v, c]

cv2.imwrite('../results/res12.jpg', np.real(res12))

ishifted = np.fft.ifftshift(res12)
res13 = np.fft.ifft2(ishifted, axes=(0, 1))
cv2.imwrite('../results/res13.jpg', np.real(res13))

k = 0.0000001
res14 = image + k * res13
cv2.imwrite('../results/res14.jpg', scale(np.real(res14)))
