import math
import numpy as np
from PIL import ImageFilter


#
arr1 = np.array([[[4, 1, 0], [4, 3, 0], [4, 1, 0]], [[4, 1, 0], [4, 1, 0], [4, 1, 0]], [[4, 1, 0], [4, 1, 0], [4, 1, 0]]])
print(arr1.shape)

gaussianKernel = np.zeros((3, 3))
for x in range(-1, 2):
    for y in range(-1, 2):
        gaussianKernel[x+1, y+1] = np.exp(- (np.power(x, 2) + np.power(y, 2)) / (2 * 1 * 1)) / (
                2 * np.pi * 1 * 1)
print(gaussianKernel)
mask = np.dstack((gaussianKernel, gaussianKernel, gaussianKernel))

print(np.average(arr1, weights=mask, axis=(0, 1)))
# print(np.sum(gaussianKernel))
# print(ImageFilter.Kernel(5,5))

# arr2 = np.array([[0], [1], [1]])

# arr2 = np.array([[0, 1, 0], [1, 0, 1], [1, 0, 0]])
# print(arr1.shape)
# print(np.average(arr1,axis=(0,1)))
# print(np.repeat(arr1, repeats=2, axis=1))

# def smallestN_indices(a, N):
#     idx = a.ravel().argsort()[:N]
#     return np.stack(np.unravel_index(idx, a.shape)).T
# print(smallestN_indices(arr, 3))
#
#
# def fact(n):
#     if n == 0:
#         return 1
#     elif n == 1:
#         return 1
#     return n * fact(n - 1)
#

# if n != 1:
#     x = int((n - 1) / 4) + 1
#     if (n - 1) % 4 == 2 or (n - 1) % 4 == 3: x = -1 * x
# else:
#     x = 0
# if n != 1 & n != 2:
#     y = math.ceil((n - 2) / 4)
#     if (n - 2) % 4 == 2 or (n - 2) % 4 == 0: y = -1 * y
# else:
#     y = 0
# print(x, y)


# def fact(n):
#     if n == 0:
#         return 1
#     elif n == 1:
#         return 1
#     x = (n * fact(n - 1))
#     if x % 10:
#         return x / 10
#     else:
#         return x % 10



# Python3 program check if characters in
# the input string follows the same order
# as determined by characters present in
# the given pattern

# Function to check if characters in the
# input string follows the same order as
# determined by characters present
# in the given pattern


# f = fact(n)
# s = str(f)
#
#
# def baz(s):
#     t = s[len(s) - 1]
#     if t != "0":
#         return int(t)
#     else:
#         baz(s[0, len(s) - 2])


# print(fact(n))

#
# f = math.factorial(n)
#
#
# def baz(f):
#     t = f % 10
#     if (t != 0):
#         print(t)
#         return
#     else:
#         baz(f // 10)
# baz(f)
#
# for i in range(len(s) - 1, 0, -1):
#     if int(s[i]) != 0:
#         print(int(s[i]))
#         break
# binary_num = input()
# javab_sam = input()
# number = 0
# for i in range(len(binary_num)):
#     number += (2 ** i) * int(binary_num[len(binary_num) - i - 1])
#
# for n in range(2, 11):
#     res3 = ''
#     m = 1
#     num = number
#     while m != 0:
#         m = math.floor(num / n)
#         res3 = str(num - m * n) + res3
#         num = m
#     if res3 == javab_sam:
#         print(n)
#     elif n == 10:
#         print("wrong answer")
# x1, y1 = map(int, input().split())
# x2, y2 = map(int, input().split())
# print(x1, y1, x2, y2)
# a = (y1 - y2) / (np.cos(x1) - np.cos(x2))
# b = y1 - a * np.cos(x1)
# print(b)
# print(res3)
# s = "jksdjl"
# print(s[0:3])
