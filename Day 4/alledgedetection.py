#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: akshitmudgal
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read Image
img = cv2.imread("images/input/building.jpg")

# Convert BGR to RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =========================
# SOBEL
# =========================

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

sobel = cv2.addWeighted(
    sobel_x,
    0.5,
    sobel_y,
    0.5,
    0
)

# =========================
# PREWITT
# =========================

kernel_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

kernel_y = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

prewitt_x = cv2.filter2D(gray, -1, kernel_x)
prewitt_y = cv2.filter2D(gray, -1, kernel_y)

prewitt = cv2.addWeighted(
    prewitt_x,
    0.5,
    prewitt_y,
    0.5,
    0
)

# =========================
# LAPLACIAN
# =========================

laplacian = cv2.Laplacian(
    gray,
    cv2.CV_64F
)

laplacian = cv2.convertScaleAbs(
    laplacian
)

# =========================
# CANNY
# =========================

canny = cv2.Canny(
    gray,
    100,
    200
)

# =========================
# DISPLAY ALL
# =========================

plt.figure(figsize=(15,8))

plt.subplot(2,3,1)
plt.imshow(rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(sobel,cmap='gray')
plt.title("Sobel")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(prewitt,cmap='gray')
plt.title("Prewitt")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(laplacian,cmap='gray')
plt.title("Laplacian")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(canny,cmap='gray')
plt.title("Canny")
plt.axis("off")

plt.tight_layout()
plt.show()