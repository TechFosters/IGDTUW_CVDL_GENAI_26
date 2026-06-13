#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: akshitmudgal
"""

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("cat.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

combined = cv2.addWeighted(
    sobel_x,
    0.5,
    sobel_y,
    0.5,
    0
)

plt.imshow(combined, cmap='gray')
plt.title("Combined Sobel")
plt.axis("off")
plt.show()

