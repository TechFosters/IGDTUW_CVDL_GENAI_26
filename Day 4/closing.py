#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: akshitmudgal
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((10,10), dtype=np.uint8)

# Big square
img[2:8,2:8] = 255

# Black hole
img[4,4] = 0

kernel = np.ones((3,3), np.uint8)

closed = cv2.morphologyEx(
    img,
    cv2.MORPH_CLOSE,
    kernel
)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original (Hole)")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(closed, cmap='gray')
plt.title("After Closing")
plt.axis("off")

plt.show()