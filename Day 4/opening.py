#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: akshitmudgal
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((10,10), dtype=np.uint8)

img[2:8,2:8] = 255

img[0,0] = 255
img[1,8] = 255
img[8,1] = 255
img[9,9] = 255
img[4,9] = 255

kernel = np.ones((3,3), np.uint8)

opened = cv2.morphologyEx(
    img,
    cv2.MORPH_OPEN,
    kernel
)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original + Noise")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(opened, cmap='gray')
plt.title("After Opening")
plt.axis("off")

plt.show()