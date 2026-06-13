#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: akshitmudgal
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create black image
img = np.zeros((10,10), dtype=np.uint8)

# Main square
img[2:8,2:8] = 255

# Add noise pixels
img[0,0] = 255
img[1,8] = 255
img[8,1] = 255
img[9,9] = 255
img[4,9] = 255

# Kernel
kernel = np.ones((3,3), np.uint8)

# Erosion
eroded = cv2.erode(img, kernel, iterations=1)

# Plot
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original + Noise")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(eroded, cmap='gray')
plt.title("After Erosion")
plt.axis("off")

plt.show()