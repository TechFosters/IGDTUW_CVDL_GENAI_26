#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create Binary Image with Noise

img = np.array([
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,255,0,0,0,0],          # Noise pixel
    [0,255,255,255,255,255,255,255,0],
    [0,255,255,255,255,255,255,255,0],
    [0,255,255,255,255,255,255,255,0],
    [0,255,255,255,255,255,255,255,0],
    [0,255,255,255,255,255,255,255,0],
    [0,0,0,255,0,0,0,0,0],          # Noise pixel
    [0,0,0,0,0,0,0,0,0]
], dtype=np.uint8)

# Create 3x3 Kernel
kernel = np.ones((3,3), np.uint8)

# Apply Erosion

eroded = cv2.erode(
    img,
    kernel,
    iterations=1
)


# Display

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(
    img,
    cmap='gray',
    interpolation='nearest'
)
plt.title("Original Image (With Noise)")
plt.xticks(range(img.shape[1]))
plt.yticks(range(img.shape[0]))

plt.subplot(1,2,2)
plt.imshow(
    eroded,
    cmap='gray',
    interpolation='nearest'
)
plt.title("After Erosion")
plt.xticks(range(eroded.shape[1]))
plt.yticks(range(eroded.shape[0]))

plt.tight_layout()
plt.show()

# ==========================================
# Print Matrices
# ==========================================

print("Original Image (0 and 1):\n")
print(img // 255)

print("\nAfter Erosion (0 and 1):\n")
print(eroded // 255)