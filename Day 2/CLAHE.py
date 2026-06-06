#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 13:22:22 2026

@author: akshitmudgal
"""

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images/input/lowlight.jpg")


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert RGB -> Grayscale
#
# CLAHE and Histogram Equalization
# are commonly applied on grayscale images.
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# HISTOGRAM EQUALIZATION
# ==================================================
#
# Applies contrast enhancement to the
# entire image at once (global operation).
#
# Good for low contrast images.
#
# Drawback:
# Sometimes over-brightens certain areas.
#
hist_eq = cv2.equalizeHist(gray)


# CLAHE
# ==================================================
#
# CLAHE = Contrast Limited Adaptive
# Histogram Equalization
#
# It improves contrast locally instead
# of processing the whole image together.
#
# Step 1:
# Divide image into small blocks (tiles)
#
# Step 2:
# Perform histogram equalization
# separately on each tile
#
# Step 3:
# Merge all tiles smoothly
#
# Result:
# Better local details
# Less over-enhancement
#

# Create CLAHE object
clahe = cv2.createCLAHE(

    # Limits contrast amplification
    # Smaller value = more natural result
    clipLimit=2,

    # Divide image into 8x8 blocks
    tileGridSize=(8,8)
)

# Apply CLAHE
clahe_img = clahe.apply(gray)


# DISPLAY RESULTS
# ==================================================

plt.figure(figsize=(15,5))

# Original grayscale image
plt.subplot(1,3,1)
plt.imshow(gray, cmap="gray")
plt.title("Original Grayscale")
plt.axis("off")

# Histogram Equalization result
plt.subplot(1,3,2)
plt.imshow(hist_eq, cmap="gray")
plt.title("Histogram Equalization")
plt.axis("off")

# CLAHE result
plt.subplot(1,3,3)
plt.imshow(clahe_img, cmap="gray")
plt.title("CLAHE")
plt.axis("off")

plt.show()