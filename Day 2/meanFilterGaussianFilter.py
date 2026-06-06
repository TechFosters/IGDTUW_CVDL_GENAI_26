#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:31:19 2026

@author: akshitmudgal
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("images/input/cat.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Parameters for Gaussian Noise
mean = 0
sigma = 50

# Generate random Gaussian noise

# Shape = same as image
# Example values:
# -20, +35, -10, +50, etc.
noise = np.random.normal(mean, sigma, img.shape)

# Add noise to image
#
# New Pixel = Original Pixel + Noise
noisy_img = img + noise

# Pixel values must remain between 0 and 255
# Then convert back to uint8 image datatype
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)


# MEAN FILTER (AVERAGING FILTER)
# =============================================
#
# Kernel Size = 9x9
#
# For every pixel:
# Take 9x9 neighboring pixels
# Calculate their average
# Replace center pixel with that average
#
# This smooths the image and reduces noise
#
# Drawback:
# Edges and details become blurry
#
mean_filtered = cv2.blur(noisy_img, (9,9))

# GAUSSIAN FILTER
# ==================================================
#
# Kernel Size = 9x9
#
# Unlike Mean Filter:
# Every neighbor does NOT get equal importance.
#
# Center pixels get higher weight.
# Far pixels get lower weight.
#
# Therefore:
# - Noise is reduced
# - Edges are preserved better
#
# sigmaX = 50
#
# Larger sigma:
# More smoothing
#
# Smaller sigma:
# Less smoothing
#
gaussian_filtered = cv2.GaussianBlur(
    noisy_img,
    (9,9),
    50
)

# Create figure window
plt.figure(figsize=(10,5))

# Original Image
plt.subplot(1,4,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

# Noisy Image
plt.subplot(1,4,2)
plt.imshow(noisy_img)
plt.title("Gaussian Noise Added")
plt.axis("off")

# Mean Filter Output
plt.subplot(1,4,3)
plt.imshow(mean_filtered)
plt.title("Mean Filter")
plt.axis("off")

# Gaussian Filter Output
plt.subplot(1,4,4)
plt.imshow(gaussian_filtered)
plt.title("Gaussian Filter")
plt.axis("off")

plt.show()