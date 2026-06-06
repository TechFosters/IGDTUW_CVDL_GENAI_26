#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:04:26 2026

@author: akshitmudgal
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image from disk
img = cv2.imread("images/input/cat.jpg")

# OpenCV loads image in BGR format
# Convert to RGB for Matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Mean of Gaussian noise
# 0 means noise is centered around zero
mean = 0

# Standard deviation
# Controls strength of speckle noise
sigma = 1.3

# Generate random Gaussian noise
#
# Shape is same as image
# Example values:
# 0.5, -0.2, 1.1, -0.8, ...
noise = np.random.normal(mean, sigma, img.shape)

# ==========================================
# SPECKLE NOISE
# ==========================================
#
# Formula:
#
# New Pixel = Original Pixel + (Original Pixel × Noise)
#
# Unlike Gaussian noise:
#
# Gaussian:
# New Pixel = Original Pixel + Noise
#
# Speckle:
# New Pixel = Original Pixel + Original Pixel × Noise
#
# Therefore noise depends on pixel intensity.
#
# Bright pixels receive larger changes.
# Dark pixels receive smaller changes.
#
# This is called Multiplicative Noise.
#
speckle_img = img + img * noise

# Keep values between 0 and 255
speckle_img = np.clip(speckle_img, 0, 255)

# Convert back to image datatype
speckle_img = speckle_img.astype(np.uint8)

# Create figure
plt.figure(figsize=(10,5))

# Original image
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

# Speckle noisy image
plt.subplot(1,2,2)
plt.imshow(speckle_img)
plt.title("Speckle Noise Added")
plt.axis("off")

plt.show()