#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 10:50:51 2026

@author: akshitmudgal
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image from folder
img = cv2.imread("images/input/cat.jpg")

# OpenCV loads image in BGR format
# Matplotlib expects RGB format
# So we convert BGR → RGB for correct colors
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Mean = average value of noise
# 0 means noise is centered around zero
# (some pixels increase, some decrease)
mean = 0

# Standard deviation controls noise strength
# Larger sigma = more visible noise
sigma = 75

# Generate Gaussian (Normal) noise
#
# np.random.normal(mean, sigma, shape)
#
# mean  = center of distribution
# sigma = spread of distribution
# shape = same size as image
#
# Output contains random values like:
# -20, +15, -5, +100, etc.
noise = np.random.normal(mean, sigma, img.shape)

# Add noise to every pixel
#
# New Pixel = Original Pixel + Noise
#
# Example:
# Pixel = 120
# Noise = +30
# New Pixel = 150
#
# Pixel = 120
# Noise = -50
# New Pixel = 70
noisy_img = img + noise

# Some values may go below 0 or above 255
#
# Example:
# 250 + 50 = 300 
# 20 - 40 = -20   
#
# np.clip() forces values into valid range
# 0   <= pixel <= 255
#
# astype(np.uint8) converts back to image datatype
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

# Create figure window
plt.figure(figsize=(10,5))

# Show original image
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

# Show noisy image
plt.subplot(1,2,2)
plt.imshow(noisy_img)
plt.title("Gaussian Noise Added")
plt.axis("off")

# Display both images
plt.show()