#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 13:36:58 2026

@author: akshitmudgal
"""

import cv2
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("images/input/lowlight.jpg")


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert RGB image to Grayscale
#
# Histogram Equalization in OpenCV's equalizeHist()
# works on single channel grayscale images.
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# HISTOGRAM EQUALIZATION
# ==================================================
#
# Goal:
# Improve image contrast.
#
# It redistributes pixel intensities across the
# available range (0 - 255).
#
# Dark images become brighter.
# Low contrast images become clearer.
#
hist_eq = cv2.equalizeHist(gray)

# Create figure
plt.figure(figsize=(15,8))

# ORIGINAL GRAYSCALE IMAGE
# ==================================================
plt.subplot(2,2,1)
plt.imshow(gray, cmap="gray")
plt.title("Original Grayscale")
plt.axis("off")


# HISTOGRAM EQUALIZED IMAGE
# ==================================================
plt.subplot(2,2,2)
plt.imshow(hist_eq, cmap="gray")
plt.title("Histogram Equalized")
plt.axis("off")


# ORIGINAL HISTOGRAM
# ==================================================
#
# ravel() converts 2D image into 1D array
#
# bins=256
# One bin for each intensity value (0-255)
#
plt.subplot(2,2,3)
plt.hist(gray.ravel(),
         bins=256,
         range=[0,256])

plt.title("Original Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")


# EQUALIZED HISTOGRAM
# ==================================================
plt.subplot(2,2,4)
plt.hist(hist_eq.ravel(),
         bins=256,
         range=[0,256])

plt.title("Equalized Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.show()