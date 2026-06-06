#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 11:53:48 2026

@author: akshitmudgal
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("images/input/cat.jpg")


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create a copy so original image remains unchanged
noisy_img = img.copy()

# Percentage of image pixels to corrupt
# 0.20 = 20%
noise_percentage = 0.20

# Get image dimensions
# rows = height
# cols = width
# channels = RGB channels (3)
rows, cols, channels = noisy_img.shape

# Total number of pixels in image

total_pixels = rows * cols

# Number of pixels that will become noise

num_noise_pixels = int(total_pixels * noise_percentage)

print("Total Pixels:", total_pixels)
print("Pixels to be corrupted:", num_noise_pixels)


# IMPORTANT:
# This loop does NOT run on every pixel of the image.
#
# Example:
# Total pixels = 10000
# noise_percentage = 20%
#
# num_noise_pixels = 2000
#
# So this loop runs only 2000 times and randomly selects
# 2000 pixel locations to add Salt noise.
#
# Another loop below runs 2000 times to add Pepper noise.
#
# Therefore:
# Salt iterations   = 2000
# Pepper iterations = 2000
# Total iterations  = 4000
#
# Note:
# Some random locations may repeat, so the number of
# unique corrupted pixels can be less than 4000.


# ADDING SALT NOISE
# ===================================
#
# Salt noise means white dots
# White pixel value = 255
#
# Randomly choose pixel locations
# and make them white
#
for i in range(num_noise_pixels):

    # Random row index
    x = np.random.randint(0, rows)

    # Random column index
    y = np.random.randint(0, cols)

    # Set RGB pixel to white
    # [255,255,255]
    noisy_img[x, y] = 255



# ADDING PEPPER NOISE
# ===================================
#
# Pepper noise means black dots
# Black pixel value = 0
#
# Randomly choose pixel locations
# and make them black
#
for i in range(num_noise_pixels):

    # Random row index
    x = np.random.randint(0, rows)

    # Random column index
    y = np.random.randint(0, cols)

    # Set RGB pixel to black
    # [0,0,0]
    noisy_img[x, y] = 0


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
plt.title("Salt and Pepper Noise Added")
plt.axis("off")

# Display images
plt.show()