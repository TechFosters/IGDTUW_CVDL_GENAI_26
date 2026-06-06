#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:57:48 2026

@author: akshitmudgal
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("images/input/cat.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


noisy_img = img.copy()

noise_percentage = 0.05

rows,cols,channels = noisy_img.shape

total_pixels = rows * cols

num_noise_pixels = int(total_pixels * noise_percentage)

print("Total Pixels: ", total_pixels)

print("Pixels to be corrupted: ", num_noise_pixels)

#sprinkling the salt

for i in range(num_noise_pixels):
    
    x = np.random.randint(0, rows)
    
    y = np.random.randint(0, cols)
    
    noisy_img[x,y] = 255
    
    #add pepper
for i in range(num_noise_pixels):
    
    x = np.random.randint(0, rows)
    
    y = np.random.randint(0, cols)
    
    noisy_img[x,y] = 0

# MEDIAN FILTER
# ==================================================
#
# Kernel Size = 5x5
#
# Median Filter does NOT calculate average.
#
# Steps:
# 1. Take neighboring pixels
# 2. Sort them
# 3. Pick middle value (Median)
# 4. Replace center pixel
#
# Excellent for Salt & Pepper Noise
# because extreme values (0 and 255)
# get removed automatically.
#
    
median_filtered = cv2.medianBlur(noisy_img, 5)
    
plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.imshow(img)
plt.title("OG Image")


plt.subplot(1,3,2)
plt.imshow(noisy_img)
plt.title("Salt and Pepper Noise Added")

plt.subplot(1,3,3)
plt.imshow(median_filtered)
plt.title("median filter Added")
plt.show()