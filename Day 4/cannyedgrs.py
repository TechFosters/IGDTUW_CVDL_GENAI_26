#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: akshitmudgal
"""

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images/input/house.jpg", 0) #directly reads grayscale this way, you can also use cvtColor

edges1 = cv2.Canny(img, 50, 150)
edges2 = cv2.Canny(img, 100, 200)
edges3 = cv2.Canny(img, 150, 300)

plt.figure(figsize=(15,5))

plt.subplot(1,4,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(edges1, cmap='gray')
plt.title("50-150")
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(edges2, cmap='gray')
plt.title("100-200")
plt.axis('off')

plt.subplot(1,4,4)
plt.imshow(edges3, cmap='gray')
plt.title("150-300")
plt.axis('off')

plt.tight_layout()
plt.show()