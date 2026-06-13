
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Create Binary Image
# Two objects with a small gap
# ==========================================

img = np.zeros((15, 20), dtype=np.uint8)

# Left object
img[5:10, 2:8] = 255

# Right object
img[5:10, 11:17] = 255

# ==========================================
# Create 3x3 Kernel
# ==========================================

kernel = np.ones((3,3), np.uint8)

# ==========================================
# Apply Dilation
# ==========================================

dilated = cv2.dilate(
    img,
    kernel,
    iterations=2   # Try 1, 2, or 3
)

# ==========================================
# Display
# ==========================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(
    img,
    cmap="gray",
    interpolation="nearest"
)
plt.title("Original Image")
plt.xticks(range(img.shape[1]))
plt.yticks(range(img.shape[0]))

plt.subplot(1,2,2)
plt.imshow(
    dilated,
    cmap="gray",
    interpolation="nearest"
)
plt.title("After Dilation")
plt.xticks(range(dilated.shape[1]))
plt.yticks(range(dilated.shape[0]))

plt.tight_layout()
plt.show()

# ==========================================
# Print Matrix
# ==========================================

print("Original Image:")
print(img//255)

print("\nAfter Dilation:")
print(dilated//255)