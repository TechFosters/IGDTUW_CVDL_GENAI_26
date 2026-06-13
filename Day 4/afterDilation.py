import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create Binary Image with Noise
img = np.array([
[0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0],
[0,0,0,255,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0]
], dtype=np.uint8)

# Create 3x3 Kernel
kernel = np.ones((3,3), np.uint8)

# Apply Dilation
dilated = cv2.dilate(
    img,
    kernel,
    iterations=1
)

# Display Images
plt.figure(figsize=(10,5))

# Original Image
plt.subplot(1,2,1)

plt.imshow(
    img,
    cmap='gray',
    interpolation='nearest'
)

plt.title("Original Image")

plt.xticks(range(img.shape[1]))
plt.yticks(range(img.shape[0]))


# Dilated Image

plt.subplot(1,2,2)

plt.imshow(
    dilated,
    cmap='gray',
    interpolation='nearest'
)

plt.title("After Dilation")

plt.xticks(range(dilated.shape[1]))
plt.yticks(range(dilated.shape[0]))

plt.tight_layout()
plt.show()


# ==========================================
# Print Matrix
# ==========================================

print("Original Image (0 and 1):\n")
print(img//255)

print("\nAfter Dilation (0 and 1):\n")
print(dilated//255)