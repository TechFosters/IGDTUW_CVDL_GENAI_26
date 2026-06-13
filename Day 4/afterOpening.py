import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create Binary Image with White Noise

img = np.zeros((15,20), dtype=np.uint8)

# Main Object

img[4:11,4:16] = 255

# Random White Noise

img[1,2] = 255

img[2,18] = 255

img[12,5] = 255

img[13,15] = 255

# Create Kernel
kernel = np.ones((3,3), np.uint8)

# Apply Opening # (Erosion followed by Dilation)


opening = cv2.morphologyEx(

    img,

    cv2.MORPH_OPEN,

    kernel

)

# Display


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.imshow(img,cmap="gray",interpolation="nearest")

plt.title("Original Image")

plt.xticks(range(img.shape[1]))

plt.yticks(range(img.shape[0]))

plt.subplot(1,2,2)

plt.imshow(opening,cmap="gray",interpolation="nearest")

plt.title("After Opening")

plt.xticks(range(opening.shape[1]))

plt.yticks(range(opening.shape[0]))



plt.tight_layout()

plt.show()


print("Original:")

print(img//255)



print("\nAfter Opening:")

print(opening//255)