import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load input image
image = cv2.imread('fire_sample.png')  # Replace with your image
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert image to HSV and YCbCr color spaces
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image_ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# Split channels
H, S, V = cv2.split(image_hsv)
Y, Cr, Cb = cv2.split(image_ycbcr)

# Fire detection thresholds (based on given criteria)
hsv_mask = (H >= 0) & (H <= 50) & (S > 100) & (V > 150)
ycbcr_mask = (Cr > 135) & (Cb < 120) & (Y > 140)

# Combine masks using logical AND
fire_mask = hsv_mask & ycbcr_mask

# Create a fire-highlighted image
fire_highlighted = image_rgb.copy()
fire_highlighted[fire_mask] = [255, 0, 0]  # Red color overlay for detected fire pixels

# Visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(fire_mask, cmap='gray')
plt.title('Fire Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(fire_highlighted)
plt.title('Fire Segmentation')
plt.axis('off')

plt.tight_layout()
plt.show()

