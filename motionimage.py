import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display images
def show_images(images, titles, figsize=(20, 10)):
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        plt.title(title)
        plt.axis('off')
    plt.show()

# Load input image
image_path = 'fire_sample.jpg'  
image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"Error: Unable to load image from '{image_path}'.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Fire Detection (HSV + YCbCr Mask)
def get_fire_mask(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Split channels
    H, S, V = cv2.split(image_hsv)
    Y, Cr, Cb = cv2.split(image_ycbcr)

    # Fire detection thresholds
    hsv_mask = (H >= 0) & (H <= 50) & (S > 100) & (V > 150)
    ycbcr_mask = (Cr > 135) & (Cb < 120) & (Y > 140)

    # Combine masks with logical AND
    fire_mask = hsv_mask & ycbcr_mask
    return np.uint8(fire_mask * 255)

fire_mask = get_fire_mask(image)

# Applying morphological refinement
kernel = np.ones((5, 5), np.uint8)
fire_mask_refined = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
fire_mask_refined = cv2.morphologyEx(fire_mask_refined, cv2.MORPH_CLOSE, kernel)

# Fire Region Analysis (Contour Detection + Area Calculation)
contours, _ = cv2.findContours(fire_mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter small fire regions
min_area = 500
valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Fire Area Calculation (Assuming 0.01 m² per pixel)
scaling_factor = 0.01
fire_area = sum(cv2.contourArea(cnt) for cnt in valid_contours) * scaling_factor
print(f"Estimated Fire Area: {fire_area:.2f} square meters")

# Drawing contours on the original image
output_image = image_rgb.copy()
for cnt in valid_contours:
    area = cv2.contourArea(cnt) * scaling_factor
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(output_image, f"{area:.2f} m²", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cv2.drawContours(output_image, valid_contours, -1, (255, 0, 0), 2)

# 3. Simulating Fire Spread Using Edge Detection (Motion Approximation)
edges = cv2.Canny(fire_mask_refined, 50, 150)  # Detect edges of the fire regions

# Simulating motion by shifting fire mask slightly
fire_mask_shifted = np.roll(fire_mask_refined, 5, axis=0)

# Fire expansion estimation by comparing shifted mask
fire_expansion = cv2.absdiff(fire_mask_refined, fire_mask_shifted)

#  results
show_images(
    [image_rgb, fire_mask_refined, output_image, edges, fire_expansion],
    ['Original Image', 'Fire Mask', 'Fire Detection + Area', 'Edge Detection (Spread)', 'Simulated Motion']
)

