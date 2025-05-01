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

# Loading Image 
image_path = 'fire_sample.jpg'
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocessing (Convert to HSV & YCbCr + Blur)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image_ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Fire Pixel Segmentation (HSV & YCbCr Masks)
H, S, V = cv2.split(image_hsv)
Y, Cr, Cb = cv2.split(image_ycbcr)

hsv_mask = (H >= 0) & (H <= 50) & (S > 100) & (V > 150)
ycbcr_mask = (Cr > 135) & (Cb < 120) & (Y > 140)

# Combining HSV and YCbCr masks using AND
fire_mask = hsv_mask & ycbcr_mask
fire_mask = np.uint8(fire_mask * 255)  # Convert to uint8

show_images([hsv_mask * 255, ycbcr_mask * 255, fire_mask], ['HSV Mask', 'YCbCr Mask', 'Combined Fire Mask'])

# Noise Removal (Morphological Operations)
kernel = np.ones((5, 5), np.uint8)
fire_mask_refined = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
fire_mask_refined = cv2.morphologyEx(fire_mask_refined, cv2.MORPH_CLOSE, kernel)

show_images([fire_mask, fire_mask_refined], ['Original Fire Mask', 'Refined Fire Mask'])

# Contour Detection and Refinement
contours, _ = cv2.findContours(fire_mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_area = 500
valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Drawing valid fire regions on the original image
fire_detected_image = image_rgb.copy()
cv2.drawContours(fire_detected_image, valid_contours, -1, (255, 0, 0), 2)  # Draw contours in blue

show_images([image_rgb, fire_detected_image], ['Original Image', 'Fire Detection Output'])

# Fire Area Calculation
total_fire_pixels = np.sum(fire_mask_refined > 0)
print(f"Total Fire Pixels: {total_fire_pixels}")

# Assuming a scaling factor (e.g., 1 pixel = 0.01 m² for demonstration purposes)
scaling_factor = 0.01  # Adjust based on your camera calibration
fire_area = total_fire_pixels * scaling_factor

print(f"Estimated Fire Area: {fire_area:.2f} square meters")

# Overlay Fire Region with Area on Image
overlay = image_rgb.copy()

for cnt in valid_contours:
    # Draw contour and compute area for each region
    cv2.drawContours(overlay, [cnt], -1, (255, 0, 0), 2)  # Blue contours
    area = cv2.contourArea(cnt) * scaling_factor
    # Place area text near the contour
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(overlay, f"{area:.2f} m²", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

show_images([overlay], ['Fire Region with Estimated Area'])
