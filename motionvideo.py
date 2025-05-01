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

# Function to calculate fire mask 
def get_fire_mask(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Split channels
    H, S, V = cv2.split(image_hsv)
    Y, Cr, Cb = cv2.split(image_ycbcr)

    # Fire detection thresholds
    hsv_mask = (H >= 0) & (H <= 50) & (S > 100) & (V > 150)
    ycbcr_mask = (Cr > 135) & (Cb < 120) & (Y > 140)

    # Logical AND operation
    fire_mask = hsv_mask & ycbcr_mask
    return np.uint8(fire_mask * 255)

# Initializing video capture
video_path = 'fire_video.mp4'  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError(f"Error opening video: {video_path}")

# Parameters for Optical Flow (Lucas-Kanade)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initializing Background Subtractor (MOG2)
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Reading the first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_fire_mask = get_fire_mask(prev_frame)

scaling_factor = 0.01  # Pixel to real-world area conversion 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converting frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Frame Differencing
    frame_diff = cv2.absdiff(prev_gray, gray)
    _, frame_diff_thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Fire Mask (Color-Based)
    fire_mask = get_fire_mask(frame)

    # Combining frame difference and fire mask to detect moving fire pixels
    moving_fire_mask = cv2.bitwise_and(fire_mask, frame_diff_thresh)

    # Optical Flow (Track Fire Movement)
    fire_points = cv2.findNonZero(moving_fire_mask)
    if fire_points is not None:
        # Calculating optical flow
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, fire_points.astype(np.float32), None, **lk_params)

        for i, (new, old) in enumerate(zip(new_points, fire_points)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)  # Green arrows for motion

    # Background Subtraction (Detecting Dynamic Regions)
    fg_mask = bg_subtractor.apply(frame)
    moving_fire_bg = cv2.bitwise_and(fg_mask, fire_mask)

    # Combining All Motion Detection Methods
    combined_motion_mask = cv2.bitwise_or(moving_fire_mask, moving_fire_bg)

    # Fire Contour Detection (Computing Fire Area)
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    total_fire_pixels = np.sum(fire_mask > 0)
    fire_area = total_fire_pixels * scaling_factor
    print(f"Estimated Fire Area: {fire_area:.2f} square meters")

    # Overlay Fire Region + Motion Vectors
    output_frame = frame.copy()
    for cnt in valid_contours:
        area = cv2.contourArea(cnt) * scaling_factor
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(output_frame, f"{area:.2f} m²", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.drawContours(output_frame, valid_contours, -1, (255, 0, 0), 2)

    # results
    combined_display = np.hstack((frame, cv2.cvtColor(combined_motion_mask, cv2.COLOR_GRAY2BGR)))
    cv2.imshow('Fire Detection + Motion Analysis', combined_display)

    # Updating previous frame
    prev_gray = gray.copy()
    prev_fire_mask = fire_mask.copy()

    # Break loop with 'q' key
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
