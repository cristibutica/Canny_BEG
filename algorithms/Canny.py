import cv2 as cv
import numpy as np

# --- MODIFIED Canny Function ---
max_lowThreshold = 100

title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3

def my_Canny(image, low_threshold, high_threshold, kernel_size=3):
    """
    Implementation of Canny Edge Detection.
    """
    # 1. Reducing noise by appling Gaussian filter (kernel size = 3)
    img_blur = cv.GaussianBlur(image, (kernel_size, kernel_size), 1.4)

    # 2. Calculus of the Gradient
    # CV_64F = Float 64
    gx = cv.Sobel(img_blur, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(img_blur, cv.CV_64F, 0, 1, ksize=3)

    # Aprroximation of the gradient (magnitude) & the angle resulted by the Sobel operator
    magnitude = cv.magnitude(gx, gy)
    angle = cv.phase(gx, gy, angleInDegrees=True)  # Returns a value between [0 - 360]

    # 3. Non-Maximum Suppression (NMS) - thins thick edges resulted by Sobel operator into single-pixel-wide lines by
    # keeping the local maxima of gradient magnitudes
    # Quantizing the angles in 4 degrees: 0, 45, 90, 135
    # The angle is new between [0 - 180]
    angle = angle % 180

    # Creating masks for the directions
    mask_0 = (angle < 22.5) | (angle >= 157.5)
    mask_45 = (angle >= 22.5) & (angle < 67.5)
    mask_90 = (angle >= 67.5) & (angle < 112.5)
    mask_135 = (angle >= 112.5) & (angle < 157.5)

    # Surppressed image is initially 0 (it has the same size as the magnitude)
    Z = np.zeros_like(magnitude)

    # Shifting the matrix for comparing its neighbours without using for loop (more efficient)
    # Padding the magnitude with zeros to avoid the errors after shifting the matrix
    mag_pad = np.pad(magnitude, ((1, 1), (1, 1)), mode='constant')

    # Neighbours: (x, y) correspond at mag_pad[1:-1, 1:-1]
    # East-West (0 degrees): (1, 0) and (1, 2)
    mag_w = mag_pad[1:-1, :-2]  # West
    mag_e = mag_pad[1:-1, 2:]  # East

    # North-South (90 degrees)
    mag_n = mag_pad[:-2, 1:-1]  # North
    mag_s = mag_pad[2:, 1:-1]  # South

    # Diagonals
    mag_ne = mag_pad[:-2, 2:]  # North-East
    mag_sw = mag_pad[2:, :-2]  # South-West
    mag_nw = mag_pad[:-2, :-2]  # North-West
    mag_se = mag_pad[2:, 2:]  # South-East

    # Appling NMS using the masks
    # We keep the pixel only if it is greater than its neighbours on the gradient direction
    keep_0 = mask_0 & (magnitude >= mag_w) & (magnitude >= mag_e)
    keep_90 = mask_90 & (magnitude >= mag_n) & (magnitude >= mag_s)
    keep_45 = mask_45 & (magnitude >= mag_nw) & (magnitude >= mag_se)
    keep_135 = mask_135 & (magnitude >= mag_ne) & (magnitude >= mag_sw)

    # Keeping the pixel if at least one value is respecting the rule
    nms_mask = keep_0 | keep_90 | keep_45 | keep_135
    Z[nms_mask] = magnitude[nms_mask]

    # 4. Hysteresis Thresholding
    # Identifying strong and potential masks
    strong_mask = Z >= high_threshold

    # The min value for the Low Threshold is 1
    safe_low = max(low_threshold, 1)
    potential_mask = Z >= safe_low

    # We use Connected Components for finding the potential masks connected to the strong masks
    # We label all the weak zones
    num_labels, labels = cv.connectedComponents(potential_mask.astype(np.uint8))

    # Keeping only the labels which have at least one strong pixel
    # labels[strong_mask] will keep only the strong labels
    # np.unique gives a unique ID to the valid labels
    valid_labels = np.unique(labels[strong_mask])

    # Ignoring label 0 (background) if it has been caught
    valid_labels = valid_labels[valid_labels > 0]

    # The final mask (all the pixels that are included in a valid label)
    final_mask = np.isin(labels, valid_labels)

    return (final_mask * 255).astype(np.uint8)

def CannyThreshold(low_threshold, high_threshold, src, src_gray):
    detected_edges = my_Canny(src_gray, low_threshold, high_threshold, kernel_size)
    mask = detected_edges != 0
    # Create the 3-channel RGB image
    dst = src * (mask[:, :, None].astype(src.dtype))

    # Return the final image
    return dst

