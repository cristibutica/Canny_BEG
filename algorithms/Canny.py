import cv2 as cv

# --- MODIFIED Canny Function ---
max_lowThreshold = 100

title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3

def CannyThreshold(low_threshold, high_threshold, src, src_gray):
    img_blur = cv.blur(src_gray, (3, 3))
    detected_edges = cv.Canny(img_blur, low_threshold, high_threshold, kernel_size)
    mask = detected_edges != 0
    # Create the 3-channel BGR image
    dst = src * (mask[:, :, None].astype(src.dtype))

    # Return the final image
    return dst

