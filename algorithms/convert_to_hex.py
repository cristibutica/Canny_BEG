import numpy as np

def convert_square_to_hex(square_image):
    """
    Converts a square-grid image to a virtual hexagonal-grid image
    based on the linear interpolation pseudo-code from the study.

    This version corrects an IndexError by correctly interpreting
    sub-pixel coordinates vs. pixel indices.

    Args:
        square_image (np.array): The input grayscale image.

    Returns:
        np.array: The new virtual hexagonal image (as uint8).
    """

    # --- 1. Setup and Dimension Checks ---
    square_image = square_image.astype(np.float64)  # Use floats for math
    original_height, original_width = square_image.shape[:2]

    # Crop to nearest multiple of 8
    h = (original_height // 8) * 8
    w = (original_width // 8) * 8

    if h == 0 or w == 0:
        print("Error: Image is too small to be cropped to a multiple of 8.")
        return None

    if h != original_height or w != original_width:
        square_image = square_image[:h, :w]
        print(f"Note: Image cropped to {h}x{w} to be a multiple of 8.")

    M = h // 8
    N = w // 8

    hex_height = 7 * M
    hex_width = 8 * N

    hex_image = np.zeros((hex_height, hex_width), dtype=np.float64)

    # --- 2. Start of Pseudo-code Translation (Corrected) ---
    for j in range(4 * N):
        j1 = j * 2  # Even column index
        j2 = j * 2 + 1  # Odd column index

        for i in range(7 * M):
            # --- // interpolation at even columns ---

            # Sub-pixel Y-coordinates
            Xy1_sub = 8 * i + 3  #
            Ay1_sub = (int(8 * i / 7)) * 7 + 3  #

            # --- THIS IS THE FIX ---
            # Convert sub-pixel coords to actual pixel indices
            Ax1_pix = j1
            Ay1_pix = int(8 * i / 7)
            Bx1_pix = j1
            By1_pix = Ay1_pix + 1  # Pixel B is just the next one down

            # Safety check (prevents reading past bottom edge)
            By1_pix = min(By1_pix, h - 1)
            # --- END OF FIX ---

            f_A = square_image[Ay1_pix, Ax1_pix]
            f_B = square_image[By1_pix, Bx1_pix]

            # Calculate beta using the sub-pixel coordinates
            beta1 = (Xy1_sub - Ay1_sub) / 7.0  #
            hex_image[i, j1] = (1 - beta1) * f_A + beta1 * f_B  #

            # --- //interpolation at odd columns ---

            # Sub-pixel Y-coordinates
            Xy2_sub = 8 * i + 4 + 3  #
            Ay2_sub = (int((8 * i + 4) / 7)) * 7 + 3  #

            # --- THIS IS THE FIX ---
            # Convert sub-pixel coords to actual pixel indices
            Ax2_pix = j2
            Ay2_pix = int((8 * i + 4) / 7)
            Bx2_pix = j2
            By2_pix = Ay2_pix + 1  # Pixel B is just the next one down

            # Safety check (prevents reading past bottom edge)
            By2_pix = min(By2_pix, h - 1)
            # --- END OF FIX ---

            f_A = square_image[Ay2_pix, Ax2_pix]
            f_B = square_image[By2_pix, Bx2_pix]

            # Calculate beta using the sub-pixel coordinates
            beta2 = (Xy2_sub - Ay2_sub) / 7.0  #
            hex_image[i, j2] = (1 - beta2) * f_A + beta2 * f_B  #

    # Return as 8-bit image for display
    return np.clip(hex_image, 0, 255).astype(np.uint8)