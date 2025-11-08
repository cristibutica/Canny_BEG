import os
import numpy as np
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtWidgets import QFileDialog
from skimage import io, color
from skimage.util import img_as_ubyte
from algorithms.convert_to_hex import convert_square_to_hex
import cv2 as cv


# --- Helper Class for Displaying Results ---
class ResultWindow(QtWidgets.QWidget):
    """A simple QWidget for displaying a numpy image array."""

    def __init__(self, title="Result"):
        super().__init__()
        self.setWindowTitle(title)

        # Create a label
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setScaledContents(True)  # Scale image to fit label
        # Tell label to fill all available space
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                       QtWidgets.QSizePolicy.Policy.Ignored)

        # Set layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # No border
        layout.addWidget(self.image_label)

    def set_image(self, numpy_image):
        """Converts a numpy array (BGR or Grayscale) to a QPixmap and displays it."""

        # Make a contiguous copy for QImage
        numpy_image = np.ascontiguousarray(numpy_image)

        # Check if image is grayscale or color
        if len(numpy_image.shape) == 2:
            # Grayscale
            height, width = numpy_image.shape
            bytes_per_line = width
            q_format = QtGui.QImage.Format.Format_Grayscale8
            q_image = QtGui.QImage(numpy_image.data, width, height, bytes_per_line, q_format)

        elif len(numpy_image.shape) == 3:
            # BGR (from OpenCV)
            # Must convert to RGB for QImage
            rgb_image = cv.cvtColor(numpy_image, cv.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_format = QtGui.QImage.Format.Format_RGB888
            q_image = QtGui.QImage(rgb_image.data, width, height, bytes_per_line, q_format)

        else:
            print("Unsupported image format.")
            return

        # Check for null image
        if q_image.isNull():
            print("Failed to create QImage from numpy array.")
            return

        pixmap = QtGui.QPixmap.fromImage(q_image)

        if pixmap.isNull():
            print("Failed to create QPixmap from QImage.")
            return

        self.image_label.setPixmap(pixmap)


# --- MODIFIED Canny Function ---
max_lowThreshold = 100
window_name = ['Edge Map Square', 'Edge Map Hexagonal']  # We can use this for titles
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3


def CannyThreshold(val, src, src_gray):
    low_threshold = val
    img_blur = cv.blur(src_gray, (3, 3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    # Create the 3-channel BGR image
    dst = src * (mask[:, :, None].astype(src.dtype))

    # Return the final image
    return dst


# --- Modified ImagePanel Class ---
class ImagePanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.original_path = None
        self.gray_path = None
        self.setWindowTitle("Hexagonal Converter")

        # --- Keep references to result windows ---
        # This is CRITICAL. If you don't store them,
        # they will be garbage-collected and disappear.
        self.square_window = None
        self.hex_window = None

        self.imageUploader = QtWidgets.QPushButton("Choose an image", self)
        self.imageUploader.clicked.connect(self.pickImage)

        self.imageLabel = QtWidgets.QLabel(self)
        # Fix for main window resizing
        self.imageLabel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                      QtWidgets.QSizePolicy.Policy.Ignored)

        self.button = QtWidgets.QPushButton("Convert to Hexagonal", self)
        self.button.clicked.connect(self.convertImage)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.button)
        layout.addWidget(self.imageUploader)

    @QtCore.Slot()
    def convertImage(self):
        if self.original_path:
            # --- 1. Load/Save Grayscale Image ---
            file_name = os.path.basename(self.original_path)
            name, ext = os.path.splitext(file_name)

            if not os.path.exists("images"):
                os.makedirs("images")

            self.gray_path = os.path.join("images", f"{name}_gray{ext}")
            self.gray_hex_path = os.path.join("images", f"{name}_gray_hex{ext}")

            if not os.path.exists(self.gray_path):
                img = io.imread(self.original_path)
                if img.shape[-1] == 4:
                    img = img[:, :, :3]
                gray = color.rgb2gray(img)
                gray_rgb = color.gray2rgb(gray)
                io.imsave(self.gray_path, img_as_ubyte(gray_rgb))

            # --- 2. Display Grayscale in GUI ---
            pixmap = QtGui.QPixmap(self.gray_path)
            self.imageLabel.setPixmap(pixmap)
            self.imageLabel.setScaledContents(True)

            # --- 3. Convert QPixmap to NumPy for processing ---
            q_image = self.imageLabel.pixmap().toImage()
            q_image = q_image.convertToFormat(QtGui.QImage.Format.Format_RGB888)

            width = q_image.width()
            height = q_image.height()

            ptr = q_image.constBits()
            arr = np.array(ptr).reshape(height, width, 3)
            src_gray_np = cv.cvtColor(arr, cv.COLOR_RGB2GRAY)  # This is the 1-channel gray

            # --- 4. Run Hexagonal Conversion ---
            print("Converting to hexagonal structure...")
            hex_image = convert_square_to_hex(src_gray_np)  # This is 1-channel gray

            if hex_image is None:
                print("Conversion failed.")
                return

            io.imsave(self.gray_hex_path, img_as_ubyte(hex_image))

            # Load the 3-channel BGR images needed for the Canny mask
            src_gray_bgr = cv.imread(self.gray_path)
            src_hex_bgr = cv.imread(self.gray_hex_path)

            print("Conversion complete. Displaying images.")

            # --- 5. Call processing function ---
            # Process the square image
            square_canny_result = CannyThreshold(5, src_gray_bgr, src_gray_np)

            # Process the hex image
            hex_canny_result = CannyThreshold(5, src_hex_bgr, hex_image)

            # --- 6. Display results in new Qt Windows ---

            # Close old windows if they exist
            if self.square_window:
                self.square_window.close()
            if self.hex_window:
                self.hex_window.close()

            # Create, resize, and show the SQUARE result window
            self.square_window = ResultWindow(title=window_name[0])
            self.square_window.set_image(square_canny_result)
            self.square_window.resize(800, 600)  # Set your desired size!
            self.square_window.show()

            # Create, resize, and show the HEXAGONAL result window
            self.hex_window = ResultWindow(title=window_name[1])
            self.hex_window.set_image(hex_canny_result)
            self.hex_window.resize(800, 600)  # Set your desired size!
            self.hex_window.show()

    @QtCore.Slot()
    def pickImage(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Pick an Image",
            ".",
            "Images (*.webp *.png *.jpg *.jpeg)"
        )
        if not file_path:
            return

        self.original_path = file_path

        pixmap = QtGui.QPixmap(file_path)
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.setScaledContents(True)

# (Your main.py or startup code would go here)