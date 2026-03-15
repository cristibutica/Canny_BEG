import os
import numpy as np
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtWidgets import QFileDialog
from skimage import io, color
from skimage.util import img_as_ubyte

from algorithms.Canny import CannyThreshold
from algorithms.convert_to_hex import convert_square_to_hex
import cv2 as cv

from widgets.ResultWindow import ResultWindow

window_name = ['Edge Map Square', 'Edge Map Hexagonal']  # We can use this for titles

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
            square_canny_result = CannyThreshold(0, 0, src_gray_bgr, src_gray_np)

            # Process the hex image
            hex_canny_result = CannyThreshold(0, 0, src_hex_bgr, hex_image)

            # --- 6. Display results in new Qt Windows ---

            # Close old windows if they exist
            if self.square_window:
                self.square_window.close()
            if self.hex_window:
                self.hex_window.close()

            # Create, resize, and show the SQUARE result window
            self.square_window = ResultWindow(src_gray_bgr, src_gray_np, title="Canny Result Square", structure_type="square")
            self.square_window.set_image(square_canny_result)
            self.square_window.resize(800, 600)
            self.square_window.show()

            self.hex_window = ResultWindow(src_hex_bgr, hex_image, title="Canny Result Hexagonal", structure_type="hex")
            self.hex_window.set_image(hex_canny_result)
            self.hex_window.resize(800, 600)
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