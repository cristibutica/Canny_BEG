# --- Helper Class for Displaying Results ---
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
import cv2 as cv

from algorithms.Canny import CannyThreshold


class ResultWindow(QtWidgets.QWidget):
    """A simple QWidget for displaying a numpy image array."""

    def __init__(self, src_gray_bgr, src_gray_np, title="Result", structure_type="square"):
        super().__init__()
        self.src_gray_bgr = src_gray_bgr
        self.src_gray_np = src_gray_np
        self.structure_type = structure_type
        self.setWindowTitle(title)
        self.setMinimumSize(400, 300)  # Set a minimum size

        # --- 1. Main Image Label ---
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setScaledContents(True)
        # Tell label to fill all available space
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                       QtWidgets.QSizePolicy.Policy.Ignored)

        # --- 2. Controls Layout ---
        # We'll use a QGridLayout for a clean label-slider-value alignment
        controls_layout = QtWidgets.QGridLayout()

        # --- Low Threshold ---
        low_label = QtWidgets.QLabel("Low Threshold:")
        self.low_threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.low_threshold_slider.setRange(0, 255)
        self.low_threshold_slider.setValue(0)

        self.low_value_label = QtWidgets.QLabel("0")
        # Set a fixed width so the layout doesn't jump
        self.low_value_label.setFixedWidth(30)
        self.low_value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        # Connect the slider's signal to the label's "setNum" slot
        self.low_threshold_slider.valueChanged.connect(self.low_value_label.setNum)
        self.low_threshold_slider.valueChanged.connect(self.change_slider_val)

        # Add to grid: (widget, row, column)
        controls_layout.addWidget(low_label, 0, 0)
        controls_layout.addWidget(self.low_threshold_slider, 0, 1)
        controls_layout.addWidget(self.low_value_label, 0, 2)

        # --- High Threshold ---
        high_label = QtWidgets.QLabel("High Threshold:")
        self.high_threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.high_threshold_slider.setRange(0, 255)
        self.high_threshold_slider.setValue(0)

        self.high_value_label = QtWidgets.QLabel("0")
        self.high_value_label.setFixedWidth(30)  # Set a fixed width
        self.high_value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        # Connect the slider
        self.high_threshold_slider.valueChanged.connect(self.high_value_label.setNum)
        self.high_threshold_slider.valueChanged.connect(self.change_slider_val)

        # Add to grid
        controls_layout.addWidget(high_label, 1, 0)
        controls_layout.addWidget(self.high_threshold_slider, 1, 1)
        controls_layout.addWidget(self.high_value_label, 1, 2)

        # Make the slider column (1) stretchable
        controls_layout.setColumnStretch(1, 1)

        # --- 3. Main Window Layout ---
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)  # Add some padding
        main_layout.setSpacing(10)  # Add spacing between widgets

        main_layout.addWidget(self.image_label)
        # Add the grid layout directly
        main_layout.addLayout(controls_layout)

        # Give all extra vertical space to the image
        main_layout.setStretchFactor(self.image_label, 1)
        # The controls layout gets no extra space
        main_layout.setStretchFactor(controls_layout, 0)


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


    def change_slider_val(self):
        low_threshold = self.low_threshold_slider.value()
        high_threshold = self.high_threshold_slider.value()

        canny_result = CannyThreshold(low_threshold, high_threshold, self.src_gray_bgr, self.src_gray_np)

        self.set_image(canny_result)


