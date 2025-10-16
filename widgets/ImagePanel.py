import os
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtWidgets import QFileDialog
from skimage import io, color
from skimage.util import img_as_ubyte

class ImagePanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.original_path = None
        self.gray_path = None
        self.setWindowTitle("Grayscale Converter")

        self.imageUploader = QtWidgets.QPushButton("Choose an image", self)
        self.imageUploader.clicked.connect(self.pickImage)

        self.imageLabel = QtWidgets.QLabel(self)

        self.button = QtWidgets.QPushButton("Convert to Grayscale", self)
        self.button.clicked.connect(self.convertImage)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.button)
        layout.addWidget(self.imageUploader)

    @QtCore.Slot()
    def convertImage(self):
        if self.original_path:
            # Extract file name only (without directories)
            file_name = os.path.basename(self.original_path)
            name, ext = os.path.splitext(file_name)

            # Build grayscale path in local 'images' folder
            self.gray_path = os.path.join("images", f"{name}_gray{ext}")

            # Convert and save if not already existing
            if not os.path.exists(self.gray_path):
                img = io.imread(self.original_path)

                # Handle images with alpha channel (RGBA → RGB)
                if img.shape[-1] == 4:
                    img = img[:, :, :3]

                gray = color.rgb2gray(img)
                gray_rgb = color.gray2rgb(gray)
                io.imsave(self.gray_path, img_as_ubyte(gray_rgb))

            # Display grayscale image
            pixmap = QtGui.QPixmap(self.gray_path)
            self.imageLabel.setPixmap(pixmap)
            self.imageLabel.setScaledContents(True)

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

        # Save original path for later conversion
        self.original_path = file_path

        pixmap = QtGui.QPixmap(file_path)
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.setScaledContents(True)