import os
from PySide6 import QtWidgets, QtGui, QtCore
from skimage import io, color
from skimage.util import img_as_ubyte

class ImagePanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowFilePath("Grayscale Converter")
        self.original_path = "images/mountains.webp"
        self.gray_path = "images/mountains_gray.png"  # new file we will create

        # Load original image
        pixmap = QtGui.QPixmap(self.original_path)

        self.imageLabel = QtWidgets.QLabel(self)
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.setScaledContents(True)

        self.button = QtWidgets.QPushButton("Convert to Grayscale", self)
        self.button.clicked.connect(self.convertImage)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.button)

    @QtCore.Slot()
    def convertImage(self):
        # Create grayscaled image if it doesn't exist yet
        if not os.path.exists(self.gray_path):
            img = io.imread(self.original_path)
            gray = color.rgb2gray(img)  # convert to grayscale
            gray_rgb = color.gray2rgb(gray)  # expand to 3 channels because QPixmap requires it
            io.imsave(self.gray_path, img_as_ubyte(gray_rgb))  # save as PNG

        # Load the new file into pixmap
        pixmap = QtGui.QPixmap(self.gray_path)
        self.imageLabel.setPixmap(pixmap)
