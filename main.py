import sys
from widgets.Test import Test
from widgets.ImagePanel import ImagePanel
from PySide6 import QtWidgets

if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = ImagePanel()
    widget.show()

    sys.exit(app.exec())
