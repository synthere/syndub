# -*- coding: utf-8 -*-
import sys, os

#pyside6-uic.exe  -o .\mainwin.py .\mainwin.ui

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QPixmap, QPalette, QBrush, QIcon, QGuiApplication

from typing import Iterator, Union
#pyinstaller dubbing.spec

def get_base_path():
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return base_path

class StartWindow(QtWidgets.QWidget):
    def __init__(self):
        super(StartWindow, self).__init__()
        self.width = 1200
        self.height = 700
        # no frame and title
        self.setWindowFlags(Qt.FramelessWindowHint)
        # background pic
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(QPixmap(os.path.join(get_base_path(), "./resource/start.png"))))#os.path.join(get_base_path(), "./resource/start.png")
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        self.setWindowIcon(QIcon(os.path.join(get_base_path(), "./resource/app.png")))
        v1 = QtWidgets.QVBoxLayout()
        v1.addStretch(1)
        h1 = QtWidgets.QHBoxLayout()
        v1.addLayout(h1)
        v1.addStretch(0)

        h1.addStretch(1)
        self.lab = QtWidgets.QLabel()
        VERSION = 1.0
        self.lab.setText(f"SynthereDub {VERSION} is Loading...")
        self.lab.setStyleSheet("""font-size:16px;color:#fff;text-align:center""")
        h1.addWidget(self.lab)
        h1.addStretch(0)
        self.setLayout(v1)
        print("init finish")
        # winsize
        self.resize(713, 368)
        self.show()
        self.center()
        print("to demain")
        QTimer.singleShot(200, self.run)

    def run(self):
        # show main
        print("to run main")
        app.setApplicationName("SynthereDub")
        app.setStyle("Fusion")

        app.setPalette(get_platt())
        app.setStyleSheet(
            "QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }"
        )
        """
        with open('./resource/style.qss', 'r', encoding='utf-8') as f:
            app.setStyleSheet(f.read())
        """
        app.setWindowIcon(QIcon(os.path.join(get_base_path(), "./resource/app.png")))
        import time
        st = time.time()
        from dubui import MainWindow
        mainw = MainWindow()
        et = time.time()
        print(f'start cost：{et - st}')
        self.close()

    def center(self):
        screen = QGuiApplication.primaryScreen()
        screen_resolution = screen.geometry()
        self.width, self.height = screen_resolution.width(), screen_resolution.height()
        self.move(QPoint(int((self.width - 559) / 2), int((self.height - 300) / 2)))

def get_platt():
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

    return palette

if __name__ == "__main__":

    app = QApplication(sys.argv)
    """
    app.setApplicationName("SynthereDub")
    app.setStyle("Fusion")

    app.setPalette(get_platt())
    app.setStyleSheet(
        "QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }"
    )
    app.setWindowIcon(QIcon(os.path.join(get_base_path(), "./resource/app.png")))
    """
    try:
        startwin = StartWindow()
    except Exception as e:
        print(f"error:{str(e)}")
    sys.exit(app.exec())

