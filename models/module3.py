import cv2
import sys
import time
import numpy as np
import random as rnd
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QDateTime, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import pyqtSlot


# Import local module
from Tracking_Func import Tack_Object

class ThreadClass(QThread):
    ImageUpdate = pyqtSignal(np.ndarray)
    FPS = pyqtSignal(int)

    def run(self):
        # Load the pre-trained model
        self.model = cv2.dnn.readNet("model_weights.caffemodel", "model_config.prototxt")

        # Open camera
        Capture = cv2.VideoCapture(0)
        Capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        Capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        prev_frame_time = 0
        while True:
            ret, frame_cap = Capture.read()
            flip_frame = cv2.flip(src=frame_cap, flipCode=-1)
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            if ret:
                self.ImageUpdate.emit(flip_frame)
                self.FPS.emit(fps)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("Opencv_PiDash.ui", self)

        # Initialize GUI components and connections
        self.init_ui()

        # Initialize threads
        self.init_threads()

    def init_ui(self):
        # Define GUI components and connections
        pass

    def init_threads(self):
        # Initialize threads for webcam and other tasks
        self.thread_opencv = ThreadClass()
        self.thread_opencv.ImageUpdate.connect(self.opencv_emit)
        self.thread_opencv.FPS.connect(self.get_FPS)
        self.thread_opencv.start()

    @pyqtSlot(np.ndarray)
    def opencv_emit(self, Image):
        # Process and display the frame
        pass

    def get_FPS(self, fps):
        # Update FPS display
        pass

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
