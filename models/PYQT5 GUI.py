import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, pyqtSignal, QThread
import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.capture = cv2.VideoCapture(0)
        self.load_model()

    def load_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = fasterrcnn_resnet50_fpn(pretrained=False)
        self.model.to(self.device)
        self.model.eval()
        # Load the pre-trained weights
        self.model.load_state_dict(torch.load("best.pt"))
        self.model.eval()

    def run(self):
        while self._run_flag:
            ret, frame = self.capture.read()
            if ret:
                # Perform inference
                frame_tensor = transforms.ToTensor()(frame).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    prediction = self.model(frame_tensor)[0]

                # Draw bounding boxes on the frame
                for box in prediction["boxes"]:
                    box = [int(coord) for coord in box.tolist()]
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                self.change_pixmap_signal.emit(frame)
        self.capture.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Viewer")
        self.video_thread = VideoThread()
        self.init_ui()

    def init_ui(self):
        self.label = QLabel(self)
        self.label.resize(640, 480)

        start_button = QPushButton('Start', self)
        start_button.clicked.connect(self.start_camera)

        stop_button = QPushButton('Stop', self)
        stop_button.clicked.connect(self.stop_camera)

        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(start_button)
        vbox.addWidget(stop_button)

        self.setLayout(vbox)

    def start_camera(self):
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

    def stop_camera(self):
        self.video_thread.stop()

    def update_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(image))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
