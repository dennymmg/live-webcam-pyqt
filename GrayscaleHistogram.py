# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 17:10:16 2025

@author: Denny
"""
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class LiveWebcam(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Webcam Feed with Capture and Histogram")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QGridLayout(self.central_widget)

        # Create figure for live feed
        self.live_figure = Figure()
        self.live_ax = self.live_figure.add_subplot(111)
        self.live_ax.set_title("Live Feed")
        self.live_canvas = FigureCanvas(self.live_figure)
        self.layout.addWidget(self.live_canvas, 0, 0)

        # Create figure for captured image
        self.captured_figure = Figure()
        self.captured_ax = self.captured_figure.add_subplot(111)
        self.captured_ax.set_title("Captured Image")
        self.captured_canvas = FigureCanvas(self.captured_figure)
        self.layout.addWidget(self.captured_canvas, 0, 1)
        self.display_capture_prompt()

        # Create figure for grayscale image
        self.gray_figure = Figure()
        self.gray_ax = self.gray_figure.add_subplot(111)
        self.gray_ax.set_title("Grayscale Image")
        self.gray_canvas = FigureCanvas(self.gray_figure)
        self.layout.addWidget(self.gray_canvas, 1, 0)

        # Create figure for histogram
        self.hist_figure = Figure()
        self.hist_ax = self.hist_figure.add_subplot(111)
        self.hist_ax.set_title("Grayscale Histogram")
        self.hist_canvas = FigureCanvas(self.hist_figure)
        self.layout.addWidget(self.hist_canvas, 1, 1)

        self.cap = cv2.VideoCapture(0)  # Open webcam
        self.timer = self.startTimer(30)  # Update every 30 ms

        self.captured_image = None
        self.gray_image = None

    def timerEvent(self, event):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            self.live_ax.clear()
            self.live_ax.set_title("Live Feed")
            self.live_ax.imshow(frame)
            self.live_ax.set_xticks([])
            self.live_ax.set_yticks([])
            self.live_canvas.draw()

    def keyPressEvent(self, event):
        if event.key() == ord('C') or event.key() == ord('c'):
            ret, frame = self.cap.read()
            if ret:
                self.captured_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.update_captured_image()
                self.process_image()

    def display_capture_prompt(self):
        self.captured_ax.clear()
        self.captured_ax.set_title("Captured Image")
        self.captured_ax.text(0.5, 0.5, "Press C to Capture", fontsize=12, ha='center', va='center')
        self.captured_ax.set_xticks([])
        self.captured_ax.set_yticks([])
        self.captured_canvas.draw()

    def update_captured_image(self):
        if self.captured_image is not None:
            self.captured_ax.clear()
            self.captured_ax.set_title("Captured Image")
            self.captured_ax.imshow(self.captured_image)
            self.captured_ax.set_xticks([])
            self.captured_ax.set_yticks([])
            self.captured_canvas.draw()

    def process_image(self):
        if self.captured_image is not None:
            # Convert image to grayscale
            self.gray_image = cv2.cvtColor(self.captured_image, cv2.COLOR_RGB2GRAY)
            
            # Display grayscale image
            self.gray_ax.clear()
            self.gray_ax.set_title("Grayscale Image")
            self.gray_ax.imshow(self.gray_image, cmap='gray')
            self.gray_ax.set_xticks([])
            self.gray_ax.set_yticks([])
            self.gray_canvas.draw()
            
            # Compute histogram
            hist = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])
            
            self.hist_ax.clear()
            self.hist_ax.set_title("Grayscale Histogram")
            self.hist_ax.plot(hist, color='black')
            self.hist_ax.set_xlim([0, 256])
            self.hist_canvas.draw()

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LiveWebcam()
    window.show()
    sys.exit(app.exec_())
