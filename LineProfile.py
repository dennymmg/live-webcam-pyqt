# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 17:10:16 2025

@author: Denny
"""
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class LiveWebcam(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Webcam Feed with Capture and Line Profile")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QGridLayout(self.central_widget)

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Camera not detected")
            sys.exit()

        # Create figure for live feed
        self.live_figure = Figure()
        self.live_ax = self.live_figure.add_subplot(111)
        self.live_ax.set_title("Live Feed")
        self.live_canvas = FigureCanvas(self.live_figure)
        self.layout.addWidget(self.live_canvas, 1, 0)

        # Create figure for captured image
        self.captured_figure = Figure()
        self.captured_ax = self.captured_figure.add_subplot(111)
        self.captured_ax.set_title("Captured Image")
        self.captured_canvas = FigureCanvas(self.captured_figure)
        self.layout.addWidget(self.captured_canvas, 1, 1)
        self.display_capture_prompt()

        # Create figure for grayscale image
        self.gray_figure = Figure()
        self.gray_ax = self.gray_figure.add_subplot(111)
        self.gray_ax.set_title("Grayscale Image")
        self.gray_canvas = FigureCanvas(self.gray_figure)
        self.layout.addWidget(self.gray_canvas, 2, 0)

        # Label to show mouse hover pixel info
        self.gray_hover_label = QLabel("Row: -, Col: -, Intensity: -")
        self.layout.addWidget(self.gray_hover_label, 3, 0)
        
        # Create figure for line profile
        self.line_profile_figure = Figure()
        self.line_profile_ax = self.line_profile_figure.add_subplot(111)
        self.line_profile_ax.set_title("Line Profile")
        self.line_profile_canvas = FigureCanvas(self.line_profile_figure)
        self.layout.addWidget(self.line_profile_canvas, 2, 1)

        self.timer = self.startTimer(30)  # Update every 30 ms

        self.captured_image = None
        self.gray_image = None
        self.line_x = None  # Track the vertical line position

        # Connect mouse hover event
        self.gray_canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

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
        elif event.key() == 16777234:  # Left Arrow Key
            if self.line_x is not None and self.line_x > 0:
                self.line_x -= 1
                self.update_grayscale_display()
        elif event.key() == 16777236:  # Right Arrow Key
            if self.line_x is not None and self.line_x < self.gray_image.shape[1] - 1:
                self.line_x += 1
                self.update_grayscale_display()

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
            self.gray_image = cv2.cvtColor(self.captured_image, cv2.COLOR_RGB2GRAY)
            self.line_x = self.gray_image.shape[1] // 2  # Initialize line at midpoint
            self.update_grayscale_display()

    def update_grayscale_display(self):
        if self.gray_image is not None:
            self.gray_ax.clear()
            self.gray_ax.set_title("Grayscale Image")
            self.gray_ax.imshow(self.gray_image, cmap='gray')
            self.gray_ax.axvline(self.line_x, color='yellow', linestyle='dashed')
            self.gray_canvas.draw()
            line_profile = self.gray_image[:, self.line_x]
            self.line_profile_ax.clear()
            self.line_profile_ax.set_title("Line Profile")
            self.line_profile_ax.plot(line_profile, color='black')
            self.line_profile_canvas.draw()

    def on_mouse_move(self, event):
        if event.xdata is not None and event.ydata is not None and self.gray_image is not None:
            row = int(event.ydata)
            col = int(event.xdata)
            if 0 <= row < self.gray_image.shape[0] and 0 <= col < self.gray_image.shape[1]:
                intensity = self.gray_image[row, col]
                self.gray_hover_label.setText(f"Row: {row}, Col: {col}, Intensity: {intensity}")

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LiveWebcam()
    window.show()
    sys.exit(app.exec_())
