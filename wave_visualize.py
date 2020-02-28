import sys
import threading
import time

import matplotlib
import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QFrame, QGridLayout, QApplication, QPushButton
from matplotlib.animation import TimedAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

matplotlib.use("Qt5Agg")


class HBMonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set window title and main-window shape
        self.setWindowTitle("Heart-Beat Monitoring")
        self.setGeometry(200, 200, 1500, 550)
        # Create main frame and add it to the grid layout
        self.frame = QFrame()
        self.layout = QGridLayout()
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)
        self.fig = MonitorFigure()
        self.layout.addWidget(self.fig, 0, 0, 1, 3)

        self.zoom_in_button = QPushButton()
        self.zoom_in_button.setText("Zoom In")
        self.zoom_in_button.setFixedSize(100, 50)
        self.zoom_in_button.clicked.connect(self.zoomInButtonAction)
        self.layout.addWidget(self.zoom_in_button, 1, 0)

        self.zoom_out_button = QPushButton()
        self.zoom_out_button.setText("Zoom Out")
        self.zoom_out_button.setFixedSize(100, 50)
        self.zoom_out_button.clicked.connect(self.zoomOutButtonAction)
        self.layout.addWidget(self.zoom_out_button, 1, 1)

        self.dataLoop = threading.Thread(name='dataLoop', target=dataSendLoop, daemon=True,
                                         args=(self.addData_callbackFunc,))
        self.dataLoop.start()
        self.show()

    def zoomInButtonAction(self):
        self.fig.zoom += 1
        self.fig.zoom = min(2, self.fig.zoom)
        self.fig.zoom_graph()

    def zoomOutButtonAction(self):
        self.fig.zoom -= 1
        self.fig.zoom = max(-2, self.fig.zoom)
        self.fig.zoom_graph()

    def addData_callbackFunc(self, sample, mlii, v5):
        self.fig.addData(sample, mlii, v5)


class MonitorFigure(FigureCanvasQTAgg, TimedAnimation):
    def __init__(self):
        self.data_sample = []
        self.data_mlii = []
        self.data_v5 = []

        self.offset = 0
        self.xlim = 1000

        self.fig = Figure(figsize=(4, 2), dpi=100, edgecolor="k")
        self.ax_mlii = self.fig.add_subplot(1, 2, 1)
        self.ax_mlii.grid(True, linestyle=":", linewidth=0.8)
        self.ax_mlii.set_facecolor("black")
        self.ax_mlii.set_xlabel("Sample #")
        self.ax_mlii.set_ylabel("MLII")
        self.line_mlii = Line2D([], [], color="lime", linewidth=0.8)
        self.ax_mlii.add_line(self.line_mlii)
        self.ax_mlii.set_ylim(850, 1300)
        self.ax_mlii.set_xlim(0, self.xlim)

        self.ax_v5 = self.fig.add_subplot(1, 2, 2)
        self.ax_v5.grid(True, linestyle=":", linewidth=0.8)
        self.ax_v5.set_facecolor("black")
        self.ax_v5.set_xlabel("Sample #")
        self.ax_v5.set_ylabel("V5")
        self.line_v5 = Line2D([], [], color="seagreen")
        self.ax_v5.add_line(self.line_v5)
        self.ax_v5.set_ylim(850, 1300)
        self.ax_v5.set_xlim(0, self.xlim)

        self.zoom = 0

        FigureCanvasQTAgg.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval=1, blit=True)

    def new_frame_seq(self):
        return iter(range(1))

    def _init_draw(self):
        lines = [self.line_mlii, self.line_v5]
        for line in lines:
            line.set_data([], [])
        return

    def addData(self, sample, mlii, v5):
        self.data_sample.append(sample)
        self.data_mlii.append(mlii)
        self.data_v5.append(v5)
        # if len(self.data_sample) > self.xlim:
        #     self.data_sample.pop(0)
        #     self.data_mlii.pop(0)
        #     self.data_v5.pop(0)

    def set_axis(self):
        self.offset = max(0, (max(self.data_sample) - self.xlim))
        self.ax_mlii.set_xlim(self.offset, self.xlim + self.offset)
        self.ax_v5.set_xlim(self.offset, self.xlim + self.offset)
        self.draw()

    def zoom_graph(self):
        if self.zoom == -2:
            self.xlim = 2000
            self.set_axis()
        elif self.zoom == -1:
            self.xlim = 1500
            self.set_axis()
        elif self.zoom == 0:
            self.xlim = 1000
            self.set_axis()
        elif self.zoom == 1:
            self.xlim = 250
            self.set_axis()
        elif self.zoom == 2:
            self.xlim = 100
            self.set_axis()

    def move(self):
        if max(self.data_sample) > self.xlim:
            self.offset = max(self.data_sample) - self.xlim
            self.ax_mlii.set_xlim(self.offset, self.xlim + self.offset)
            self.ax_v5.set_xlim(self.offset, self.xlim + self.offset)
            self.draw()

    def _step(self, *args):
        # Extends the _step() method for the TimedAnimation class.
        try:
            TimedAnimation._step(self, *args)

        except Exception as e:
            print(e)
            TimedAnimation._stop(self)
            pass

    def _draw_frame(self, framedata):
        self.move()
        self.line_mlii.set_data(self.data_sample, self.data_mlii)
        self.line_v5.set_data(self.data_sample, self.data_v5)
        self._drawn_artists = [self.line_mlii, self.line_v5]


class Communicate(QObject):
    data_signal = pyqtSignal(float, float, float)


def read_data():
    raw_df = pd.read_csv("data_files/raw/100.csv")

    x_sample = raw_df.iloc[:, 0].to_numpy()
    y_mlii = raw_df.iloc[:, 1].to_numpy()
    y_v5 = raw_df.iloc[:, 2].to_numpy()
    return x_sample, y_mlii, y_v5


def dataSendLoop(addData_callbackFunc):
    src = Communicate()
    src.data_signal.connect(addData_callbackFunc)
    read = False
    if not read:
        data_sample, data_mlii, data_v5 = read_data()
        read = True

    for i in range(len(data_sample)):
        time.sleep(0.008)
        src.data_signal.emit(data_sample[i], data_mlii[i], data_v5[i])


app = QApplication(sys.argv)
GUI = HBMonitorWindow()
sys.exit(app.exec_())
