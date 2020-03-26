import numpy as np
import pandas as pd


class DatasetGenerator:
    def __init__(self, raw_path, annot_path, beat_width):
        self.beat_width = beat_width

        self.raw_data = pd.read_csv(raw_path).to_numpy()
        self.annot = pd.read_csv(annot_path).to_numpy()

        self.pos_margin = max(self.beat_width, 64) // 64
        self.neg_margin = -(max(self.beat_width, 64) // 64)

        self.x = self.get_x()
        self.y = self.get_y()

    def get_x(self):
        x = []
        for beat in self.annot[self.pos_margin:self.neg_margin, 2]:
            x.append(self.raw_data[int(beat) - self.beat_width: int(beat) + self.beat_width, 1])
        return np.array(x)

    def get_y(self):
        y = []
        for beat in self.annot[self.pos_margin:self.neg_margin, 3]:
            y.append(beat)
        return np.array(y)


b_gen = DatasetGenerator(raw_path="data_files/raw/100.csv",
                         annot_path="data_files/annotations/csv/100annotations.csv",
                         beat_width=64)
