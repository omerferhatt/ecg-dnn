import os
import glob

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


class DatasetGenerator:
    def __init__(self, raw_path, annot_path, beat_width, random_seed=0):
        self.beat_width = beat_width
        if random_seed == 0:
            self.random_seed = np.random.randint(0, 1000)
        else:
            self.random_seed = random_seed

        self.raw_path = raw_path
        self.annot_path = annot_path
        self.raw_files, self.annot_files, self.total_patient = self.get_all_csv()

        self.margin = (max(self.beat_width, 64) // 64) + 1

        self.raw_data, self.annot_data = self.get_data_to_arr()
        self.shuffle_data()

    def get_all_csv(self):
        try:
            raw_files = glob.glob(os.path.join(self.raw_path, "*"))
            annot_files = glob.glob(os.path.join(self.annot_path, "*"))

            if len(raw_files) != len(annot_files):
                raise ValueError
            else:
                total_patient = len(raw_files)

            return raw_files, annot_files, total_patient

        except ValueError as e:
            print(e, "Raw files don't match with annotations")

    def get_data_to_arr(self):
        raw_arr = []
        annot_arr = []

        for patient in range(self.total_patient):
            raw_data = pd.read_csv(self.raw_files[patient]).to_numpy()
            annot_data = pd.read_csv(self.annot_files[patient]).to_numpy()

            for beat in annot_data[self.margin:-self.margin, [2, 3]]:
                raw = np.fft.fft(raw_data[int(beat[0]) - self.beat_width: int(beat[0]) + self.beat_width, 1])
                raw_arr.append(raw)
                annot_arr.append(beat[1])

        raw_arr = np.array(raw_arr)
        raw_arr = raw_arr[:, :, np.newaxis]

        le = LabelEncoder()
        annot_arr = le.fit_transform(np.array(annot_arr))

        return raw_arr, annot_arr

    def shuffle_data(self):
        np.random.seed(self.random_seed)
        np.random.shuffle(self.raw_data)
        np.random.seed(self.random_seed)
        np.random.shuffle(self.annot_data)


d = DatasetGenerator("data_files/raw", "data_files/annotations/csv", beat_width=64, random_seed=10)
