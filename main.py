import pandas as pd


class BeatGenerator:
    def __init__(self, raw_path, annot_path):
        self.raw_data = pd.read_csv(raw_path).to_numpy()
        self.annot = pd.read_csv(annot_path).to_numpy()

        self.train = []

    def get_beat(self):
        for beat in self.annot[1:, 2]:
            self.train.append(self.raw_data[int(beat) - 64: int(beat) + 64, 1])


b_gen = BeatGenerator(raw_path="data_files/raw/100.csv",
                      annot_path="data_files/annotations/csv/100annotations.csv")

b_gen.get_beat()
