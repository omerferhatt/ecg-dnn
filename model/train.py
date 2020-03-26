from dataset_generator import DatasetGenerator
from model import create_model

model = create_model(beat_width=128)

data_generator = DatasetGenerator(raw_path="../data_files/raw/100.csv",
                                  annot_path="../data_files/annotations/csv/100annotations.csv",
                                  beat_width=128)

model.fit(data_generator.x, data_generator.y, epochs=10, batch_size=10)
