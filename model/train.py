import os
import tensorflow as tf

os.chdir("..")

import model
from data_generator import DatasetGenerator

BEAT_WIDTH = 128

model = model.create_model(beat_width=BEAT_WIDTH)

data_generator = DatasetGenerator(raw_path="data_files/raw",
                                  annot_path="data_files/annotations/csv",
                                  beat_width=BEAT_WIDTH,
                                  random_seed=100)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="model/logs", histogram_freq=1)

model.fit(data_generator.raw_data, data_generator.annot_data,
          epochs=10, batch_size=20,
          callbacks=[tensorboard_callback])
