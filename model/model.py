from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def create_model(beat_width=64):
    inp = Input(shape=(beat_width * 2, 1), name="input")

    c1 = Conv1D(64, kernel_size=5, activation="relu", name="conv1d_1")(inp)
    c2 = Conv1D(64, kernel_size=5, activation="relu", name="conv1d_2")(c1)
    mp1 = MaxPooling1D(pool_size=2, strides=2)(c2)

    c3 = Conv1D(128, kernel_size=7, activation="relu", name="conv1d_3")(mp1)
    c4 = Conv1D(256, kernel_size=7, activation="relu", name="conv1d_4")(c3)

    mp2 = MaxPooling1D(pool_size=2, strides=2)(c4)
    c5 = Conv1D(512, kernel_size=7, activation="relu", name="conv1d_5")(mp2)

    f = Flatten(name="flatten")(c4)
    ds1 = Dense(512, activation="relu", name="dense_1")(f)
    ds2 = Dense(256, activation="relu", name="dense_2")(ds1)
    ds3 = Dense(256, activation="relu", name="dense_3")(ds2)
    ds4 = Dense(128, activation="relu", name="dense_4")(ds3)
    out_ds4 = Dense(13, activation="softmax", name="out_dense_5")(ds3)

    model = Model(inputs=inp, outputs=out_ds4)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
