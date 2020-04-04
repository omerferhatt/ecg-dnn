#  MIT License
#
#  Copyright (c) 2020 Omer Ferhat Sarioglu
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

#  MIT License
#
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def create_model(beat_width=64):
    inp = Input(shape=(beat_width * 2, 1), name="input")

    c1 = Conv1D(64, kernel_size=7, activation="relu", name="conv1d_1")(inp)
    mp1 = MaxPooling1D(pool_size=2, strides=2)(c1)

    c3 = Conv1D(128, kernel_size=5, activation="relu", name="conv1d_3")(mp1)
    c4 = Conv1D(128, kernel_size=5, activation="relu", name="conv1d_4")(c3)

    mp2 = MaxPooling1D(pool_size=2, strides=2)(c4)
    c5 = Conv1D(256, kernel_size=5, activation="relu", name="conv1d_5")(mp2)

    f = Flatten(name="flatten")(c5)
    ds1 = Dense(256, activation="relu", name="dense_1")(f)
    ds2 = Dense(128, activation="relu", name="dense_2")(ds1)
    ds3 = Dense(64, activation="relu", name="dense_3")(ds2)
    out_ds4 = Dense(23, activation="softmax", name="out_dense_5")(ds3)

    model = Model(inputs=inp, outputs=out_ds4)
    opt = Adam(0.0005)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
