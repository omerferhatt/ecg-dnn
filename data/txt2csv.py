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

from os.path import join

import numpy as np
import pandas as pd


def txt2csv(file_path, save_dir):
    """
    Converts MIT-BIH dataset .txt files to .csv which more readable with pandas

    :param file_path: .txt file path
    :param save_dir: Converted file save directory
    :return: Saves .csv file near to .txt file's location
    """

    headers = _txt_header_extract(file_path)
    rows = _txt_row_extracter(file_path)

    df = pd.DataFrame(rows, columns=headers)

    file_name = file_path.split(".")[0].split("/")[-1]

    save_ext = ".csv"
    save_file = f"{file_name}{save_ext}"
    save_path = join(save_dir, save_file)

    df.to_csv(save_path)


def _txt_header_extract(file_path):
    """
    Gets headers from .txt file

    :param file_path: .txt file path
    :return: Header list
    """
    with open(file_path, "r") as txt_file:
        txt = txt_file.readline()
        row = txt.split("\n")
        hdr = " ".join(row[0].split()).split(" ")
        hdr.remove("#")
        return hdr


def _txt_row_extracter(file_path):
    """
    Retrieves and corrects the information on the rows

    :param file_path: .txt file path
    :return: Data rows
    """
    with open(file_path, "r") as txt_file:
        txt = txt_file.read()
        row = txt.split("\n")
        info = []

        for index, r in enumerate(row[1:-1]):
            i = " ".join(r.split()).split(" ")
            if len(i) != 7:
                i.append("")
            info.append(np.array(i))
        return np.array(info)
