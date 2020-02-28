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
