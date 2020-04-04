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

import argparse
import glob
import os
import sys

from tqdm import tqdm

from data.txt2csv import txt2csv

parser = argparse.ArgumentParser(
    prog="txt2csv",
    description="Converts csv styled txt files to csv files and saves them into the directory"
)

parser.add_argument("-save", "--s", type=str, default="annotations/csv", help="Save directory path")

args = parser.parse_args(["-save", "annotations/csv"])

os.chdir("")
save_dir = args.s

raw_csv_list = glob.glob("*/*.csv", recursive=True)
txt_list = glob.glob("**/*.txt", recursive=True)

for txt in tqdm(txt_list, position=0, leave=True, unit="file"):
    txt2csv(file_path=txt, save_dir=save_dir)

sys.exit()
