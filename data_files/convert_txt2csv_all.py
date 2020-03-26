import argparse
import glob
import os
import sys

from tqdm import tqdm

from data_files.txt2csv import txt2csv

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
