# ECG Heartbeat Classification with 1D-CNN

This project aimed to classify the heart beat types with the 1D-CNN model trained with the MIT-BIH dataset.

## Current status

| Status      | Act                            |
|:-----------:|:-------------------------------|
| Finished    |  Data preprocess               |
| Finished    |  Data analysis                 |
| Finished    |  Base model created            |
| On progress |  Model improvement             |
|             |  Testing on different datasets | 



### Project Directory Hierarchy  
  
Project hierarchy is below with subfolders and files:  
  
	 ecg-anomaly-detection 
	 	 |-- LICENSE
		 |-- .gitignore
		 |-- main.py
		 |-- dataset_generator.py
		 |-- wave_visualize.py
		 |-- README.md
		 \-- model
		   |-- model.py
		   |-- train.py
		 \-- data_files
		   |-- txt2csv.py
		   |-- convert_all_txt2csv.py
		   \-- raw
		     |-- 100.csv
		     |-- 101.csv
		     |-- (45 more files ...)
		     |-- 234.csv
		   \-- annotations
		     \-- csv
		       |-- 100annotations.csv
		       |-- 101annotations.csv
		       |-- (45 more files ...)
		       |-- 234annotations.csv
		     \-- txt
		       |-- 100annotations.txt
		       |-- 101annotations.txt
		       |-- (45 more files ...)
		       |-- 234annotations.txt

### Requirements

 - Tested on Ubuntu 18.04 and with Python 3.7.x
 - Anaconda 4.8.3 (or Miniconda) 
 - Python Libraries
	 - NumPy
	 - Tensorflow 2.x
	 - Matplotlib
	 - Scikit-Learn
	 - Pandas
	 - PyQT5 (for Wave Visualizing)
	 - Tqdm

#### Heartbeat Wave Visualizing

![Heartbeat wave visualizing demo](https://i.hizliresim.com/2oPqSG.png)

Sample monitor is above.

#### Data files

Data files uploaded to Google Drive: https://drive.google.com/file/d/18cUJOHtxICzj-nmxhZk1fL4cbFJGUP6p/view?usp=sharing
