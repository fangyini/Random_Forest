# Random Forests

## Getting started:
This project acts as a room classifier, being a part of the project 'Room 
Recognition Challenge'. It reads the matrices resulting from the image with all 
the related objects detected, and computes the final room type classification.

These instructions will get you a copy of the project up and running on your 
local machine for development and testing purposes.

## Version information:
### Language: Python 3
### IDE: PyCharm Community 2017.3

## Prerequisites
### Package requirement:
Three Python packages are required: NumPy, Math and Random.
If you want to see tree visualisation or graphs, please also install
Pydot, Pandas and Matplotlib and uncomment the related parts in the code.

### Input file requirement:
Two input files are required: object_result.dat (a two-dimension binary matrix),
types_result.dat (a one-dimension matrix consisting of numbers ranging from
0 to 4)

### Installing:
If required packages are not installed, please execute the following three
commands in the terminal:
```
pip install numpy
pip install math
pip install random
```

## Running
The script decision_tree.py can be launched under PyCharm IDE or terminal.
For terminal users, please execute the following command in the terminal:
```
python3 decision_tree.py
```

The output messages consist of four parts: confusion matrix, precision rate, 
recall rate and classification rate.
