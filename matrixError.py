import pandas as pd
import numpy as np

data = pd.read_csv('../CitiesPrediction/trainCP.csv').values[:,1:]

matrix = [[0 for _ in range(10)] for _ in range(10)]
for i in range(len(data)):
    matrix[data[i][0]][data[i][1]] += 1
matrix = pd.DataFrame(np.asarray(matrix))
matrix.to_csv('trainMatrix.csv')

from os import path
import sys
sys.path.append(path.abspath('/Users/unlimitediw/PycharmProjects/MLGWU/ML'))
