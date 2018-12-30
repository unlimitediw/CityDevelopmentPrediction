import pandas as pd
import numpy as np

from os import path
import sys
sys.path.append(path.abspath('/Users/unlimitediw/PycharmProjects/MLGWU/ML'))

# Basic confusion matrix
data = pd.read_csv('../CitiesPrediction/trainCP.csv').values[:,1:]

matrix = [[0 for _ in range(10)] for _ in range(10)]
for i in range(len(data)):
    matrix[data[i][0]][data[i][1]] += 1
matrix = pd.DataFrame(np.asarray(matrix))
matrix.to_csv('trainMatrix.csv')

# Instance Feature Initialization
# CityName,Area,GreenAreaPerPers,Polycentricity,PopulationInCore,#Gov,#GovInCore,Population,Latitude,Longitude,GDP
LosAngeles = [83682.18, 5.09, 1, 100.0, 169, 169, 3884307, 34.0194, -118.4108, 891793.72]
Eindhoven = [1199.68, 827.0, 2, 44.85, 19, 2, 209170, 51.441642, 5.469722, 31087.24]
X = ['Area', 'GreenArea', 'Polycentricity', 'PopulationInCore', '#Gov', '#GovInCore', 'Population']
Y1 = [83682.18, 5.09, 1, 100.0, 169, 169, 3884307]
Y2 = [1199.68, 827.0, 2, 44.85, 19, 2, 209170]
GDP1 = 891793.72
GDP2 = 31087.24
index = np.arange(7)
for i in range(len(Y1)):
    Y1[i] = float('%.3f' % np.log2(Y1[i]))
    Y2[i] = float('%.3f' % np.log2(Y2[i]))
Y1 = tuple(Y1)
Y2 = tuple(Y2)

### 柱状图画法 记得收集
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 5))
bar_width = 0.25
opacity = 0.8
# fig, ax = plt.subplots()
plt.ylabel("Log2 Features Value Comparison")
plt.title("LosAngeles vs Eindhoven")
plt.bar(index, Y1, bar_width, alpha=opacity, color='C1', label='LosAngeles')
plt.bar(index + bar_width, Y2, bar_width, alpha=opacity, color='C2', label='Eindhoven')
plt.xticks(index + bar_width,
           ('Area', 'GreenArea', 'Polycentricity', 'PopulationInCore', '#Gov', '#GovInCore', 'Population'))
plt.tight_layout()
plt.legend()
plt.show()

### 饼图画法
plt.pie([GDP1, GDP2], colors=['C1', 'C2'], explode=[0, 0.1], labels=["LosAngeles", "Eindhoven"], shadow=True,
        autopct="%1.1f%%", pctdistance=0.8)
plt.title("GDP Comparison of LosAngeles and Eindhoven")
plt.show()