import csv
import numpy as np
import ast

mycsv = csv.reader(open('D:/YJ/data/train.csv'))
i = 0

for row in mycsv:
    if i == 0:
        i += 1
        continue
    #elif i < 6410:
    else:
        trjdata = row[8]
        savefile = ast.literal_eval(trjdata)
        np.save(f'D:/YJ/data/single_trajectory_data/{i-1}.npy', savefile)
        i += 1
    #else:
#        break
