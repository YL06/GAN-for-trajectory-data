import numpy as np
import os
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
import random
import csv

ori_data_path = 'E:/data/ori_trj_data/real_training_data_10000/'
gen_data_path = 'D:/YJ/TSG_results/'
#gen_data_path = 'D:/YJ/06_05/112.npy'

min_d_global = 1000
added_d = 0
#for file in tqdm(random.sample(os.listdir(ori_data_path), 100)):
with open('C:/Users/info/YJ/Utility/sim.csv', 'w') as f:
    # create the csv writer
    writer=csv.writer(f, delimiter=',',lineterminator='\n',)
    i = 1
    for traj in tqdm(os.listdir(gen_data_path)):
        u = np.load(gen_data_path + traj, allow_pickle=True)
        min_d_traj = ''
        min_d = 1000
        for file in tqdm(os.listdir(ori_data_path)):
            v = np.load(ori_data_path + file, allow_pickle=True)
            distance = max(directed_hausdorff(v, u)[0], directed_hausdorff(u, v)[0])
            if distance < min_d:
                min_d = distance
                min_d_traj = file
        row = [min_d, traj, min_d_traj]
        writer.writerow(row)
        if min_d_global > min_d:
            min_d_global = min_d
        added_d += min_d
        i += 1
#for traj in tqdm(os.listdir(gen_data_path)):
#    u = np.load(gen_data_path+traj, allow_pickle=True)
#    for file in tqdm(os.listdir(ori_data_path)):
#        v = np.load(ori_data_path+file, allow_pickle=True)
#        distance = max(directed_hausdorff(v, u)[0], directed_hausdorff(u, v)[0])
#        if distance < min_d:
#            min_d = distance
#            min_d_traj = file
print("i", i)
print("added d", added_d)
print("avg", added_d/i)
print("min_d", min_d)