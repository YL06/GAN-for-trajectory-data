import math

import numpy as np
from tqdm import tqdm
import os
import random
from connect import get_all_enter_exit_point,grid
import cv2 as cv
import pickle
import time
import multiprocessing as mp

# x is lon, y is lat
d_lat = 0.0035685
d_lath = 0.00178425
d_lon = 0.00419625
d_lonh = 0.002098125
min_lon=-8.687466
min_lat=41.123232
max_lon = -8.553186
max_lat = 41.237424
center_lon = -8.685367875
center_lat = 41.12501625
map_whole = cv.imread('C:/Users/info/YJ/pre_process/map_generation/output_merge.png')
map_grids = grid(map_whole)



def get_direction(grid_a, grid_b):
    dx, dy = grid_b[0]-grid_a[0], grid_b[1]-grid_a[1]
    # print("dx, dy", dx, dy)
    # check for "jumps"
    if abs(dx) > 1:
        dx = dx/abs(dx)
    if abs(dy) > 1:
        dy = dy/abs(dy)
    # added if for diagonals
    if abs(dx) + abs(dy) == 2:
        r = random.randint(0,1)
        dx = r * dx
        dy = (1-r) * dy
    direction = {(-1, 0): 'up', (1, 0): 'down',
                 (0, 1): 'right', (0, -1): 'left'}
    return (direction[(dx, dy)], direction[(-dx, -dy)])

def get_direction_list(grid_data):
    enter_list = []
    exit_list = []
    length = len(grid_data)
    for i in range(length-1):
        exit, enter = get_direction(grid_data[i], grid_data[i+1])
        enter_list.append(enter)
        exit_list.append(exit)
    enter_list.insert(0, get_direction(grid_data[0], grid_data[1])[1])
    exit_list.insert(-1, get_direction(grid_data[-2], grid_data[-1])[0])
    return enter_list, exit_list

#TODO: save file to numpy
# append as list
# store to numpy
# load numpy file, to list

def get_dist(point_a, point_b):
    x1, y1 = point_a
    x2, y2 = point_b
    return math.hypot(x1-x2, y1-y2)

def shortest_dist(point_list, point):
    dis = 1000
    for p in point_list:
        distance = get_dist(p, point)
        if distance < dis:
            dis = distance
            return_point = p
    return return_point



# input from get_grid_seq_mat
def create_seq2(trj_data, map_grids, foldername):
    grid_seq = Extract(trj_data)
    # print("grid_seq", grid_seq, type(grid_seq))
    enter_list, exit_list = get_direction_list(grid_seq)
    i = 0
    for line in trj_data:
        name = '%s_%s' % (int(grid_seq[i][0]), int(grid_seq[i][1]))
        #file_name = f'F:/BA/data42/{foldername}/'+name+'.npy'
        file_name = f'E:/BA/data_42_1000/' + name + '.p'
        enter_dir = enter_list[i]
        exit_dir = exit_list[i]
        # random enter and exit point for start and end of a trajectory? or use dist
        # choose enter and exit point with shortest distance
        enter_l = np.load(f'F:/BA/enterpoints/{enter_dir}/' + name + '.npy')
        exit_l = np.load(f'F:/BA/enterpoints/{exit_dir}/' + name + '.npy')
        #enter_l, exit_l = get_all_enter_exit_point(map_grids[name], enter_dir, exit_dir)
        enter_l = enter_l.tolist()
        exit_l = exit_l.tolist()
        #enter = enter[0].tolist()
        #exit = exit[0].tolist()
        seq = normalize_seq(line)
        enter = shortest_dist(enter_l, seq[0])
        exit = shortest_dist(exit_l, seq[-1])
        #print("normalized seq", seq)
        seq.insert(0, enter)
        seq.append(exit)
        #seq = np.append(enter, seq, axis=0)
        #seq = np.append(seq, exit, axis=0)
        if os.path.exists(file_name):
            save_file = pickle.load(open(file_name, "rb"))
            #np_file = np.load(file_name, allow_pickle=True)
            #save_file = np_file.tolist()
            #seq = np.append(np_file, seq)
            #np.save(file_name, seq)
        else:
            save_file = []
        save_file.append(seq)
        #save_numpy_file = np.array(save_file)
        #np.save(file_name, save_numpy_file)
        #dont save for testing
        pickle.dump(save_file, open(file_name, "wb"))
        i += 1

# input one line of get_grid_seq_mat
# output a seq list
def normalize_seq(complete_data):
    c_x, c_y = get_grid_center(complete_data[0])
    seq = []
    #print("complete", complete_data)
    for cor in complete_data[1]:
        #print("cor", cor)
        a = (cor[0] - c_x) / d_lonh
        b = (cor[1] - c_y) / d_lath
        seq.append([a, b])
    #print("seq no", seq)
    # seq = np.array([np.array(xi) for xi in seq])
    return seq


def get_grid_center(gd):
    y = (31 - gd[0]) * d_lat + center_lat
    x = gd[1] * d_lon + center_lon
    return x,y


# imput trajectory sequence
# output matrix [[[gridx,gridy],seq]] with seq = [[x1,y1],[x2,y2]..]
# convert numpy to list to store this step
def get_grid_seq_mat(trj_seq):
    grid_seq_mat = []
    seq = []
    c_x,c_y = get_grid(trj_seq[0])
    # not 100% correct, cuts trajectory sequences, where the endspot is reached mid trajectory
    # fixed with lenght
    i = 1
    for trj in trj_seq:
        # print("coordinate", trj)
        x, y = get_grid(trj)
        # last sequence
        if i == len(trj_seq):
            grid_seq_mat.append([[c_x, c_y], seq])
            continue
        if c_x != x or c_y != y:
            grid_seq_mat.append([[c_x,c_y], seq])
            seq = []
            c_x, c_y = x, y
        to_list = trj.tolist()
        seq.append(to_list)
        i += 1
    return grid_seq_mat


# get the grid of a trajectory
def get_grid(traj):
    y = (traj[0] - min_lon) // d_lon
    x = 31 - (traj[1] - min_lat) // d_lat
    return x,y


# check if trj are outside of grid
# return true if they are
# return false if they are inside grid
def check_outside(traj):
    traj_x = traj[:,0]
    traj_y = traj[:,1]
    # true: all trj inside grid
    check_x = np.all(np.logical_and((traj_x < max_lon),(traj_x > min_lon)))
    check_y = np.all(np.logical_and((traj_y < max_lat),(traj_y > min_lat)))
    return not (check_x and check_y)

# extract first element of each sublist in a list of lists
def Extract(lst):
    return [item[0] for item in lst]

# use math.dist
def multi(iter):
    #for i in range(iter, iter+10):
    #foldername = str(iter)
    #if not os.path.exists(f'F:/BA/multi/{foldername}'):
    #    os.makedirs(f'F:/BA/multi/{foldername}')
    #for filename in pbar(os.listdir(f'F:/BA/single_trajectory_folder/{foldername}')):
    #    file_name = f'F:/BA/single_trajectory_folder/{foldername}/{filename}'
    for filename in tqdm(os.listdir('E:/different_sizes/first-stage-GAN/grid_data_forfirst/data_1000/')):
        file_name = f'E:/different_sizes/second-stage-GAN/preprocessed/data_1000/{filename}'
        if not os.path.exists(file_name):
            continue
        tj = np.load(file_name, allow_pickle=True)
        if len(tj) == 0:
            continue
        if check_outside(tj):
            continue
        mat = get_grid_seq_mat(tj)
        # trajectory sequence located only in one grid
        if len(mat) == 1:
            continue
        # foldername = 0 just for saving this
        foldername = 0
        create_seq2(mat, map_grids, foldername)

def main():
    pool = mp.Pool(1)
    pool.map(multi, [1])
    pool.close()
    #map_whole = cv.imread('P:/BA/TSG/TSG/pre_process/map_generation/output_merge.png')
    #map_grids = grid(map_whole)
#    for i in range(1):
#        #file_name = f'P:/BA/TSG/TSG/prepare_data/single_trajectory_data/0.npy'
#        file_name = f'P:/BA/TSG/TSG/prepare_data/testing_data/short.npy'
#        test_single = np.load(file_name)
#        if len(test_single) == 0:
#            continue
#        if check_outside(test_single):
#            continue
#        mat = get_grid_seq_mat(test_single)
#        # trajectory sequence located only in one grid
#        if len(mat) == 1:
#            continue
#        create_seq2(mat, map_grids)
    #f = open('P:/BA/TSG/TSG/First_stage_gan/traj_all.txt', 'r')
    #data = f.readlines()
    #f.close()
    #with open('log1.txt', 'w') as file:
    #i = 0
    #for foldername in os.listdir('F:/BA/single_trajectory_folder'):
    #    print("working on folder", foldername, i)
    #    if not os.path.exists(f'F:/BA/data42/{foldername}'):
    #        os.makedirs(f'F:/BA/data42/{foldername}')
    #    i += 1
    #    if foldername == '0':
    #        continue
    #    if foldername == '1':
    #        continue
    #    if foldername == '10':
    #        continue
    #    if foldername == '100':
    #        continue
    #    if foldername == '101':
    #        continue
    #    if foldername == '102':
    #        continue
    #    pbar = ProgressBar()
    #    count = 1
    #    for filename in pbar(os.listdir(f'F:/BA/single_trajectory_folder/{foldername}')):
    #for i in tqdm(data):
    #    name = i[:-5]
     #       #tic = time.perf_counter()
     #       file_name = f'F:/BA/single_trajectory_folder/{foldername}/{filename}'
     #       if not os.path.exists(file_name):
     #           continue
     #       tj = np.load(file_name, allow_pickle=True)
    #        if len(tj) == 0:
    #            continue
    #        if check_outside(tj):
   #             continue
   #         #print("working on file", filename)
     #       mat = get_grid_seq_mat(tj)
     #       # trajectory sequence located only in one grid
     #       if len(mat) == 1:
     #           continue
            #toc = time.perf_counter()
            #tic2 = time.perf_counter()
     #       create_seq2(mat, map_grids, foldername)
            #toc2 = time.perf_counter()
            #if count % 100 == 0:
            #    print(f"prepare data in {(toc - tic)/count:0.4f} seconds")
            #    print(f"calc data in {(toc2 - tic2)/count:0.4f} seconds")
            #count += 1



if __name__ == '__main__':
     main()
