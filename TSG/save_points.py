from connect import get_all_enter_exit_point,grid
import cv2 as cv
import numpy as np



def main():
    map_whole = cv.imread('P:/BA/TSG/TSG/pre_process/map_generation/output_merge.png')
    map_grids = grid(map_whole)
    for i in range(32):
        for j in range(32):
            left_file, right_file = get_all_enter_exit_point(map_grids[f'{i}_{j}'], 'left', 'right')
            up_file, down_file = get_all_enter_exit_point(map_grids[f'{i}_{j}'], 'up', 'down')
            np.save(f'F:/BA/enterpoints/left/{i}_{j}.npy', left_file)
            np.save(f'F:/BA/enterpoints/right/{i}_{j}.npy', right_file)
            np.save(f'F:/BA/enterpoints/up/{i}_{j}.npy', up_file)
            np.save(f'F:/BA/enterpoints/down/{i}_{j}.npy', down_file)




if __name__ == '__main__':
     main()
