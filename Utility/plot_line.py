# coding:utf-8
import os
import cv2 as cv
import os
import copy
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from tqdm import tqdm
from sklearn.model_selection import train_test_split
Image.MAX_IMAGE_PIXELS = 300000000

min_lon=-8.687466
min_lat=41.123232
max_lon = -8.553186
max_lat = 41.237424
lon = 0.1342800000000004
lat = 0.11419199999999563
grids_num =32
width = 14976
height = 16832

# point (x,y)
def paint_dot(x, y):
      pixel_x = int(((x - min_lon) / lon) * width)
      pixel_y = int(((y - min_lat) / lat) * height)
      for i in range(9):
            for j in range(9):
                  map_whole.putpixel((pixel_x-4+i, pixel_y-4+j), (30, 30, 30))



#get the grids
def paint_grid():
      for i in range(31):
            for j in range(height):
                  for m in range(5):
                        map_whole.putpixel((((i+1)*grid_w)+(m-1), j),(70,70,70))
      for i in range(31):
            for j in range(width):
                  for m in range(5):
                        map_whole.putpixel((j, (((i+1)*grid_h)+(m-1))),(70,70,70))


def pixel(x, y):
      pixel_x = int(((x - min_lon) / lon) * width)
      pixel_y = int(((y - min_lat) / lat) * height)
      return pixel_x,pixel_y

map_path = 'C:/Users/info/YJ/pre_process/map_generation/output_merge.png'
map_whole = Image.open(map_path)
#print("lon", max_lon-min_lon)
#print("lat", max_lat-min_lat)

grid_w = int(width/32)
grid_h = int(height/32)

gen_data_path = 'D:/YJ/TSG_results/'


#ImageDraw.line(xy, fill=None, width=0, joint=None)

for traj in tqdm(os.listdir(gen_data_path)):
      u = np.load(gen_data_path + traj, allow_pickle=True)
      length, w = u.shape
      draw = ImageDraw.Draw(map_whole)
      for i in range(length - 1):
            try:
                  pixels = pixel(u[i][0], u[i][1])
                  pixeld = pixel(u[i + 1][0], u[i + 1][1])
                  draw.line([pixels, pixeld], (70, 70, 70), 10)
            except Exception:
                  pass
paint_grid()


map_whole = map_whole.save('C:/Users/info/YJ/Utility/test_line.png')