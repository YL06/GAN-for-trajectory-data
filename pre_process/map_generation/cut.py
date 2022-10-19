import cv2 as cv
import os
import numpy as np

dir = 'output\\'
#334 442

# new output

# YJ: this whole map_generation part is still hardcoded
# I changed the values to be suitable for my display resolution (1980x1080).
# The values may change for the same resolution dependent of the size of the windows taskbar

# changed dy from 63 to 61
# dx 473 dy 61
dx = 406
dy = 19
num=0
for i in os.listdir(dir):
    if i[-3:]!='png':
        continue
    pict = cv.imread(os.path.join(dir,i))
    pict = pict[dy:-dy,dx:-dx,:]
    cv.imencode('.png', pict)[1].tofile('output_cut/%s.png'%('0'*(4-len(str(num)))+str(num)))
    num+=1



