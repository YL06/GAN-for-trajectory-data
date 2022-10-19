import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


import random

ori_data = 'E:/data/ori_trj_data/real_training_data_10000/'
# gen_data = '/home/wxr/projects/gan/output_generated/'
#TODO: limit to 350 length? max is 508
length1 = [0]*200
count = 0
maxl = 0
# for file in tqdm(os.listdir(gen_data)):
for file in tqdm(random.sample(os.listdir(ori_data), 10000)):
    count += 1
    tmp = np.load(ori_data+file, allow_pickle=True)
    templ = tmp.size
    print("templ", type(templ))
    if templ > 200:
        if maxl < templ:
            maxl = templ
        length1[199] += 1
    else:
        length1[templ-1] += 1
print(count)
s = [x/ count for x in length1]
#print("type", type(length1), maxl)


# In[38]:


plt.bar(range(200), s, alpha=0.7)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
# plt.xlim(right = 330)
plt.ylim(top = 0.080)
plt.yticks(np.arange(0, 0.07, 0.01))
plt.title('Original dataset')
plt.savefig("ori_length.pdf")


# old version
# for generated trajectory
# we round down the time to a multiple of 15

#gen_data = 'C:/Users/info/YJ/Connect/step_1_output_10000/'
# gen_data = '/home/wxr/projects/gan/output_generated/'
#TODO: limit to 350 length?
#length2 = [0]*200
#count = 0
#maxl = 0
# for file in tqdm(os.listdir(gen_data)):
#for file in tqdm(random.sample(os.listdir(gen_data), 10000)):
#    count += 1
#    tmp = np.load(gen_data+file, allow_pickle=True)
#    # round down the time to a multiple of 15, and 15seconds equals to 1
#    time = tmp[:,2]
#    rounded = time // 15
#    templ = int(np.sum(rounded))
#    if templ == 1:
#        print("one")
#        #print("time", time)
#        #print("templ", templ, type(templ))
#    if templ > 200:
#        if maxl < templ:
#            maxl = templ
#        length2[199] += 1
#    else:
#        length2[templ-1] += 1
#print(count)


#plt.bar(range(200), length2, alpha=0.7)
#plt.xlabel('Trajectory Length')
#plt.ylabel('Number of Trajectory')
# plt.xlim(right = 330)
#plt.ylim(top = 200)
#plt.yticks(np.arange(0, 10, 20))
#plt.title('Synthetic dataset')
#plt.savefig("gen_length.pdf")

gen_data = 'D:/YJ/TSG_results/'
# gen_data = '/home/wxr/projects/gan/output_generated/'
length2 = [0]*200
count = 0
maxl = 0
# for file in tqdm(os.listdir(gen_data)):
for file in tqdm(os.listdir(gen_data)):
    count += 1
    tmp = np.load(gen_data+file, allow_pickle=True)
    # round down the time to a multiple of 15, and 15seconds equals to 1
    templ,y = tmp.shape
    if templ > 200:
        if maxl < templ:
            maxl = templ
        length2[199] += 1
    else:
        length2[templ-1] += 1
print(count)
s = [x/ count for x in length2]
print(np.amax(s))
#print(s)
#length2 = [x/s for x in length2]

plt.bar(range(200), s, alpha=0.7)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
# plt.xlim(right = 330)
#plt.ylim(top = 200)
#plt.yticks(np.arange(0, 10, 20))
plt.ylim(top = 0.080)
plt.yticks(np.arange(0, 0.07, 0.01))
plt.title('Synthetic dataset')
plt.savefig("gen_length.pdf")