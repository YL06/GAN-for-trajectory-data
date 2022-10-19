import os
import numpy as np
import pickle

# multiple folder
#def main():
#    for i in range(32):
#        for j in range(32):
#            data42 = []
#            for f in range(1):
#                file_name = f'F:/BA/multi/{f}/{i}_{j}.p'
#                if not os.path.isfile(file_name):
#                    continue
#                data42 += pickle.load(open(file_name, "rb"))
#            if not data42:
#                continue
#            np_file = np.array(data42)
#            np.save(f'F:/BA/data1/{i}_{j}.npy', np_file)

#single folder
def main():
    for i in range(32):
        for j in range(32):
            data42 = []
            file_name = f'E:/different_sizes/second-stage-GAN/preprocessed/p/data_5000/{i}_{j}.p'
            if not os.path.isfile(file_name):
                continue
            data42 += pickle.load(open(file_name, "rb"))
            if not data42:
                continue
            np_file = np.array(data42)
            np.save(f'E:/different_sizes/second-stage-GAN/preprocessed/data_5000/{i}_{j}.npy', np_file)

if __name__ == '__main__':
    main()
