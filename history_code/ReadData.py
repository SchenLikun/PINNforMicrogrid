'''
17th July 2023 created by Likun Chen (chenlk@whu.edu.cn)
This file aims to read and transform data from RTDS into .npz/.txt/.npy
'''

import numpy as np
# import matplotlib.pyplot as plt

all_data = np.loadtxt(open("./data/SYN_ON_1ST.csv","rb"), delimiter=",", skiprows=1)

# all data
time = all_data[:, 0]
P_gen = all_data[:, 1]
Angle = all_data[:, 2]
P_mach = all_data[:, 3]
W = all_data[:, 4]

# read step
step = 10
# transfer data
time_trans = time[0:len(time):step]
P_gen_trans = P_gen[0:len(P_gen):step]
Angle_trans = Angle[0:len(Angle):step]
P_mach_trans = P_mach[0:len(P_mach):step]
W_trans = W[0:len(W):step]

np.savetxt('TestSet_P_gen.txt', np.vstack((time, P_gen)).T)
np.savetxt('TrainSet_P_gen.txt', np.vstack((time_trans, P_gen_trans)).T)


