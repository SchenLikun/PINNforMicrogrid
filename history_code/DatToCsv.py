import os
import re
import numpy as np
import pandas as pd

path_0 = "/"  #原文件目录
path_1 = "/"  #存放目录

path_0 = os.getcwd()
path_1 = os.getcwd()


if path_0[-1] != '/':
    path_0 += '/'

if path_1[-1] != '/':
    path_1 += '/'

filelist = os.listdir(path_0)  # 目录下文件列表
dat_file = [f for f in filelist if '.dat' in f]

for file in dat_file:
    try:
        lines = open(file, "r").readlines()
        # read output data in fnamevar (this line is a long story...)
        Chat = np.array(
            [
                np.fromstring(
                    min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
                    sep=",",
                )
                for line in lines
            ]
        )
        pd.DataFrame(Chat).to_csv(path_1 + file[:-4] + '.csv')
    except:
        print(file)
