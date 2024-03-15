import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

input_data = pd.read_csv('../../data/1011PVtest.csv')
fnamevar = "../../variables_test2pv1014.dat"

input_data.columns = ['Time', 'V', 'I', 'Is']
input_data.drop(columns='Is', inplace=True)
input_data.set_index('Time', inplace=True)

step_time = input_data.index[1] - input_data.index[0]
dI_dt = (input_data.I.to_numpy()[2:] - input_data.I.to_numpy()[:-2]) / step_time / 2
dV_dt = (input_data.V.to_numpy()[2:] - input_data.V.to_numpy()[:-2]) / step_time / 2

lines = open(fnamevar, "r").readlines()
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

# Kt, Rs, Rsh, Iso
Kt = Chat[:, 0]
Rs = Chat[:, 1]
Rsh = Chat[:, 2] * 1e4
Iso = Chat[:, 3]

left = 1 + Iso*Rs/Kt*np.exp((np.repeat(np.reshape(input_data.V.to_numpy()[1:-1],(-1,1)), Chat.shape[0], axis =1)+np.outer(input_data.I.to_numpy()[1:-1], Rs))/Kt)  + Rs/Rsh

right = - np.repeat(np.reshape(dV_dt,(-1,1)), Chat.shape[0], axis =1) * (Iso/Kt*np.exp((np.repeat(np.reshape(input_data.V.to_numpy()[1:-1],(-1,1)), Chat.shape[0], axis =1)+np.outer(input_data.I.to_numpy()[1:-1], Rs))/Kt) + 1/Rsh)

result = left*np.repeat(np.reshape(dI_dt,(-1,1)), Chat.shape[0], axis =1) - right

plt.plot(np.mean(np.abs(result), axis=0))

plt.figure()
plt.plot(np.mean(np.abs(result), axis=0))
plt.plot(range(Chat.shape[0]), np.ones(Chat.shape[0]) * 0, "k--")
plt.ylim((-0.1, 1))
plt.show()

plt.figure()
plt.plot(np.mean(np.abs(result), axis=0))
plt.plot(range(Chat.shape[0]), np.ones(Chat.shape[0]) * 0, "k--")
plt.xlim((0,200))
plt.show()

l, c = Chat.shape
plt.plot(range(l), Chat[:, 0], color="#10739e", linestyle='--')
plt.plot(range(l), Chat[:, 1], color="#f2931e", linestyle='--')
plt.plot(range(l), Chat[:, 2], color="#ae4132", linestyle='--')
plt.plot(range(l), Chat[:, 3], color="#23445d", linestyle='--')
# plt.plot(range(l), np.ones(Chat[:, 0].shape) * 1.5, "r--")
# plt.plot(range(l), np.ones(Chat[:, 1].shape) * 0.15, "b--")
plt.legend(["$K_t$","$R_{s}$","$R_{sh}$","$I_so$"], loc="right")
plt.xlabel("Epoch")
plt.show()
