import pandas as pd
import deepxde as dde
import numpy as np
import re
import matplotlib.pyplot as plt
from deepxde.backend import torch

Kt = dde.Variable(1.0)
Rs = dde.Variable(1.0)
Rsh = dde.Variable(1.0)
Iso = dde.Variable(1.0)

Var_list = [Kt, Rs, Rsh, Iso]

input_data = pd.read_csv('data/1011PVtest.csv')

step_time = input_data.Time[1] - input_data.Time[0]
st = 0  # start time (10% proportion)
et = 1  # end time
input_data.Time -= step_time * int(input_data.shape[0] * st)
input_data = input_data[int(input_data.shape[0] * st): int(input_data.shape[0] * et)]

input_data.columns = ['Time', 'V', 'I', 'Is']
input_data.drop(columns='Is', inplace=True)
input_data.set_index('Time', inplace=True)

weight_list = (1 / input_data.describe().loc['mean']).tolist()

x = input_data.index.to_numpy()

geom = dde.geometry.TimeDomain(0, x[-1])

y = input_data.to_numpy()

observe_t = x.reshape(-1, 1)

# Get the training data
observe_y0 = dde.icbc.PointSetBC(observe_t, y[:, 0:1], component=0)
observe_y1 = dde.icbc.PointSetBC(observe_t, y[:, 1:2], component=1)


def Microgrid_system(x, y):
    """Modified Lorenz system (with exogenous input).
    H * d2_delta/dt2 (This is dw/dt) + D * d_delta/dt + P_ex - P_mach = 0
    """
    V, I = y[:, 0:1], y[:, 1:2]

    dV_dt = dde.grad.jacobian(y, x, i=0)
    dI_dt = dde.grad.jacobian(y, x, i=1)

    left = (1 + Iso * Rs / Kt * torch.exp((V + I * Rs) / Kt) + Rs / (Rsh * 1000)) * dI_dt
    right = - dV_dt * (Iso / Kt * torch.exp((Rs * I + V) / Kt) + 1 / (1000 * Rsh))

    return [
        left - right
    ]


data = dde.data.PDE(
    geom,
    Microgrid_system,
    [observe_y0, observe_y1],
    anchors=observe_t,
)

# data = dde.callbacks.PDEPointResampler(
#     geom,
#     Microgrid_system,
#     [observe_y0, observe_y1],
#     # anchors=observe_t,
# )

net = dde.nn.FNN([2] + [40] * 4 + [2],
                 "relu",
                 "Glorot normal"
                 )

data_y = y
data_t = x


def feature_transform(t):
    t = 0.01 * t
    return torch.concat(
        (torch.sin(t), torch.sin(2 * t)),
        axis=1,
    )


net.apply_feature_transform(feature_transform)


def output_transform(t, y):
    idx = data_y.shape[0] - 1
    k = (data_y[idx] - data_y[0]) / (data_t[idx] - data_t[0])
    b = (data_t[idx] * data_y[0] - data_t[0] * data_y[idx]) / (
            data_t[idx] - data_t[0]
    )
    linear = torch.as_tensor(k) * t + torch.as_tensor(b)
    factor = torch.tanh(t) * torch.tanh(idx - t)
    return linear + factor * torch.Tensor([1., 1.]) * y


net.apply_output_transform(output_transform)

model = dde.Model(data, net)
model.compile("adam",
              lr=0.001,
              # loss_weights=weight_list + [1e-2],
              external_trainable_variables=Var_list)

fnamevar = "variables_PV.dat"
variable = dde.callbacks.VariableValue(Var_list, period=1000, filename=fnamevar)
model.train(iterations=100000, callbacks=[variable, dde.callbacks.PDEPointResampler(period=200)])

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

print(Chat[-10:, :])

l, c = Chat.shape
plt.plot(range(l), Chat[:, 0], "r-")
plt.plot(range(l), Chat[:, 1], "b-")
plt.plot(range(l), Chat[:, 2], "y-")
plt.plot(range(l), Chat[:, 3], "g-")
# plt.plot(range(l), np.ones(Chat[:, 0].shape) * 1.5, "r--")
# plt.plot(range(l), np.ones(Chat[:, 1].shape) * 0.15, "b--")
# plt.legend(["H","D","True H","True D"], loc="right")
plt.xlabel("Epoch")

yhat = model.predict(observe_t)
plt.figure()
plt.plot(observe_t, y, "-", observe_t, yhat, "--")
plt.xlabel("Time")
# plt.legend(["x", "y", "z", "xh", "yh", "zh"])
# plt.title("Training data")
plt.show()
