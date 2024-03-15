# 2023-09-21 15:42:22 进行测试
import pandas as pd
import deepxde as dde
import numpy as np
import re
import matplotlib.pyplot as plt

H = dde.Variable(1.4)

input_data = pd.read_csv('data/920test1_te.csv', index_col=0)
input_data = input_data[1300:4000]
input_data.Time -= 0.00005 * 1300
input_data.set_index('Time', inplace=True)
input_data.drop(columns=['Pe','delta', 'Pt'], inplace=True)

x = input_data.index.to_numpy()
geom = dde.geometry.TimeDomain(0, x[-1])

def boundary(_, on_initial):
    return on_initial

x0 = input_data.iloc[0].tolist()
y = input_data.to_numpy()


# Get the training data
observe_t = x.reshape(-1, 1)
observe_y0 = dde.icbc.PointSetBC(observe_t, y[:, 0:1], component=0)
observe_y1 = dde.icbc.PointSetBC(observe_t, y[:, 1:2], component=1)
observe_y2 = dde.icbc.PointSetBC(observe_t, y[:, 2:3], component=2)


# Initial conditions
ic1 = dde.icbc.IC(geom, lambda X: x0[0], boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda X: x0[1], boundary, component=1)
ic3 = dde.icbc.IC(geom, lambda X: x0[2], boundary, component=2)


# H * d2_delta/dt2 (This is dw/dt) + D * d_delta/dt + P_ex - P_mach = 0
def Microgrid_system(x, y):
    """Modified Lorenz system (with exogenous input).
    H * d2_delta/dt2 (This is dw/dt) + D * d_delta/dt + P_ex - P_mach = 0
    """

    Tm, w, Te = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    dw_dt = dde.grad.jacobian(y, x, i=1)
    return [
        Tm - Te - 2 * H * dw_dt
    ]


data = dde.data.PDE(
    geom,
    Microgrid_system,
    [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
    # num_domain=400,
    # num_boundary=3,
    anchors=observe_t,
#     auxiliary_var_function=ex_func2(input_data).fun,
)

net = dde.nn.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("adam", lr=0.001, external_trainable_variables=[H])

fnamevar = "variables_test1.dat"
variable = dde.callbacks.VariableValue([H], period=200, filename=fnamevar)
model.train(iterations=1000000, callbacks=[variable])


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


l, c = Chat.shape
plt.plot(range(l), Chat[:, 0], "r-")

plt.plot(range(l), np.ones(Chat[:, 0].shape) * 1.5, "r--")

plt.legend(["Hhat","True H"], loc="right")
plt.xlabel("Epoch")
yhat = model.predict(observe_t)
plt.figure()
plt.plot(observe_t, y, "-", observe_t, yhat, "--")
plt.xlabel("Time")
# plt.legend(["x", "y", "z", "xh", "yh", "zh"])
# plt.title("Training data")
plt.show()