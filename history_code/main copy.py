import numpy as np
import deepxde as dde
import pandas as pd
from deepxde.backend import torch
import re
import matplotlib.pyplot as plt


def sbinn(data_t, data_y):


    H = dde.Variable(0.0)
    D = dde.Variable(0.0)

    var_list_ = [H, D]


    def ODE(x, y):
        
        w, Tm, Te = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        
        dw_dt = dde.grad.jacobian(y, x, i=0, j=0)
        return [
            Tm-Te - 2*H*dw_dt - D*(w-1)
        ]
    
    geom = dde.geometry.TimeDomain(data_t[0, 0], data_t[-1, 0])

    # Observes
    observe_w = dde.PointSetBC(data_t, y[:, 0:1], component=0)
    observe_Tm = dde.PointSetBC(data_t, y[:, 1:2], component=1)
    observe_Te = dde.PointSetBC(data_t, y[:, 2:3], component=2)

    data = dde.data.PDE(geom, ODE, [observe_w, observe_Tm, observe_Te], anchors=data_t)

    net = dde.maps.FNN([3] + [128] * 3 + [3], "swish", "Glorot normal")

    def feature_transform(t):
        t = 0.01 * t
        return torch.concat(
            (
                t,
                torch.sin(t),
                torch.sin(2 * t),
            ),
            axis=1,
        )

    net.apply_feature_transform(feature_transform)

    def output_transform(t, y):
        idx = 3999
        k = (data_y[idx] - data_y[0]) / (data_t[idx] - data_t[0])
        b = (data_t[idx] * data_y[0] - data_t[0] * data_y[idx]) / (
            data_t[idx] - data_t[0]
        )
        linear = torch.as_tensor(k) * t + torch.as_tensor(b)
        factor = torch.tanh(t) * torch.tanh(idx - t)
        return linear + factor * torch.Tensor([1., 1., 1.]) * y

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    firsttrain = 10000
    callbackperiod = 1000
    maxepochs = 5000000

    # model.compile("adam", lr=1e-3, loss_weights=[0, 0, 0, 0, 0, 0, 1e-2])
    # model.train(epochs=firsttrain, display_every=1000)
    model.compile(
        "adam",
        lr=5e-4,
        # loss_weights=[1, 1, 1e-2, 1, 1, 1, 1e-2],
        external_trainable_variables=var_list_,
    )
    variablefilename = "variables_sbinn.dat"
    variable = dde.callbacks.VariableValue(
        var_list_, period=callbackperiod, filename=variablefilename
    )
    losshistory, train_state = model.train(
        epochs=maxepochs, display_every=1000, callbacks=[variable]
    )

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    return model

input_data = pd.read_csv('data/927testforTe2.csv')
step_time = input_data.Time[1] - input_data.Time[0]
st = 0 # start time (10% proportion)
et = 1 # end time
input_data.Time -= step_time * int(input_data.shape[0] * st)
input_data = input_data[int(input_data.shape[0]*st): int(input_data.shape[0]*et)]
input_data.set_index('Time', inplace=True)
input_data.columns=['Tm','w', 'Te']
input_data = input_data.reindex(columns = ['w','Tm','Te'])
x = input_data.index.to_numpy().reshape(-1, 1)
y = input_data.to_numpy()

model = sbinn(x, y)

variablefilename = "variables_sbinn.dat"
lines = open(variablefilename,"r").readlines()
Chat = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key = len),
            sep = ","
        )
        for line in lines
    ]
)
l, c = Chat.shape
plt.figure()
plt.plot(range(l), Chat[:, 0], "r-")
plt.plot(range(l), Chat[:, 1], "b-")
plt.plot(range(l), np.ones(Chat[:, 0].shape) * 1.5, "r--")
plt.plot(range(l), np.ones(Chat[:, 1].shape) * 0.15, "b--")
plt.legend(["H", "D", "HTrue", "DTrue"])
plt.xlabel("epoch")
plt.show()

yhat = model.predict(x)
plt.figure()
plt.plot(x, y, "-", x, yhat, "--")
plt.xlabel("Time")
# plt.legend(["x", "y", "z", "xh", "yh", "zh"])
# plt.title("Training data")
plt.show()