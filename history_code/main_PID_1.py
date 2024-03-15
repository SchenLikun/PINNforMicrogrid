import pandas as pd
import deepxde as dde
import numpy as np
import re
import matplotlib.pyplot as plt
import datetime
import time
from deepxde.backend import torch



# H = dde.Variable(0.0)
# D = dde.Variable(0.0)
# input_variable = [H, D]


# input_data_path = 'data/927testforTe1.csv'
# data_col_name = ['Time', 'Tm', 'w', 'Te']
# real_value_list = [1.5, 0.15]

class Microgird():
    '''
        self.st = 0.1
        self.et = 0.5
        self.num_domain = 0
        self.num_boundary = 0
        self.net = dde.nn.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform")
        self.method = 'adam'
        self.lr = 0.001
        self.period = 200
        self.iterations = 20000
    '''
    def __init__(self, path=None, variable_list=None, data_col_name=None, fun_PI=None, real_value_list=None, date=None, index=None, **kwargs): 
        # self.st = 0.2
        # self.et = 0.5
        self.num_domain = 0
        self.num_boundary = 0
        self.num_test = None
        self.net = dde.nn.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform")
        self.method = "adam"
        self.lr = 0.0001
        self.period = 200
        self.iterations = 20000
        
        if date is None:
            now = datetime.datetime.now()
            date = now.strftime("%m%d")
        if index is None:
            self.fnamevar_path = "variables_" + date + ".dat"
        else:
            self.fnamevar_path = "variables_" + date + "_" + str(index) + ".dat"

        self.__dict__.update(kwargs)
        self.iterations = int(self.iterations)

        input_data = pd.read_csv(path)
        if data_col_name is not None:
            input_data.columns = data_col_name

        step_time = input_data.Time[1] - input_data.Time[0]
        input_data.Time -= step_time * int(input_data.shape[0] * self.st)
        input_data = input_data[int(input_data.shape[0]*self.st): int(input_data.shape[0]*self.et)]
        input_data.columns = data_col_name
        input_data.set_index('Time', inplace=True)
        input_data *= 1e3

        self.input_data = input_data
        self.variable_list = variable_list
        self.Microgrid_system = fun_PI
        self.real_value_list = real_value_list

    def build_model(self, sample=False):
        def boundary(_, on_initial):
            return on_initial

        input_data = self.input_data
        x = input_data.index.to_numpy()
        geom = dde.geometry.TimeDomain(0, x[-1])
        x0 = input_data.iloc[0].tolist()
        # ic1 = dde.icbc.IC(geom, lambda X: x0[0], boundary, component=0)
        # ic2 = dde.icbc.IC(geom, lambda X: x0[1], boundary, component=1)
        # ic3 = dde.icbc.IC(geom, lambda X: x0[2], boundary, component=2)
        y = input_data.to_numpy()
        observe_t = x.reshape(-1, 1)
        n = y.shape[0]
        if sample:
            idx = np.append(
                np.random.choice(np.arange(1, n - 1), size=n // 3, replace=False), [0, n - 1]
            )
        else:
            idx = [_ for _ in range(n)]
        observe_y0 = dde.icbc.PointSetBC(observe_t[idx], y[idx, 0:1], component=0)
        observe_y1 = dde.icbc.PointSetBC(observe_t[idx], y[idx, 1:2], component=1)
        # observe_y2 = dde.icbc.PointSetBC(observe_t[idx], y[idx, 2:3], component=2)

        data = dde.data.PDE(
            geom,
            self.Microgrid_system,
            # [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
            [observe_y0, observe_y1],
            num_domain=self.num_domain,
            num_boundary=self.num_boundary,
            num_test=self.num_test,
            anchors=observe_t,
        #     auxiliary_var_function=ex_func2(input_data).fun,
        )

        def feature_transform(t):
            t = 0.01 * t
            return torch.concat(
                (
                    t,
                    torch.sin(t),
                    torch.sin(2 * t),
                    torch.sin(3 * t),
                    torch.sin(4 * t),
                    torch.sin(5 * t),
                ),
                axis=1,
            )

        self.net.apply_feature_transform(feature_transform)


        data_t = observe_t
        data_y = y
        def output_transform(t, y):
            idx = 1999
            k = (data_y[idx] - data_y[0]) / (data_t[idx] - data_t[0])
            b = (data_t[idx] * data_y[0] - data_t[0] * data_y[idx]) / (
                data_t[idx] - data_t[0]
            )
            linear = torch.as_tensor(k) * t + torch.as_tensor(b)
            factor = torch.tanh(t) * torch.tanh(idx - t)
            return linear + factor * torch.Tensor([1, 1]) * y

        self.net.apply_output_transform(output_transform)


        model = dde.Model(data, self.net)
        model.compile(self.method, lr=self.lr, external_trainable_variables=self.variable_list)

        self.model = model
        self.x = x
        self.y = y
        self.observe_t = observe_t

    def train(self):
        # st = time.time()
        fnamevar = self.fnamevar_path
        variable = dde.callbacks.VariableValue(self.variable_list, period=self.period, filename=fnamevar)
        self.model.train(iterations=self.iterations, callbacks=[variable])
        # print('Running time:' + str(time.time()-st) + ' sec')

    def plot(self):
        lines = open(self.fnamevar_path, "r").readlines()
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
        plt.figure()
        for _ in range(c):
            plt.plot(range(l), Chat[:, _])
        for _ in self.real_value_list:
            plt.plot(range(l), np.ones(Chat[:, 0].shape) * _, "k--")
        plt.xlabel("Epoch")
        plt.show()

        try:
            yhat = self.model.predict(self.observe_t)
            plt.figure()
            plt.plot(self.observe_t, self.y, "-", self.observe_t, yhat, "--")
            plt.xlabel("Time")
            plt.show()
        except:
            pass

if __name__ == "__main__":
    k_p = dde.Variable(1.0)
    k_i = dde.Variable(1.0)
    input_variable = [k_p]#, k_i]

    err = pd.read_csv('data/1211_PI2_v2.csv')['err'].to_numpy().reshape(-1, 1)

    err = np.vstack((err ,err, err))

    def Microgrid_system(x, y):

        U, I = y[:, 0:1], y[:, 1:2]

        dU_dt = dde.grad.jacobian(y, x, i=0)
        dI_dt = dde.grad.jacobian(y, x, i=1)
        
        return [
            (k_p*dI_dt/10+I*0.5) - dU_dt - torch.as_tensor(err)
        ]

    input_data_path = 'data/1211_PI2.csv'
    data_col_name = ['Time', 'I', 'U']
    real_value_list = [0.25]#, 0.5]

    microgird = Microgird(path=input_data_path, 
                          variable_list=input_variable, 
                          data_col_name=data_col_name, 
                          fun_PI=Microgrid_system, 
                          real_value_list=real_value_list,
                          st=0,
                          et=1,
                          # num_test=400,
                          lr=0.0001,
                        #   date='0929',
                          index='PID_base',
                          net=dde.nn.FNN([6] + [100] * 3 + [2], "tanh", "Glorot uniform"),
                          iterations=1000000
                          )
    
    microgird.build_model(sample=None)
    microgird.train()
    microgird.plot()


