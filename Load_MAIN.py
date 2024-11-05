import pandas as pd
import deepxde as dde
import numpy as np
import re
import datetime
import os
import time
import matplotlib.pyplot as plt

class Load():
    def __init__(self, path=None, date=None, index=None, **kwargs):

        self.st = 0.
        self.et = 1.
        self.num_domain = 0
        self.num_boundary = 0
        self.num_test = None

        self.net = dde.nn.FNN([1] + [100] * 3 + [5], "swish", "Glorot uniform")


        self.method = "adam"
        self.lr = 0.0001
        self.period = 200
        self.iterations = 20000

  
        self.path = path
        self.date = date
        self.index = index

        if date is None:
            now = datetime.datetime.now()
            self.date = now.strftime("%m%d")
        if index is None:
            self.fnamevar_path = "Load_variables_ModelSave" + self.date + ".dat"
        else:
            self.fnamevar_path = "Load_variables_ModelSave" + self.date + "_" + str(index) + ".dat"

        self.set_variable()
        self.__dict__.update(kwargs)
        self.iterations = int(self.iterations)
        self.check_dat_file()


        input_data = pd.read_csv(path)
        step_time = input_data.Time[1] - input_data.Time[0]
        input_data.Time -= step_time * int(input_data.shape[0] * self.st)
        input_data = input_data[int(input_data.shape[0]*self.st): int(input_data.shape[0]*self.et)]

        self.input_data = input_data

        self.rename_csv()

        Pzip = dde.Variable(self.init_Pzip)
        alpha_1 = dde.Variable(self.alpha_1)
        alpha_2 = dde.Variable(self.alpha_2)


        self.variable_list = [Pzip, alpha_1, alpha_2]

        def ode(x, y):

            Ptot, Qtot, V_pu, freq = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
            dPtot = dde.grad.jacobian(y, x, i=0)
            dQtot = dde.grad.jacobian(y, x, i=1)
            dV = dde.grad.jacobian(y, x, i=2) * 13.8
            df_dt = dde.grad.jacobian(y, x, i=3)

            alpha_3 = 1 - alpha_1 - alpha_2

            return [
                Ptot - Pzip*(alpha_1*V_pu**2+alpha_2*V_pu+alpha_3),
                -dPtot + Pzip*(2 * V_pu * alpha_1 * dV + alpha_2 * dV / 13.8)

            ]



        self.ode = ode

    def check_dat_file(self):
        if os.path.exists(self.fnamevar_path):
            new_index = input(
                "The dat file already exists, please enter a new index to modify, or press Enter to continue")
            if new_index != '':
                self.index = new_index
                self.fnamevar_path = "Load_variables_ModelSave_" + self.date + "_" + str(new_index) + ".dat"
    
    def rename_csv(self):
        dict_col = {
            'Time': 'Time',
            'Ptot_Area1': 'Ptot',
            'Qtot_Area1': 'Qtot',
            'V101RMS': 'V_pu',
            'PMU1freq': 'freq',
            }
        self.input_data.rename(columns=lambda x: dict_col[x.split('|')[-1]], inplace=True)
        self.input_data = self.input_data[list(dict_col.values())]
        self.input_data.set_index('Time', inplace=True)

    def set_variable(self):
        self.init_Pzip = 1.
        self.alpha_1 = 1.
        self.alpha_2 = 1.

    def check_init_data(self):
        data = self.input_data
        time_step = data.index[1] - data.index[0]
        def data_plot(I, O, kp, ki):
            I_t = (data[I].to_numpy()[2:] - data[I].to_numpy()[:-2]) / (2 * time_step)
            O_t = (data[O].to_numpy()[2:] - data[O].to_numpy()[:-2]) / (2 * time_step)
            print(pd.DataFrame([I_t*kp+data[I].to_numpy()[1:-1]*ki, O_t]).T.plot())



    def build(self, transform=None, **kwargs):
        if transform == ('input' or 'all'):
            self.net = dde.nn.FNN([6] + [100] * 3 + [4], "tanh", "Glorot uniform")
        self.__dict__.update(kwargs)
        input_data = self.input_data
        x = input_data.index.to_numpy()
        geom = dde.geometry.TimeDomain(0, x[-1])
        y = input_data.to_numpy()
        observe_t = x.reshape(-1, 1)
        n = y.shape[0]
        y_real = y
        def boundary(_, on_initial):
            return on_initial
        observe_y0 = dde.icbc.PointSetBC(observe_t, y[:, 0:1], component=0)
        observe_y1 = dde.icbc.PointSetBC(observe_t, y[:, 1:2], component=1)
        observe_y2 = dde.icbc.PointSetBC(observe_t, y[:, 2:3], component=2)
        observe_y3 = dde.icbc.PointSetBC(observe_t, y[:, 2:3], component=3)


        data = dde.data.PDE(
            geom,
            self.ode,
            [observe_y0, observe_y1, observe_y1, observe_y2, observe_y3],
            # [],
            anchors=observe_t,
        )

        if transform is not None:
            from deepxde.backend import torch
            tran_in = False
            tran_out = False
            if transform == 'input':
                tran_in = True
            if transform == 'output':
                tran_out = True
            if transform == 'all':
                tran_in = True
                tran_out = True

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

            data_t = observe_t
            data_y = y

            def output_transform(t, y):
                idx = n - 1
                
                k = (data_y[idx] - data_y[0]) / (data_t[idx] - data_t[0])
                b = (data_t[idx] * data_y[0] - data_t[0] * data_y[idx]) / (
                        data_t[idx] - data_t[0]
                )
                linear = torch.as_tensor(k) * t + torch.as_tensor(b)
                factor = torch.tanh(t) * torch.tanh(idx - t)
                y2 = linear + factor * torch.Tensor([1, 1, 1, 1]) * y
                y2[:4, :] = torch.Tensor(y_real[:4, :])
                y2[-5:, :] = torch.Tensor(y_real[-5:, :])
                return y2

            if tran_in:
                self.net.apply_feature_transform(feature_transform)
            if tran_out:
                self.net.apply_output_transform(output_transform)

        model = dde.Model(data, self.net)
        model.compile(self.method, lr=self.lr,
                    #    loss_weights=[1,10,1,1,1], 
                       external_trainable_variables=self.variable_list)

        self.model = model
        self.x = x
        self.y = y
        self.observe_t = observe_t

    def train(self):
        fnamevar = self.fnamevar_path
        variable = dde.callbacks.VariableValue(self.variable_list, period=self.period, filename=fnamevar)
        if self.index is None:
            check_point_path = 'save_model/model_' + self.date + '.ckpt'
        else:
            check_point_path = 'save_model/model_' + self.date + '_' + str(self.index) + '.ckpt'
        checker= dde.callbacks.ModelCheckpoint(check_point_path, save_better_only = True, period = 50000)
        self.model.train(iterations=self.iterations, callbacks=[variable, checker])

    # def load_model(self, model_path, load_fnamevar_path = None):
    #     if load_fnamevar_path is None:
    #         try:
    #             load_fnamevar_path = self.fnamevar_path
    #             lines = open(load_fnamevar_path, "r").readlines()
    #             Chat = np.array(
    #                 [
    #                     np.fromstring(
    #                         min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
    #                         sep=",",
    #                     )
    #                     for line in lines
    #                 ]
    #             )
    #             self.init_PQkp = Chat[-1, 0]
    #             self.init_PQki = Chat[-1, 1]
    #             self.init_Ikp = Chat[-1, 2]
    #             self.init_Iki = Chat[-1, 3]

    #             print('Load variable values from ' + load_fnamevar_path)

    #         except:
    #             pass
    #     else:
    #         lines = open(load_fnamevar_path, "r").readlines()
    #         Chat = np.array(
    #             [
    #                 np.fromstring(
    #                     min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
    #                     sep=",",
    #                 )
    #                 for line in lines
    #             ]
    #         )
    #         self.init_PQkp = Chat[-1, 0]
    #         self.init_PQki = Chat[-1, 1]
    #         self.init_Ikp = Chat[-1, 2]
    #         self.init_Iki = Chat[-1, 3]

        # PQkp = dde.Variable(self.init_PQkp)
        # PQki = dde.Variable(self.init_PQki)
        # Ikp = dde.Variable(self.init_Ikp)
        # Iki = dde.Variable(self.init_Iki)
        # self.variable_list = [PQkp, PQki, Ikp, Iki]
        # self.model.compile(self.method, lr=self.lr, external_trainable_variables=self.variable_list)
        # self.model.restore(model_path, verbose=1)


if __name__ == "__main__":

    input_data_path = 'data/load/flt_ab_test1.csv'

    microgird = Load(path=input_data_path,
                    # st=0.2,
                    # et=0.5,
                    # index = None
                    )

    # # microgird.build()
    microgird.build(transform='all',
                    net=dde.nn.FNN([6] + [100] * 3 + [4], "swish", "Glorot uniform"),
                    lr=1e-3,
                    iterations=6000000)
    # microgird.load_model('save_model/model_testNone.ckpt-200000.pt',
    #                      load_fnamevar_path = 'Load_variables_ModelSave_0301.dat')
    microgird.train()
    # print(microgird.input_data.shape)
    # y_hat = microgird.model.predict(microgird.observe_t)
    # plt.plot(y_hat, microgird.observe_t)
