import pandas as pd
import deepxde as dde
import numpy as np
import re
import datetime
import os
import time
import torch

class Load:
    def __init__(self, path=None, date=None, index=None, **kwargs):
        self.st = 0.
        self.et = 1.
        self.num_domain = 0
        self.num_boundary = 0
        self.num_test = None

        self.net = dde.nn.FNN([1] + [100] * 3 + [21], "swish", "Glorot uniform")

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
        self.input_data['edp'] = 0
        self.input_data['eqp'] = 0
        self.initialize_variables()

    def initialize_variables(self):
        self.Pzip = dde.Variable(self.init_Pzip)
        self.Qzip = dde.Variable(self.init_Qzip)
        self.alpha_1 = dde.Variable(self.init_alpha_1)
        self.alpha_2 = dde.Variable(self.init_alpha_2)
        self.alpha_4 = dde.Variable(self.init_alpha_4)
        self.alpha_5 = dde.Variable(self.init_alpha_5)
        self.a = dde.Variable(self.init_a)
        self.b = dde.Variable(self.init_b)
        self.c = dde.Variable(self.init_c)
        self.Hm = dde.Variable(self.init_Hm)
        self.xr = dde.Variable(self.init_xr)
        self.rr = dde.Variable(self.init_rr)
        self.xs = dde.Variable(self.init_xs)
        self.rs = dde.Variable(self.init_rs)
        self.xmu = dde.Variable(self.init_xmu)
        self.R_line = dde.Variable(self.init_R_line)
        self.X_line = dde.Variable(self.init_X_line)

        self.variable_list = [self.Pzip, self.Qzip, self.alpha_1, self.alpha_2, self.alpha_4, self.alpha_5, self.a, self.b, self.c, self.Hm, self.xr, self.xs, self.rr, self.xmu, self.rs, self.R_line, self.X_line]

    def check_dat_file(self):
        if os.path.exists(self.fnamevar_path):
            new_index = input("The dat file already exists, please enter a new index to modify, or press Enter to continue---- \n")
            if new_index != '':
                self.index = new_index
                self.fnamevar_path = "Load_variables_ModelSave_" + self.date + "_" + str(new_index) + ".dat"

    def rename_csv(self):
        dict_col = {
            'Time': 'Time',
            'I107d_rms': 'Idrms',
            'I107q_rms': 'Iqrms',
            'Omega107V': 'Ws',
            'Tmpu_M1': 'Tmpu',
            'V107d_rms': 'Vdrms',
            'V107q_rms': 'Vqrms',
            'PSTAT1M1': 'ph',
            'QSTAT1M1': 'qh',
            'SPDOUT1M1': 'Wr',
            'TELECT1M1': 'Te',
            'I1070_rms': 'I0rms',
            'V1070_rms': 'V0rms',
            'P_POI1': 'P_poi',
            'Ptot_Area1': 'P_tot1',
            'P_DG': 'P_dg',
            'Q_POI1': 'Q_poi',
            'Qtot_Area1': 'Q_tot1',
            'Q_DG': 'Q_dg',
            'V101RMS': 'V101'
        }
        self.input_data.rename(columns=lambda x: dict_col[x.split('|')[-1]] if x.split('|')[-1] in dict_col.keys() else x, inplace=True)
        self.input_data = self.input_data[list(dict_col.values())]
        self.input_data.set_index('Time', inplace=True)

    def set_variable(self):
        self.init_Pzip = 1.
        self.init_Qzip = 1.
        self.init_alpha_1 = 1.
        self.init_alpha_2 = 1.
        self.init_alpha_4 = 1.
        self.init_alpha_5 = 1.
        self.init_a = 1.
        self.init_b = 1.
        self.init_c = 1.
        self.init_Hm = 1.
        self.init_xr = 1e-1
        self.init_rr = 4.5768e-2
        self.init_xs = 1e-1
        self.init_rs = 3e-2
        self.init_xmu = 9.
        self.init_R_line = 1.
        self.init_X_line = 1.

    def build(self, transform=None, **kwargs):
        if transform == ('input' or 'all'):
            self.net = dde.nn.FNN([6] + [100] * 3 + [4], "tanh", "Glorot uniform")
        self.__dict__.update(kwargs)
        input_data = self.input_data
        x = input_data.index.to_numpy()
        geom = dde.geometry.TimeDomain(0, x[-1])
        y = input_data.to_numpy()
        global y0
        y0 = y[0, :]
        observe_t = x.reshape(-1, 1)
        n = y.shape[0]
        y_real = y

        def boundary(_, on_initial):
            return on_initial

        observe_y0 = dde.icbc.PointSetBC(observe_t, y[:, 8:9], component=8)  # Wr
        observe_y1 = dde.icbc.PointSetBC(observe_t, y[:, 3:4], component=3)  # Tm
        observe_y2 = dde.icbc.PointSetBC(observe_t, y[:, 9:10], component=9)  # Te
        observe_y3 = dde.icbc.PointSetBC(observe_t, y[:, 4:5], component=4)  # Ud
        observe_y4 = dde.icbc.PointSetBC(observe_t, y[:, 5:6], component=5)  # Uq
        observe_y5 = dde.icbc.PointSetBC(observe_t, y[:, 0:1], component=0)  # Id
        observe_y6 = dde.icbc.PointSetBC(observe_t, y[:, 1:2], component=1)  # Iq
        observe_y7 = dde.icbc.PointSetBC(observe_t, y[:, 18:19], component=18)  # Vbus
        observe_y8 = dde.icbc.PointSetBC(observe_t, y[:, 13:14], component=13)  # Ptot
        observe_y9 = dde.icbc.PointSetBC(observe_t, y[:, 16:17], component=16)  # Qtot
        observe_y10 = dde.icbc.PointSetBC(observe_t, y[:, 12:13], component=12)  # Ppoi
        observe_y11 = dde.icbc.PointSetBC(observe_t, y[:, 15:16], component=15)  # Qpoi
        observe_y12 = dde.icbc.PointSetBC(observe_t, y[:, 14:15], component=14)  # Pdg
        observe_y13 = dde.icbc.PointSetBC(observe_t, y[:, 17:18], component=17)  # Qdg
        observe_y14 = dde.icbc.PointSetBC(observe_t, y[:, 6:7], component=6)  # Pim
        observe_y15 = dde.icbc.PointSetBC(observe_t, y[:, 7:8], component=7)  # Qim

        data = dde.data.PDE(
            geom,
            self.ode,
            [observe_y0, observe_y1, observe_y1, observe_y2, observe_y3, observe_y4, observe_y5, observe_y6, observe_y7, observe_y8, observe_y9, observe_y10, observe_y11, observe_y12, observe_y13, observe_y14, observe_y15],
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
            y_0 = torch.as_tensor(y0)
            def output_transform(t, y):
                idx = n - 1
                k = (data_y[idx] - data_y[0]) / (data_t[idx] - data_t[0])
                b = (data_t[idx] * data_y[0] - data_t[0] * data_y[idx]) / (data_t[idx] - data_t[0])
                linear = torch.as_tensor(k) * t + torch.as_tensor(b)
                factor = torch.tanh(t) * torch.tanh(idx - t)
                y2 = linear + factor * torch.Tensor([1 for _ in range(21)]) * y
                return y_0 * torch.exp(-t) + (1-torch.exp(-t))*y2

            if tran_in:
                self.net.apply_feature_transform(feature_transform)
            if tran_out:
                self.net.apply_output_transform(output_transform)

        model = dde.Model(data, self.net)
        model.compile(self.method, lr=self.lr, external_trainable_variables=self.variable_list)

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
        checker= dde.callbacks.ModelCheckpoint(check_point_path, save_better_only = True, period = 100000)

        for _ in range(500):
            # rs--> 14,   2e-2 <= rs <= 5e-2
            self.variable_list[14].assign_(torch.clamp(self.variable_list[14], 2e-2, 5e-2))

            # rr--> 12,   2e-2 <= rr <= 5e-2
            self.variable_list[12].assign_(torch.clamp(self.variable_list[12], 2e-2, 5e-2))

            # xs--> 11,   1e-1 <= xs <= 5e-1
            self.variable_list[11].assign_(torch.clamp(self.variable_list[11], 1e-1, 5e-1))

            # xr--> 10,   1e-1 <= xr <= 5e-1
            self.variable_list[10].assign_(torch.clamp(self.variable_list[10], 1e-1, 5e-1))

            # xmu--> 13,   0.75 <= xmu <= 10
            self.variable_list[13].assign_(torch.clamp(self.variable_list[13], 0.75, 10))

            # Hm--> 9,   0.05 <= Hm <= 1
            self.variable_list[9].assign_(torch.clamp(self.variable_list[9], 0.05, 1))

            # Start Training
            self.model.train(iterations=int(self.iterations/500), callbacks=[variable, checker])

    def load_model(self, model_path, load_fnamevar_path = None):
        if load_fnamevar_path is None:
            try:
                load_fnamevar_path = self.fnamevar_path
                lines = open(load_fnamevar_path, "r").readlines()
                Chat = np.array(
                    [
                        np.fromstring(
                            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
                            sep=",",
                        )
                        for line in lines
                    ]
                )

                self.init_Pzip = Chat[-1, 0]
                self.init_Qzip = Chat[-1, 1]
                self.init_alpha_1 = Chat[-1, 2]
                self.init_alpha_2 = Chat[-1, 3]
                self.init_alpha_4 = Chat[-1, 4]
                self.init_alpha_5 = Chat[-1, 5]
                self.init_a = Chat[-1, 6]
                self.init_b = Chat[-1, 7]
                self.init_c = Chat[-1, 8]
                self.init_Hm = Chat[-1, 9]
                self.init_xr = Chat[-1, 10]
                self.init_xs = Chat[-1, 11]
                self.init_rr = Chat[-1, 12]
                self.init_xmu = Chat[-1, 13]
                self.init_rs = Chat[-1, 14]
                self.init_R_line = Chat[-1, 15]
                self.init_X_line = Chat[-1, 16]

                print('Load variable values from ' + load_fnamevar_path)

            except:
                pass
        else:
            lines = open(load_fnamevar_path, "r").readlines()
            Chat = np.array(
                [
                    np.fromstring(
                        min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
                        sep=",",
                    )
                    for line in lines
                ]
            )
            self.init_Pzip = Chat[-1, 0]
            self.init_Qzip = Chat[-1, 1]
            self.init_alpha_1 = Chat[-1, 2]
            self.init_alpha_2 = Chat[-1, 3]
            self.init_alpha_4 = Chat[-1, 4]
            self.init_alpha_5 = Chat[-1, 5]
            self.init_a = Chat[-1, 6]
            self.init_b = Chat[-1, 7]
            self.init_c = Chat[-1, 8]
            self.init_Hm = Chat[-1, 9]
            self.init_xr = Chat[-1, 10]
            self.init_xs = Chat[-1, 11]
            self.init_rr = Chat[-1, 12]
            self.init_xmu = Chat[-1, 13]
            self.init_rs = Chat[-1, 14]
            self.init_R_line = Chat[-1, 15]
            self.init_X_line = Chat[-1, 16]
            self.initialize_variables()

if __name__ == "__main__":
    # observation data file
    input_data_path = 'Load_data/0616_island_IM.csv'

    microgird = Load(path=input_data_path,
                     st=0.02,
                     et=0.05,
                     index = 'Smaller_Domain'
                    )

    microgird.build(transform='all',
                    net=dde.nn.FNN([6] + [128] * 3 + [21], "swish", "Glorot uniform"),
                    lr=1e-6,
                    iterations=5000000)

    microgird.train()
