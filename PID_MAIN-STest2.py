import pandas as pd
import deepxde as dde
import numpy as np
import re
import datetime
import os
import time
import matplotlib.pyplot as plt
import torch

class PID():
    def __init__(self, path=None, date=None, index=None, **kwargs):

        self.st = 0.
        self.et = 1.
        self.num_domain = 0
        self.num_boundary = 0
        self.num_test = None

        '''
        输入的维度：时间t + input transform
        '''
        self.net = dde.nn.FNN([1] + [100] * 3 + [5], "swish", "Glorot uniform")

        self.method = "adam"
        self.lr = 0.0001
        self.period = 200
        self.iterations = 20000

        '''
        基本路径设置
        '''
        self.path = path
        self.date = date
        self.index = index
        '''
        保存数据，同时防止数据冲突
        '''
        if date is None:
            now = datetime.datetime.now()
            self.date = now.strftime("%m%d")
        if index is None:
            self.fnamevar_path = "PI_variables_ModelSave_" + self.date + ".dat"
        else:
            self.fnamevar_path = "PI_variables_ModelSave_" + self.date + "_" + str(index) + ".dat"

        self.set_variable()
        self.__dict__.update(kwargs)
        self.iterations = int(self.iterations)
        self.check_dat_file()

        '''
        读取数据
        '''
        input_data = pd.read_csv(path)
        step_time = input_data.Time[1] - input_data.Time[0]
        input_data.Time -= step_time * int(input_data.shape[0] * self.st)
        input_data = input_data[int(input_data.shape[0] * self.st): int(input_data.shape[0] * self.et)]

        self.input_data = input_data

        self.rename_pid_csv()

        PQkp = dde.Variable(self.init_PQkp)
        PQki = dde.Variable(self.init_PQki)
        Ikp = dde.Variable(self.init_Ikp)
        Iki = dde.Variable(self.init_Iki)

        self.variable_list = [PQkp, PQki, Ikp, Iki]

        '''
        这里修改微分方程及其表达式
        对于一个PI环节而言标准表达式为(Input) * Kp + (Input) * Ki * 1/s = (Output)
        i.e., Kp * s * Input + Ki * Input = s * Output
        '''

        def PI_ode(x, y):
            # 确保 x 和 y 都启用 requires_grad
            x = x.clone().detach().requires_grad_(True)
            y = y.clone().detach().requires_grad_(True)

            DeltaQ, Iqref, Iq, DeltaIq, DeltaUq = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4], y[:, 4:5]

            # 使用 torch.autograd.grad 计算 dI_dt1 和 dO_dt1
            dI_dt1 = torch.autograd.grad(outputs=y[:, 0], inputs=x, grad_outputs=torch.ones_like(y[:, 0]), create_graph=True, allow_unused=True)[0]
            dO_dt1 = torch.autograd.grad(outputs=y[:, 1], inputs=x, grad_outputs=torch.ones_like(y[:, 1]), create_graph=True, allow_unused=True)[0]
            dI_dt2 = torch.autograd.grad(outputs=y[:, 3], inputs=x, grad_outputs=torch.ones_like(y[:, 3]), create_graph=True, allow_unused=True)[0]
            dO_dt2 = torch.autograd.grad(outputs=y[:, 4], inputs=x, grad_outputs=torch.ones_like(y[:, 4]), create_graph=True, allow_unused=True)[0]

            '''
            这里修改微分方程及其表达式
            '''
            return [
                -(PQkp * dI_dt1 + DeltaQ * PQki) - dO_dt1,
                (Ikp * dI_dt2 + DeltaIq * Iki) - dO_dt2,
                Iqref - Iq - DeltaIq
            ]

        self.pi_ode = PI_ode
        self.real_value_list = [self.real_PQkp, self.real_PQki, self.real_Ikp, self.real_Iki]

    '''
    确认dat文件不冲突
    '''
    def check_dat_file(self):
        if os.path.exists(self.fnamevar_path):
            new_index = input(
                "The dat file already exists, please enter a new index to modify, or press Enter to continue")
            if new_index != '':
                self.index = new_index
                self.fnamevar_path = "PI_variables_ModelSave_" + self.date + "_" + str(new_index) + ".dat"

    def rename_pid_csv(self):
        dict_col_Q = {
            'Time': 'Time',
            'Qerr8': 'DeltaQ',
            'Isqrefongrid8': 'Iqref',
            'IsqA8': 'Iq',
            'Isqerr8': 'DeltaIq',
            'uqA8': 'DeltaUq'
        }
        self.input_data.rename(columns=lambda x: dict_col_Q[x.split('|')[-1]], inplace=True)
        self.input_data = self.input_data[list(dict_col_Q.values())]
        self.input_data.set_index('Time', inplace=True)

    def set_variable(self):
        self.init_PQkp = 1.
        self.init_PQki = 1.
        self.init_Ikp = 1.
        self.init_Iki = 1.

        self.real_PQkp = 0.025
        self.real_PQki = 5.
        self.real_Ikp = 0.025
        self.real_Iki = 0.5

    def build(self, transform=None, **kwargs):
        if transform == ('input' or 'all'):
            self.net = dde.nn.FNN([6] + [100] * 3 + [5], "tanh", "Glorot uniform")
        self.__dict__.update(kwargs)
        input_data = self.input_data
        x = torch.tensor(input_data.index.to_numpy(), dtype=torch.float32).view(-1, 1)
        geom = dde.geometry.TimeDomain(0, x[-1].item())
        y = torch.tensor(input_data.to_numpy(), dtype=torch.float32)
        observe_t = x

        # 转换为 NumPy 格式并将其移动到 CPU
        observe_t_np = observe_t.cpu().numpy()
        y_np = y.cpu().numpy()

        def boundary(_, on_initial):
            return on_initial

        '''
        这里观测点设置
        '''
        observe_y0 = dde.icbc.PointSetBC(observe_t_np, y_np[:, 0:1], component=0)
        observe_y1 = dde.icbc.PointSetBC(observe_t_np, y_np[:, 1:2], component=1)
        observe_y2 = dde.icbc.PointSetBC(observe_t_np, y_np[:, 2:3], component=2)
        observe_y3 = dde.icbc.PointSetBC(observe_t_np, y_np[:, 3:4], component=3)
        observe_y4 = dde.icbc.PointSetBC(observe_t_np, y_np[:, 4:5], component=4)
        ic0 = dde.icbc.IC(geom, lambda X: y_np[0, 0], boundary, component=0)
        ic1 = dde.icbc.IC(geom, lambda X: y_np[0, 1], boundary, component=1)
        ic2 = dde.icbc.IC(geom, lambda X: y_np[0, 2], boundary, component=2)
        ic3 = dde.icbc.IC(geom, lambda X: y_np[0, 3], boundary, component=3)
        ic4 = dde.icbc.IC(geom, lambda X: y_np[0, 4], boundary, component=4)

        data = dde.data.PDE(
            geom,
            self.pi_ode,
            [ic0, observe_y0, ic1, ic2, observe_y2, ic3, ic4, observe_y4],
            anchors=observe_t_np,
        )

        model = dde.Model(data, self.net)
        model.compile(self.method, lr=self.lr, external_trainable_variables=self.variable_list, loss=self.custom_loss)
        self.model = model
        self.observe_t = observe_t

    def custom_loss(self, y_true, y_pred):
        alpha = 0.1  # 设置alpha的值
        exp_factor = torch.exp(-alpha * self.observe_t ** 2)
        prediction_data_loss = torch.mean((y_true - y_pred) ** 2)
        physics_eq_loss = torch.mean(
            torch.stack([torch.sum(term ** 2) for term in self.pi_ode(self.observe_t, y_pred) if term is not None])
        )
        custom_loss_value = torch.mean((1 - exp_factor) * prediction_data_loss + exp_factor * physics_eq_loss)
        return custom_loss_value

    def train(self):
        fnamevar = self.fnamevar_path
        variable = dde.callbacks.VariableValue(self.variable_list, period=self.period, filename=fnamevar)
        if self.index is None:
            check_point_path = 'save_model/model_' + self.date + '.ckpt'
        else:
            check_point_path = 'save_model/model_' + self.date + '_' + str(self.index) + '.ckpt'
        checker = dde.callbacks.ModelCheckpoint(check_point_path, save_better_only=True, period=50000)
        self.model.train(iterations=self.iterations, callbacks=[variable, checker])


if __name__ == "__main__":
    input_data_path = './data/pi/0122FltQControl.csv'

    microgird = PID(path=input_data_path, index=None, st=0., et=1.)
    microgird.build(transform='all', net=dde.nn.FNN([6] + [100] * 3 + [5], "swish", "Glorot uniform"), lr=1e-3, iterations=5000000)
    microgird.train()
    print(microgird.input_data.shape)
    print(microgird.input_data)
