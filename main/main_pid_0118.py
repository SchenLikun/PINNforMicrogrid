import pandas as pd
import deepxde as dde
import numpy as np
import re
import datetime
import os
import time


class PID():
    def __init__(self, path=None, date=None, index=None, **kwargs):
        # 基本函数设置
        self.st = 0.
        self.et = 1.
        self.num_domain = 0
        self.num_boundary = 0
        self.num_test = None
        self.net = dde.nn.FNN([1] + [100] * 3 + [2], "tanh", "Glorot uniform")
        self.method = "adam"
        self.lr = 0.0001
        self.period = 200
        self.iterations = 20000
        self.init_Kp = 1.
        self.init_Ki = 1.

        # 基本路径设置
        self.path = path
        self.date = date
        # 保存数据，同时防止数据冲突
        if date is None:
            now = datetime.datetime.now()
            self.date = now.strftime("%m%d")
        if index is None:
            self.fnamevar_path = "PI_variables_" + self.date + ".dat"
        else:
            self.fnamevar_path = "PI_variables_" + self.date + "_" + str(index) + ".dat"

        self.__dict__.update(kwargs)
        self.iterations = int(self.iterations)
        self.check_dat_file()

        # 读取数据
        input_data = pd.read_csv(path)
        # PID参数设置文件中I代表input，O代表output
        input_data.columns = ['Time', 'O', 'I']
        # 步长
        step_time = input_data.Time[1] - input_data.Time[0]
        input_data.Time -= step_time * int(input_data.shape[0] * self.st)
        input_data = input_data[int(input_data.shape[0] * self.st): int(input_data.shape[0] * self.et)]

        input_data.set_index('Time', inplace=True)
        # 扩大数据倍数方便计算，同时防止精度丢失，当数据数量级较小的时候使用
        # input_data *= 1e3

        self.input_data = input_data
        self.read_file_info()
        k_p = dde.Variable(self.init_Kp)
        k_i = dde.Variable(self.init_Ki)
        self.variable_list = [k_p, k_i]

        '''
        这里修改微分方程及其表达形式
        对于一个PI环节而言标准表达式为(Input) * Kp + (Input) * Ki * 1/s = (Output)
        i.e., Kp * s * Input + Ki * Input = s * Output
        '''

        def PI_ode(x, y):

            O, I = y[:, 0:1], y[:, 1:2]

            dO_dt = dde.grad.jacobian(y, x, i=0)
            dI_dt = dde.grad.jacobian(y, x, i=1)

            return [
                (k_p * dI_dt + I * k_i) - dO_dt
            ]

        '''
        这里修改微分方程及其表达形式
        '''

        self.pi_ode = PI_ode
        self.real_value_list = [self.real_Kp, self.init_Ki]

    # 确认dat文件不冲突
    def check_dat_file(self):
        if os.path.exists(self.fnamevar_path):
            new_index = input(
                "The dat file already exists, please enter a new index to modify, or press Enter to continue")
            if new_index != '' and self.index is not None:
                self.fnamevar_path = "PI_variables_" + self.date + "_" + str(new_index) + ".dat"
    '''
    # 读取文件信息
    '''
    def read_file_info(self):
        infos = re.findall(r'\d+\.?\d*', self.path.split('/')[-1])
        init_Kp = np.float64(infos[1])
        init_Ki = np.float64(infos[2])
        step = np.float64(infos[3])
        times = np.float64(infos[4])
        self.real_Kp = init_Kp + step * .1 * times
        self.real_Ki = init_Ki + step * times

    def build(self, transform=None, **kwargs):
        self.__dict__.update(kwargs)
        input_data = self.input_data
        x = input_data.index.to_numpy()
        geom = dde.geometry.TimeDomain(0, x[-1])
        y = input_data.to_numpy()
        observe_t = x.reshape(-1, 1)
        n = y.shape[0]
        observe_y0 = dde.icbc.PointSetBC(observe_t, y[:, 0:1], component=0)
        observe_y1 = dde.icbc.PointSetBC(observe_t, y[:, 1:2], component=1)

        data = dde.data.PDE(
            geom,
            self.pi_ode,
            [observe_y0, observe_y1],
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

            '''
            这里是输入转换：Input transform
            '''
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

            '''
            这里是输出转换：Output transform
            '''
            def output_transform(t, y):
                idx = n - 1
                k = (data_y[idx] - data_y[0]) / (data_t[idx] - data_t[0])
                b = (data_t[idx] * data_y[0] - data_t[0] * data_y[idx]) / (
                        data_t[idx] - data_t[0]
                )
                linear = torch.as_tensor(k) * t + torch.as_tensor(b)
                factor = torch.tanh(t) * torch.tanh(idx - t)
                return linear + factor * torch.Tensor([1, 1]) * y

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
        self.model.train(iterations=self.iterations, callbacks=[variable])


if __name__ == "__main__":

    input_data_path = '../data/pi/3phasestableKp0.5Ki5Step0.1No.0.csv'

    microgird = PID(path=input_data_path,
                    init_Kp=1.,
                    init_Ki=1.
                    )

    # microgird.build()
    microgird.build(transform='all',
                    net=dde.nn.FNN([6] + [100] * 3 + [2], "tanh", "Glorot uniform"),
                    lr=1e-3,
                    iterations=1000000)

    microgird.train()
