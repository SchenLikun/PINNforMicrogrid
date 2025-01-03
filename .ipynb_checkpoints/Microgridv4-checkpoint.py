#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import deepxde as dde
import numpy as np
import re
import matplotlib.pyplot as plt


# In[2]:


H = dde.Variable(0.0)
D = dde.Variable(0.0)


# In[3]:


# '''
# 24th July, 2023 created by Likun Chen, Wuhan University
# This is for testing the PINN parameter estimation, object: syn-machine
# EXT Parameters:
#     Vi , initial terminal voltage : 1.0 pu
#     Ta , voltage regulator time constant : 0.4 sec
#     Kf , rate feedback gain : 0.03
# SYN Parameters:
#     H , inertia constant : 1.7 MWs/MVA
#     D , synchronous mechanical damping : 0.12 pu/pu
#     Xa , stator leakage reactance : 0.130 pu
#     Xd , d-axis unsaturated reactance : 1.79 pu
#     Xd' , d-axis unsaturated transient reactance : 0.169 pu
#     Xd'' , d-axis unsaturated Sub-Trans reactance : 0.135 pu
#     Xq , q-axis unsaturated reactance : 1.71 pu
#     Xq' , q-axis unsaturated transient reactance : 0.228 pu
#     Xq'' , q-axis unsaturated Sub-Trans reactance : 0.2 pu
# For 2-order equation, the state variables are rev (w) and phase angle (delta),
# parameters waiting estimated are H , D , Eq' = Ed' = const
#     H * d2_delta/dt2 (This is dw/dt) + D * d_delta/dt + P_ex - P_mach = 0
#     P_ex is external power balance and P_mach is the mechanical power
#     time， P_ex, delta, P_mach, dw
# '''


# In[4]:


input_data = pd.read_csv('data/926HDtest.csv')


# In[5]:


step_time = input_data.Time[1] - input_data.Time[0]
st = 0.1 # start time (10% proportion)
et = 0.5 # end time
input_data.Time -= step_time * int(input_data.shape[0] * st)
input_data = input_data[int(input_data.shape[0]*st): int(input_data.shape[0]*et)]


# In[6]:


input_data.set_index('Time', inplace=True)


# In[7]:


input_data.columns=['Pe','w', 'Pt']


# In[8]:


input_data.plot()


# In[9]:


x = input_data.index.to_numpy()
x


# In[10]:


geom = dde.geometry.TimeDomain(0, x[-1])


# In[11]:


def boundary(_, on_initial):
    return on_initial


# In[12]:


x0 = input_data.iloc[0].tolist()


# In[13]:


# Initial conditions
ic1 = dde.icbc.IC(geom, lambda X: x0[0], boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda X: x0[1], boundary, component=1)
ic3 = dde.icbc.IC(geom, lambda X: x0[2], boundary, component=2)


# In[14]:


y = input_data.to_numpy()
y.shape


# In[15]:


observe_t = x.reshape(-1, 1)


# In[16]:


# Get the training data
observe_y0 = dde.icbc.PointSetBC(observe_t, y[:, 0:1], component=0)
observe_y1 = dde.icbc.PointSetBC(observe_t, y[:, 1:2], component=1)
observe_y2 = dde.icbc.PointSetBC(observe_t, y[:, 2:3], component=2)


# In[17]:


# H * d2_delta/dt2 (This is dw/dt) + D * d_delta/dt + P_ex - P_mach = 0
def Microgrid_system(x, y):
    """Modified Lorenz system (with exogenous input).
    H * d2_delta/dt2 (This is dw/dt) + D * d_delta/dt + P_ex - P_mach = 0
    """
    time = x
    Tm, w, Te = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    dw_dt = dde.grad.jacobian(y, x, i=1)
    return [
        Tm-Te - 2*H*dw_dt - D*(w-1)
    ]


# In[18]:


# class ex_func2():
#     def __init__(self, input_data):
#         self.input_data = input_data
#     def fun(self, t):
#         t = t.reshape(-1, )
#         return self.input_data.iloc[t]['Subsystem #1|CTLs|Vars|PGEN'].to_numpy().reshape(-1, 1)


# In[19]:


data = dde.data.PDE(
    geom,
    Microgrid_system,
    [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
    # num_domain=400,
    # num_boundary=3,
    anchors=observe_t,
#     auxiliary_var_function=ex_func2(input_data).fun,
)


# In[20]:


net = dde.nn.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("adam", lr=0.001, external_trainable_variables=[H])



# In[ ]:


fnamevar = "variables_test1.dat"
variable = dde.callbacks.VariableValue([H, D], period=200, filename=fnamevar)
model.train(iterations=80000, callbacks=[variable])


# In[ ]:


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


# In[ ]:


Chat.shape


# In[ ]:


l, c = Chat.shape
plt.plot(range(l), Chat[:, 0], "r-")

plt.plot(range(l), np.ones(Chat[:, 0].shape) * 1.5, "r--")

plt.legend(["Hhat","True H"], loc="right")
plt.xlabel("Epoch")


# In[ ]:


# pd.Series((input_data.Tm - input_data.Te).to_numpy()[1:] / ((input_data.w.to_numpy()[1:] - input_data.w.to_numpy()[:-1]) / (input_data.index[1] - input_data.index[0])) / 2).plot()


# In[ ]:


yhat = model.predict(observe_t)
plt.figure()
plt.plot(observe_t, y, "-", observe_t, yhat, "--")
plt.xlabel("Time")
# plt.legend(["x", "y", "z", "xh", "yh", "zh"])
# plt.title("Training data")
plt.show()


# In[ ]:




