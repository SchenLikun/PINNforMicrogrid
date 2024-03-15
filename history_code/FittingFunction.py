'''
Created by Likun Chen, 20th July 2023
This File aims to fit the sampling data from the real-world with PINN construction
'''
import deepxde as dde

fname_train = "./TrainSet_P_gen.txt"
fname_test = "./TestSet_P_gen.txt"

data = dde.data.DataSet(
    fname_train=fname_train,
    fname_test=fname_test,
    col_x=(0,),
    col_y=(1,),
    standardize=True,
)

layer_size = [1] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)

model.compile("adam", lr=0.001, metrics=["l2 relative error"])

losshistory, train_state = model.train(iterations=50000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
