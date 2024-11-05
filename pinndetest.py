# import pinnde.ode_Solvers as ode_Solvers
# import numpy as np
#
# eqn = "utt + 4*u"
# order = 2
# inits = [-2, 10]
# t_bdry = [0,np.pi/4]
# N_pde = 100
# epochs = 1000
#
# mymodel = ode_Solvers.solveODE_BVP(eqn, order, inits, t_bdry, N_pde, epochs)
#
# mymodel.plot_epoch_loss()
#
# mymodel.plot_solution_prediction()
import pinnde.ode_Solvers as ode_Solvers
import numpy as np

eqn = "utt + 4*u"
order = 2
inits = [-2, 10]
t_bdry = [0, np.pi/4]
N_pde = 100
sensor_range = [-2, 2]
num_sensors = 3000
epochs = 1500

mymodel = ode_Solvers.solveODE_DeepONet_BVP(eqn, order, inits, t_bdry, N_pde, sensor_range,
                                            num_sensors, epochs, constraint = "hard")

mymodel.plot_epoch_loss()

mymodel.plot_solution_prediction()