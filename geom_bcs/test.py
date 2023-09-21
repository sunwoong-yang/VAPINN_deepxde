import deepxde as dde

import numpy as np
import copy
from utils import update_collocation, plot_pts, plot_flowfield, eval_pde_loss
from Lid_Driven import liddriven_pde, get_geom_bcs

# Data
geom, bcs = get_geom_bcs()
data = dde.data.PDE(geom, liddriven_pde, bcs, num_domain=1000, num_boundary=200, num_test=5000, train_distribution='LHS')

# Model
layer_size = [2] + [20]*5 + [3]
activation = 'tanh'
initializer = 'Glorot uniform'

########################### Vanilla PINN ###########################

# Compile & Train - ADAM
'''
For more options: https://deepxde.readthedocs.io/en/latest/modules/deepxde.html#module-deepxde.model
'''
# VA Model
net = dde.nn.FNN(layer_size, activation, initializer)

# Compile & Train - ADAM
data_vanilla = copy.deepcopy(data)
model_vanilla = dde.Model(data_vanilla, net)
model_vanilla.compile("adam", lr=1e-3)
losshistory, train_state = model_vanilla.train(iterations = 10000, display_every = 1000)

update_collocation(model_vanilla, data_vanilla, N_adapt=0)
losshistory, train_state = model_vanilla.train(iterations = 10000, display_every = 1000)

update_collocation(model_vanilla, data_vanilla, N_adapt=0)
losshistory, train_state = model_vanilla.train(iterations = 10000, display_every = 1000)

# Compile & Train - L-BFGS-B
# dde.optimizers.set_LBFGS_options(maxiter=10000)
# model_vanilla.compile(optimizer = 'L-BFGS-B')
# losshistory, train_state = model_vanilla.train(display_every = 1000)