import deepxde as dde

import numpy as np
import torch
'''
Based on these reference codes
[1] Lid-driven cavity problem: https://github.com/i207M/PINNacle/blob/595ab6898a30d27ac6cd44ff0a465482f8c52f5c/src/pde/ns.py
[2] Visualization: https://github.com/lululxvi/deepxde/issues/634
'''
dde.config.set_random_seed(42)

# PDE
def liddriven_pde(x, u):
    nu = 0.01
    u_vel, v_vel, _ = u[:, [0]], u[:, [1]], u[:, [2]]
    u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
    u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

    v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
    v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
    v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

    p_x = dde.grad.jacobian(u, x, i=2, j=0)
    p_y = dde.grad.jacobian(u, x, i=2, j=1)

    momentum_x = (u_vel * u_vel_x + v_vel * u_vel_y + p_x - nu * (u_vel_xx + u_vel_yy))
    momentum_y = (u_vel * v_vel_x + v_vel * v_vel_y + p_y - nu * (v_vel_xx + v_vel_yy))
    continuity = u_vel_x + v_vel_y

    return [momentum_x, momentum_y, continuity]

# Geometry
bbox=[0, 1, 0, 1]
geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])

# BC
def boundary_top(x, on_boundary):
    return on_boundary and np.isclose(x[1], bbox[3])

def boundary_not_top(x, on_boundary):
    return on_boundary and not np.isclose(x[1], bbox[3])
# def funU(x):
#     return 1.0
# def funV(x):
#     return 0.0
# bc_u = dde.DirichletBC(geom, funU, boundary_not_top, component=0)
# bc_v = dde.DirichletBC(geom, funV, boundary_not_top, component=1)
bc_u = dde.DirichletBC(geom, (lambda _: 1), boundary_not_top, component=0)
bc_v = dde.DirichletBC(geom, (lambda _: 0), boundary_not_top, component=1)
bcs = [bc_u, bc_v]

# Data
data = dde.data.PDE(geom, liddriven_pde, bcs, num_domain=2000, num_boundary=400, num_test=5000)

# Model
layer_size = [2] + [20]*5 + [3]
activation = 'tanh'
initializer = 'Glorot uniform'
net = dde.nn.FNN(layer_size, activation, initializer)

# Hard-constraints
def output_transform(x, u):
    '''
    Hard-constraints are imposed only on the top plane
    : when y=1 -> u=1 & v=0
    '''
    u_x = u[:, [0]] * (x[:, [1]] - 1) + 1
    u_v = u[:, [1]] * (x[:, [1]] - 1)
    return torch.concat((u_x, u_v, u[:,[2]]), axis=1)

net.apply_output_transform(output_transform)

# Compilation
'''
For more options: https://deepxde.readthedocs.io/en/latest/modules/deepxde.html#module-deepxde.model
'''
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)

# Train
losshistory, train_state = model.train(iterations = 10000, display_every = 100, model_save_path ='../')
dde.saveplot(losshistory, train_state, issave = True, isplot = True)