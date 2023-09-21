import deepxde as dde

import numpy as np
import copy
from utils import update_collocation, plot_pts, plot_flowfield, eval_pde_loss

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

bc_top_u = dde.DirichletBC(geom, (lambda _: 1), boundary_top, component=0)
bc_top_v = dde.DirichletBC(geom, (lambda _: 0), boundary_top, component=1)
bc_wall_u = dde.DirichletBC(geom, (lambda _: 0), boundary_not_top, component=0)
bc_wall_v = dde.DirichletBC(geom, (lambda _: 0), boundary_not_top, component=1)
bcs = [bc_top_u, bc_top_v, bc_wall_u, bc_wall_v]
# bcs = [bc_wall_u, bc_wall_v]

# Data
data = dde.data.PDE(geom, liddriven_pde, bcs, num_domain=1000, num_boundary=200, num_test=5000, train_distribution='LHS')

# Model
layer_size = [2] + [20]*5 + [3]
activation = 'tanh'
initializer = 'Glorot uniform'

# Hard-constraints
# def output_transform(x, u):
#     '''
#     Hard-constraints are imposed only on the top plane
#     : when y=1 -> u=1 & v=0
#     '''
#     u_x = u[:, [0]] * (x[:, [1]] - 1) + 1
#     u_v = u[:, [1]] * (x[:, [1]] - 1)
#     return torch.concat((u_x, u_v, u[:,[2]]), axis=1)
#
# net.apply_output_transform(output_transform)


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
losshistory, train_state = model_vanilla.train(iterations = 5000, display_every = 1000)

update_collocation(model_vanilla, data_vanilla, N_adapt=0)
losshistory, train_state = model_vanilla.train(iterations = 5000, display_every = 1000)

update_collocation(model_vanilla, data_vanilla, N_adapt=0)
losshistory, train_state = model_vanilla.train(iterations = 5000, display_every = 1000)

# Compile & Train - L-BFGS-B
dde.optimizers.set_LBFGS_options(maxiter=10000)
model_vanilla.compile(optimizer = 'L-BFGS-B')
losshistory, train_state = model_vanilla.train(display_every = 1000)

########################### VA-PINN ###########################

# VA Model
net_VA = dde.nn.FNN(layer_size, activation, initializer)

# Compile & Train - ADAM
data_VA = copy.deepcopy(data)
model_VA = dde.Model(data_VA, net_VA)
model_VA.compile("adam", lr=1e-3)
losshistory, train_state = model_VA.train(iterations = 5000, display_every = 1000)

# Implement VA algorithm
N_adapt = 500
update_collocation(model_VA, data_VA, N_adapt=N_adapt, criterion="v")
plot_pts(data_VA, N_adapt=N_adapt, tag="V0")
losshistory, train_state = model_VA.train(iterations = 5000, display_every = 1000)

update_collocation(model_VA, data_VA, N_adapt=N_adapt, criterion="v")
plot_pts(data_VA, N_adapt=N_adapt, tag="V1")
losshistory, train_state = model_VA.train(iterations = 5000, display_every = 1000)

# Compile & Train - L-BFGS-B
dde.optimizers.set_LBFGS_options(maxiter=10000)
model_VA.compile(optimizer = 'L-BFGS-B')
losshistory, train_state = model_VA.train(display_every = 1000)

########################### advPA-PINN ###########################

# VA Model
net_PA = dde.nn.FNN(layer_size, activation, initializer)

# Compile & Train - ADAM
data_PA = copy.deepcopy(data)
model_PA = dde.Model(data_PA, net_PA)
model_PA.compile("adam", lr=1e-3)
losshistory, train_state = model_PA.train(iterations = 5000, display_every = 1000)

# Implement VA algorithm
update_collocation(model_PA, data_PA, N_adapt=N_adapt, criterion="p")
plot_pts(data_PA, N_adapt=N_adapt, tag="P0")
losshistory, train_state = model_PA.train(iterations = 5000, display_every = 1000)

update_collocation(model_PA, data_PA, N_adapt=N_adapt, criterion="p")
plot_pts(data_PA, N_adapt=N_adapt, tag="P1")
losshistory, train_state = model_PA.train(iterations = 5000, display_every = 1000)

# Compile & Train - L-BFGS-B
dde.optimizers.set_LBFGS_options(maxiter=10000)
model_PA.compile(optimizer = 'L-BFGS-B')
losshistory, train_state = model_PA.train(display_every = 1000)

########################### VPA-PINN ###########################

# VA Model
net_VPA = dde.nn.FNN(layer_size, activation, initializer)

# Compile & Train - ADAM
data_VPA = copy.deepcopy(data)
model_VPA = dde.Model(data_VPA, net_VPA)
model_VPA.compile("adam", lr=1e-3)
losshistory, train_state = model_VPA.train(iterations = 5000, display_every = 1000)

# Implement VA algorithm
update_collocation(model_VPA, data_VPA, N_adapt=N_adapt, criterion="both")
plot_pts(data_VPA, N_adapt=N_adapt, tag="VP0")
losshistory, train_state = model_VPA.train(iterations = 5000, display_every = 1000)

update_collocation(model_VPA, data_VPA, N_adapt=N_adapt, criterion="both")
plot_pts(data_VPA, N_adapt=N_adapt, tag="VP1")
losshistory, train_state = model_VPA.train(iterations = 5000, display_every = 1000)

# Compile & Train - L-BFGS-B
dde.optimizers.set_LBFGS_options(maxiter=10000)
model_VPA.compile(optimizer = 'L-BFGS-B')
losshistory, train_state = model_VPA.train(display_every = 1000)
########################### Draw flowfield ###########################

x1_test = np.linspace(bbox[0], bbox[1], 101)
x2_test = np.linspace(bbox[2], bbox[3], 102)
X_test = np.zeros((len(x1_test)*len(x2_test), 2))
X_test[:, 0] = np.vstack((x1_test,)*len(x2_test)).reshape(-1)
X_test[:, 1] = np.vstack((x2_test,)*len(x1_test)).T.reshape(-1)

Y_test = model_vanilla.predict(X_test)
Y_test_VA = model_VA.predict(X_test)
Y_test_PA = model_PA.predict(X_test)
Y_test_VPA = model_VPA.predict(X_test)


# plot_flowfield(x1=x1_test, x2=x2_test, y1=Y_test[:,0], y2=Y_test[:,1], tag='Vanilla', stream=True)
plot_flowfield(x1=x1_test, x2=x2_test, y1=Y_test[:,0], y2=Y_test[:,1], tag='Vanilla', stream=False)
# plot_flowfield(x1=x1_test, x2=x2_test, y1=Y_test_VA[:,0], y2=Y_test_VA[:,1], tag='VA', stream=True)
plot_flowfield(x1=x1_test, x2=x2_test, y1=Y_test_VA[:,0], y2=Y_test_VA[:,1], tag='V', stream=False)
plot_flowfield(x1=x1_test, x2=x2_test, y1=Y_test_PA[:,0], y2=Y_test_PA[:,1], tag='P', stream=False)
plot_flowfield(x1=x1_test, x2=x2_test, y1=Y_test_VPA[:,0], y2=Y_test_VPA[:,1], tag='VP', stream=False)

print("**** Vanilla test losses ****")
print(eval_pde_loss(model_vanilla))
print("**** VA test losses ****")
print(eval_pde_loss(model_VA))
print("**** PA test losses ****")
print(eval_pde_loss(model_PA))
print("**** VPA test losses ****")
print(eval_pde_loss(model_VPA))