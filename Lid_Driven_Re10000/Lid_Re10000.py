import deepxde as dde
import copy
import numpy as np
from utils import update_collocation, plot_pts, plot_flowfield, eval_pde_loss, remove_figs_models
from train_PINN import get_NN, train_PINN
from geom_bcs.Lid_Driven import get_liddriven_geom_bcs

dde.config.set_random_seed(42)
remove_figs_models()

def liddriven_pde(x, u):
    nu = 0.0001
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

geom, bcs = get_liddriven_geom_bcs()
data = dde.data.PDE(geom, liddriven_pde, bcs, num_domain=10000, num_boundary=2000, num_test=5000, train_distribution='LHS')

N_adapt_ = 2000
compile_kwargs_ = {"optimizer":"adam", "lr":1e-3}
adam_iterations_ = [30000]*3
lbfgs_iterations_ = None
# lbfgs_iterations_ = [0,0,25000]
network_kwars_ = {"layer_size" : [2] + [40] * 10 + [3],
              "activation" : 'tanh',
              "initializer" : 'Glorot uniform'}
#########################################################################
"""
Vanilla
"""
net = get_NN(**network_kwars_)


PINN_model, data_updated =  train_PINN(net,
                                       data,
                                       compile_kwargs = compile_kwargs_,
                                       adam_iterations=adam_iterations_,
                                       N_adapt=N_adapt_,
                                       type_adapt=0,
                                       lbfgs_iterations=lbfgs_iterations_,
                                       # lbfgs_iterations=[5,5],
                                       save_tag="Van",
                                       initialize_levels=True
                                       )

#########################################################################
"""
VA (Vorticity-Aware)
"""
net_VA = get_NN(**network_kwars_)

PINN_model_VA, data_updated =  train_PINN(net_VA,
                                       data,
                                       compile_kwargs = compile_kwargs_,
                                       adam_iterations=adam_iterations_,
                                       N_adapt=N_adapt_,
                                       type_adapt=1,
                                       lbfgs_iterations=lbfgs_iterations_,
                                       # lbfgs_iterations=[5,5],
                                       save_tag="VA",
                                       initialize_levels=True
                                       )

#########################################################################
"""
PA (advPressure-Aware)
"""
net_PA = get_NN(**network_kwars_)

PINN_model_PA, data_updated =  train_PINN(net_PA,
                                       data,
                                       compile_kwargs = compile_kwargs_,
                                       adam_iterations=adam_iterations_,
                                       N_adapt=N_adapt_,
                                       type_adapt=2,
                                       lbfgs_iterations=lbfgs_iterations_,
                                       # lbfgs_iterations=[5,5],
                                       save_tag="PA",
                                       initialize_levels=True
                                       )

#########################################################################
"""
VPA (Vorticity-advP-Aware)
"""
net_VPA = get_NN(**network_kwars_)

PINN_model_VPA, data_updated =  train_PINN(net_VPA,
                                       data,
                                       compile_kwargs = compile_kwargs_,
                                       adam_iterations=adam_iterations_,
                                       N_adapt=N_adapt_,
                                       type_adapt=3,
                                       lbfgs_iterations=lbfgs_iterations_,
                                       # lbfgs_iterations=[5,5],
                                       save_tag="VPA",
                                       initialize_levels=True
                                       )

#########################################################################
"""
RA (residual-aware)
"""
net_RA = get_NN(**network_kwars_)

PINN_model_RA, data_updated =  train_PINN(net_RA,
                                       data,
                                       compile_kwargs = compile_kwargs_,
                                       adam_iterations=adam_iterations_,
                                       N_adapt=N_adapt_,
                                       type_adapt=4,
                                       lbfgs_iterations=lbfgs_iterations_,
                                       # lbfgs_iterations=[5,5],
                                       save_tag="RA",
                                       initialize_levels=True
                                       )

#########################################################################
"""
GA (gradient-aware)
"""
net_GA = get_NN(**network_kwars_)

PINN_model_GA, data_updated =  train_PINN(net_GA,
                                       data,
                                       compile_kwargs = compile_kwargs_,
                                       adam_iterations=adam_iterations_,
                                       N_adapt=N_adapt_,
                                       type_adapt=5,
                                       lbfgs_iterations=lbfgs_iterations_,
                                       # lbfgs_iterations=[5,5],
                                       save_tag="GA",
                                       initialize_levels=True
                                       )

########################### Draw flowfield ###########################

x1_test = np.linspace(0, 1, 101)
x2_test = np.linspace(0, 1, 101)
X_test = np.zeros((len(x1_test)*len(x2_test), 2))
X_test[:, 0] = np.vstack((x1_test,)*len(x2_test)).reshape(-1)
X_test[:, 1] = np.vstack((x2_test,)*len(x1_test)).T.reshape(-1)

Y_test = PINN_model.predict(X_test)
Y_test_VA = PINN_model_VA.predict(X_test)
Y_test_PA = PINN_model_PA.predict(X_test)
Y_test_VPA = PINN_model_VPA.predict(X_test)
Y_test_RA = PINN_model_RA.predict(X_test)
Y_test_GA = PINN_model_GA.predict(X_test)

plot_flowfield(x1=x1_test, x2=x2_test, y1=Y_test[:,0], y2=Y_test[:,1], tag='Van', stream=True, initialize_levels=True)
plot_flowfield(x1=x1_test, x2=x2_test, y1=Y_test_VA[:,0], y2=Y_test_VA[:,1], tag='VA', stream=True, initialize_levels=True)
plot_flowfield(x1=x1_test, x2=x2_test, y1=Y_test_PA[:,0], y2=Y_test_PA[:,1], tag='PA', stream=True, initialize_levels=True)
plot_flowfield(x1=x1_test, x2=x2_test, y1=Y_test_VPA[:,0], y2=Y_test_VPA[:,1], tag='VP', stream=True, initialize_levels=True)
plot_flowfield(x1=x1_test, x2=x2_test, y1=Y_test_RA[:,0], y2=Y_test_RA[:,1], tag='RA', stream=True, initialize_levels=True)
plot_flowfield(x1=x1_test, x2=x2_test, y1=Y_test_GA[:,0], y2=Y_test_GA[:,1], tag='GA', stream=True, initialize_levels=True)


print("**** Vanilla test losses ****")
print(eval_pde_loss(PINN_model))
print("**** VA test losses ****")
print(eval_pde_loss(PINN_model_VA))
print("**** PA test losses ****")
print(eval_pde_loss(PINN_model_PA))
print("**** VPA test losses ****")
print(eval_pde_loss(PINN_model_VPA))
print("**** RA test losses ****")
print(eval_pde_loss(PINN_model_RA))
print("**** GA test losses ****")
print(eval_pde_loss(PINN_model_GA))