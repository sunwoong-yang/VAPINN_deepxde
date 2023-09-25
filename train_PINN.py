import deepxde as dde
import copy
from utils import update_collocation, plot_pts

def get_NN(layer_size = [2] + [20] * 5 + [3],
              activation = 'tanh',
              initializer = 'Glorot uniform'):
	net = dde.nn.FNN(layer_size, activation, initializer)
	return net

# def train_PINN(layer_size = [2] + [20] * 5 + [3],
#                activation = 'tanh',
#                initializer = 'Glorot uniform',
#                ):
def train_PINN(net, data,
               compile_kwargs = {"optimizer":"adam", "lr":1e-3},
               adam_iterations=[10000],
               N_adapt=100,
               type_adapt=0,
               lbfgs_iterations=None,
               save_tag="",
               initialize_levels=False,
               ):

	# Compile & Train - ADAM
	'''
	For more options: https://deepxde.readthedocs.io/en/latest/modules/deepxde.html#module-deepxde.model
	'''
	# VA Model
	# net = dde.nn.FNN(layer_size, activation, initializer)

	# Compile & Train - ADAM
	data_vanilla = copy.deepcopy(data)
	model_vanilla = dde.Model(data_vanilla, net)
	model_vanilla.compile(**compile_kwargs)

	if lbfgs_iterations is None:
		lbfgs_iterations = [0] * len(adam_iterations)
	elif len(adam_iterations) != len(lbfgs_iterations):
		raise Exception("Different len between adam_iterations & lbfgs_iterations")

	for enu, (adam_iter, lbfgs_iter) in enumerate(zip(adam_iterations, lbfgs_iterations)):

		# Train with Adam
		losshistory, train_state = model_vanilla.train(iterations = adam_iter, display_every = 1e8, model_save_path = "./saved_models/" + save_tag )

		# Train with l-bfgs-b
		if not lbfgs_iter in [0, None]:
			dde.optimizers.set_LBFGS_options(maxiter=lbfgs_iter)
			model_vanilla.compile(optimizer='L-BFGS-B')
			losshistory, train_state = model_vanilla.train(display_every=1e8, model_save_path = "./saved_models/" + save_tag)

		# Do not update collocation in the last iteration
		if enu != len(adam_iterations)-1:
			update_collocation(model_vanilla, data_vanilla, N_adapt=N_adapt, type_adapt=type_adapt)
			plot_pts(data_vanilla, N_adapt=N_adapt, tag=save_tag, type_adapt=type_adapt)

	return model_vanilla, data_vanilla