import numpy as np
from scipy.interpolate import griddata
from utils import plot_flowfield
xyz = np.load("./C.npy")
vel = np.load("./U.npy")

x = xyz[:,0]
y = xyz[:,1]
z = xyz[:,2]

x1_test = np.linspace(-0., 1., 101)
x2_test = np.linspace(-0., 1., 101)
X_test = np.zeros((len(x1_test)*len(x2_test), 2))
X_test[:, 0] = np.vstack((x1_test,)*len(x2_test)).reshape(-1)
X_test[:, 1] = np.vstack((x2_test,)*len(x1_test)).T.reshape(-1)

u = vel[:,0]
v = vel[:,1]

u_inter = griddata((x, y), u, X_test, method='linear')
v_inter = griddata((x, y), v, X_test, method='linear')
# vel_inter = np.concatenate((u_inter.reshape(-1,1), v_inter.reshape(-1,1)), axis=1)

plot_flowfield(x1=x1_test, x2=x2_test, y1=u_inter, y2=v_inter, tag='OpenFOAM', stream=False)
plot_flowfield(x1=x1_test, x2=x2_test, y1=u_inter, y2=v_inter, tag='OpenFOAM', stream=True)
