import deepxde as dde

import numpy as np

'''
Based on these reference codes
[1] Cylinder problem: https://arxiv.org/pdf/2306.00230.pdf (Re200 section)
[2] Overall code structure is referred from: https://github.com/lululxvi/deepxde/issues/634
'''

# xmin, xmax = -8., 25
# ymin, ymax = -8., 8.
xmin, xmax = -5., 20 # Ref 논문보다 줄여봄 (너무 과한듯해서)
ymin, ymax = -5., 5. # Ref 논문보다 줄여봄 (너무 과한듯해서)
circ_center = [0, 0]
circ_radi = 0.5
# rho  = 1.0
# mu   = 0.005 # Re=200
def get_cylinder_geom_bcs():
	# Geometry
	farfield = dde.geometry.Rectangle([xmin, ymin], [xmax, ymax])
	airfoil = dde.geometry.Disk(circ_center, circ_radi)
	geom = dde.geometry.CSGDifference(farfield, airfoil)
	timedomain = dde.geometry.TimeDomain(0, 150)
	geomtime = dde.geometry.GeometryXTime(geom, timedomain)

	# BC
	def boundary_inlet(X, on_boundary):
		x, y, t = X
		return on_boundary and np.isclose(x, xmin)

	def boundary_top_bottom(X, on_boundary):
		x, y, t = X
		return on_boundary and (np.isclose(y, ymax) or np.isclose(y, ymin))

	def boundary_outlet(X, on_boundary):
		x, y, t = X
		return on_boundary and np.isclose(x, xmax)

	def boundary_airfoil(X, on_boundary):
		x, y, t = X
		return on_boundary and (not farfield.on_boundary((x, y)))

	def initial(X, on_initial):
		x, y, t = X
		return on_initial and np.isclose(t, 0)

	bc_inlet_u = dde.DirichletBC(geomtime, (lambda _: 1), boundary_inlet, component=0)
	bc_inlet_v = dde.DirichletBC(geomtime, (lambda _: 0), boundary_inlet, component=1)
	bc_top_bottom_u = dde.DirichletBC(geomtime, (lambda _: 1), boundary_top_bottom, component=0)
	bc_top_bottom_v = dde.DirichletBC(geomtime, (lambda _: 0), boundary_top_bottom, component=1)
	bc_outlet = dde.DirichletBC(geomtime, (lambda _: 0), boundary_outlet, component=2)
	bc_airfoil_u = dde.DirichletBC(geomtime, (lambda _: 0), boundary_airfoil, component=0)
	bc_airfoil_v = dde.DirichletBC(geomtime, (lambda _: 0), boundary_airfoil, component=1)
	ic_u = dde.IC(geomtime, (lambda _: 1), initial, component=0)
	ic_v = dde.IC(geomtime, (lambda _: 0), initial, component=1)

	bcs = [bc_inlet_u, bc_inlet_v, bc_top_bottom_u, bc_top_bottom_v, bc_outlet, bc_airfoil_u, bc_airfoil_v, ic_u, ic_v]

	return geomtime, bcs