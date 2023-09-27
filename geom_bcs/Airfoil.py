import deepxde as dde

import numpy as np


'''
Based on these reference codes
[1] Airfoil problem: https://arc.aiaa.org/doi/abs/10.2514/6.2022-0187
[2] Visualization: https://github.com/lululxvi/deepxde/issues/634
'''

xmin, xmax = -10., 20
ymin, ymax = -10., 10.


def boundaryNACA4D(M, P, SS, c, n, offset_x, offset_y):
	"""
	Compute the coordinates of a NACA 4-digits airfoil

	Args:
		M:  maximum camber value (*100)
		P:  position of the maximum camber alog the chord (*10)
		SS: maximum thickness (*100)
		c:  chord length
		n:  the total points sampled will be 2*n
	"""
	m = M / 100
	p = P / 10
	t = SS / 100

	if (m == 0):
		p = 1

	# Chord discretization (cosine discretization)
	xv = np.linspace(0.0, c, n + 1)
	xv = c / 2.0 * (1.0 - np.cos(np.pi * xv / c))

	# Thickness distribution
	ytfcn = lambda x: 5 * t * c * (0.2969 * (x / c) ** 0.5 -
	                               0.1260 * (x / c) -
	                               0.3516 * (x / c) ** 2 +
	                               0.2843 * (x / c) ** 3 -
	                               0.1015 * (x / c) ** 4)
	yt = ytfcn(xv)

	# Camber line
	yc = np.zeros(np.size(xv))

	for ii in range(n + 1):
		if xv[ii] <= p * c:
			yc[ii] = c * (m / p ** 2 * (xv[ii] / c) * (2 * p - (xv[ii] / c)))
		else:
			yc[ii] = c * (m / (1 - p) ** 2 * (1 + (2 * p - (xv[ii] / c)) * (xv[ii] / c) - 2 * p))

	# Camber line slope
	dyc = np.zeros(np.size(xv))

	for ii in range(n + 1):
		if xv[ii] <= p * c:
			dyc[ii] = m / p ** 2 * 2 * (p - xv[ii] / c)
		else:
			dyc[ii] = m / (1 - p) ** 2 * 2 * (p - xv[ii] / c)

	# Boundary coordinates and sorting
	th = np.arctan2(dyc, 1)
	xU = xv - yt * np.sin(th)
	yU = yc + yt * np.cos(th)
	xL = xv + yt * np.sin(th)
	yL = yc - yt * np.cos(th)

	x = np.zeros(2 * n + 1)
	y = np.zeros(2 * n + 1)

	for ii in range(n):
		x[ii] = xL[n - ii]
		y[ii] = yL[n - ii]

	x[n: 2 * n + 1] = xU
	y[n: 2 * n + 1] = yU

	return np.vstack((x + offset_x, y + offset_y)).T

def get_airfoil_geom_bcs(AoA=0):
	# Geometry
	farfield = dde.geometry.Rectangle([xmin, ymin], [xmax, ymax])
	airfoil = dde.geometry.Polygon(boundaryNACA4D(0, 0, 12, 1, 150, 0, 0))
	geom = dde.geometry.CSGDifference(farfield, airfoil)

	# BC
	def boundary_inlet_top_bottom(X, on_boundary):
		x, y = X
		return on_boundary and (np.isclose(x, xmin) or np.isclose(y, ymax) or np.isclose(y, ymin))

	def boundary_outlet(X, on_boundary):
		x, y = X
		return on_boundary and np.isclose(x, xmax)

	def boundary_airfoil(X, on_boundary):
		x, y = X
		return on_boundary and (not farfield.on_boundary((x, y)))

	bc_inlet_top_bottom_u = dde.DirichletBC(geom, (lambda _: 1*np.cos(AoA*np.pi/180)), boundary_inlet_top_bottom, component=0)
	bc_inlet_top_bottom_v = dde.DirichletBC(geom, (lambda _: 1*np.sin(AoA*np.pi/180)), boundary_inlet_top_bottom, component=1)
	bc_outlet_p = dde.DirichletBC(geom, (lambda _: 0), boundary_outlet, component=2)
	bc_airfoil_u = dde.DirichletBC(geom, (lambda _: 0), boundary_airfoil, component=0)
	bc_airfoil_v = dde.DirichletBC(geom, (lambda _: 0), boundary_airfoil, component=1)
	bcs = [bc_inlet_top_bottom_u, bc_inlet_top_bottom_v, bc_outlet_p, bc_airfoil_u, bc_airfoil_v]

	return geom, bcs


