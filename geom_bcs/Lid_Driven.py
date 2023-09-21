import deepxde as dde

import numpy as np
import copy
from utils import update_collocation, plot_pts, plot_flowfield, eval_pde_loss

'''
Based on these reference codes
[1] Lid-driven cavity problem: https://github.com/i207M/PINNacle/blob/595ab6898a30d27ac6cd44ff0a465482f8c52f5c/src/pde/ns.py
[2] Visualization: https://github.com/lululxvi/deepxde/issues/634
'''

def get_liddriven_geom_bcs():
	# Geometry
	bbox = [0, 1, 0, 1]
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

	return geom, bcs


