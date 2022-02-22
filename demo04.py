# Demo showing of a few things:
# 
# 1. calculation of a flow in irregular domain
# 2. given a particle, find which element it's contained in.
#

import fipy
import gen_mesh
import utils

import numpy as np
from matplotlib import pyplot

try:
    import cmocean
    my_cm = cmocean.cm.haline
except:
    my_cm = pyplot.cm.viridis
#

s = 5
pts = np.array([ [np.cos(th), np.sin(th)] for th in 2*np.pi/s*np.arange(s) ])

cellSize = 0.1

mesh = gen_mesh.gen_mesh(pts,cellSize)

flow = fipy.CellVariable(name = r"$u$ (flow)", mesh=mesh, value=0.)

flow.constrain(0., mesh.exteriorFaces) # dirichlet boundary condition

forcing = fipy.CellVariable(mesh=mesh, name='pressure gradient', value=2.)

eq = (-fipy.DiffusionTerm(coeff=1.) == forcing) # set up poisson eqn
eq.solve(var=flow)

# visualize
fig,ax = pyplot.subplots(1,1, figsize=(6,6))

utils.vis_fe_2d(flow, cbar=False, cmap=my_cm, ax=ax, edgecolor='w', linewidth=0.1)

# Throw some particles in and identify the cells they belong to.
ntp = 1000
test_pts = np.random.randn(ntp,2)

# Identify point cell locations.
mask,idx = utils.locate_cell(test_pts, mesh)


# for some reason, there is a delay in updating the facecolors 
# within the script; at least when working with ipython.
# I've thrown in a pyplot.pause(0.5) to try to get around this; 
# seems to work, but I doubt this is a robust fix.
fig.show()
pyplot.ion()

# 
pyplot.pause(0.1)

# color the cells each of the points belongs to.
fcs = ax.collections[0].get_facecolors()

for m,ii in zip(mask,idx):
    if not m:
        # not in interior; skip
        continue
    else:
#        fcs[ii] = pyplot.cm.magma(np.random.rand())
        fcs[ii] = [0.5,0,0,1]
#

# update facecolors
ax.collections[0].set_facecolors(fcs)

# conveniently, the mask can be used to visualize interior/exterior points too.
ax.scatter(test_pts[:,0], test_pts[:,1], c=mask, cmap=pyplot.cm.Greys_r, s=5, alpha=0.5)

ax.axis('square')
ax.set_xlim([-1.2,1.2])
ax.set_ylim([-1.2,1.2])

fig.tight_layout()

