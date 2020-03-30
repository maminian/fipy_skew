import fipy

from fipy import CellVariable, Gmsh2D, TransientTerm, DiffusionTerm

import numpy as np
from matplotlib import pyplot


import gen_mesh
import utils



import cmocean  # just for colormap
my_cm = cmocean.cm.haline
my_cm2 = cmocean.cm.curl
my_cm3 = pyplot.cm.PRGn


##
# CellVariable tools
#
def c_integrate(cvar):
    # is there a higher-order version of this? 
    # does it make sense to do anything more sophisticated?
    return (cvar.mesh.cellVolumes * cvar.value).sum()
#

def c_average(cvar):
    # There's also a built-in cvar.cellVolumeAverage
    return c_integrate(cvar)/(cvar.mesh.cellVolumes.sum())
#

###

def colorize(arr):
    # helper function to replace fipy's colormap; map data to [0,1].
    return (arr - arr.min())/(arr.max() - arr.min())



##################

s = 5
pts = np.array([ [np.cos(th), np.sin(th)] for th in 2*np.pi/s*np.arange(s) ])

oneish = np.sqrt(10)/3. - 0.054  # something vaguely irrational
cellSize = 0.04*oneish

mesh = gen_mesh.gen_mesh(pts,cellSize)


flow = fipy.CellVariable(name = r"$u$ (flow)", mesh=mesh, value=0.)

flow.constrain(0., mesh.exteriorFaces) # dirichlet boundary condition

forcing = fipy.CellVariable(mesh=mesh, name='pressure gradient', value=2.)

eq = (-fipy.DiffusionTerm(coeff=1.) == forcing) # set up poisson eqn
eq.solve(var=flow)


#######

# part 2 - use the flow as the driver for a Neumann problem (cell problem)
# laplace(theta) = tphi, normal_deriv(theta) = 0.
theta = CellVariable(name = r"$g_1$ (cell problem)", mesh=mesh, value=0.)

# does this set Neumann boundary conditions?
theta.faceGrad.constrain([0], mesh.exteriorFaces)

# assign arbitrary value to a single thing
# adjusted after the fact by mean-zero solvability condition at next level.
theta.constrain(0, [1 if i==0 else 0 for i in range(len(mesh.facesLeft))])

# subtract mean from flow
mz_flow = CellVariable(name = "mean-zero flow", mesh=mesh, value=0.)
mz_flow.value = flow.value - c_average(flow)

# set up cell equation; theta in some texts; g_1 in my thesis.
cell_eq = (-DiffusionTerm(coeff=1.) == mz_flow)

# solve (?)
cell_eq.solve(var=theta)

# adjust theta
mz_theta = CellVariable(name = "mean-zero cell", mesh=mesh, value=0.)
mz_theta.value = theta.value - c_average(theta)


# look at level curves of the cell problem. Should look perp to boundary.
#ax.tricontour(cellX, cellY, mz_theta.value, 11, colors='w')

######################

# set up and solve g_2 problem.

ug1 = mz_flow*mz_theta

g2_driver = CellVariable(name = r"$g_2$ driver", mesh=mesh, value=0.)
g2_driver.value = 2*(ug1.value - c_average(ug1))

g2 = CellVariable(name=r'$g_2$', mesh=mesh, value=0.)

# set up cell equation; theta in some texts; g_1 in my thesis.
g2_eq = (-DiffusionTerm(coeff=1.) == g2_driver)

# solve (?)
g2_eq.solve(var=g2)

g2min = g2.value.min()
g2max = g2.value.max()

g2range = max(abs(g2min), abs(g2max))



# look at level curves of the cell problem. Should look perp to boundary.


####
# big vis

fig,ax = pyplot.subplots(1,3, figsize=(15,5), sharex=True, sharey=True)

utils.vis_fe_2d(flow, antialiased=True, cmap=my_cm, ax=ax[0], cbar=False)
utils.vis_fe_2d(theta, antialiased=True, cmap=my_cm2, ax=ax[1], cbar=False, cstyle='divergent')
utils.vis_fe_2d(g2, antialiased=True, cmap=my_cm3, ax=ax[2], cbar=False, cstyle='divergent')


#

if True:
    cellX,cellY = mesh.cellCenters.value
    ax[0].tricontour(cellX, cellY, flow.value, 11, colors='k')
    
    thmin,thmax = theta.value.min(), theta.value.max()
    thbnd = max(abs(thmin), abs(thmax))
    ax[1].tricontour(cellX, cellY, theta.value, np.linspace(-thbnd,thbnd,15), colors='k')
    
    g2min,g2max = g2.value.min(), g2.value.max()
    g2bnd = max(abs(g2min), abs(g2max))
    ax[2].tricontour(cellX, cellY, g2.value, np.linspace(-g2bnd,g2bnd,15), colors='k')

#
ax[0].axis('square')
ax[1].axis('square')
ax[2].axis('square')
fig.tight_layout()
fig.show()

fig.savefig('spatial_solutions.png', dpi=120, bbox_inches='tight')
