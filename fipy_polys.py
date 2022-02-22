# Demo showing off solution of flow problem:
#
# Laplacian(u) = -2
# u on the boundary is zero.
#

import fipy
import gen_mesh

import numpy as np
from matplotlib import pyplot

try:
    import cmocean
    my_cm = cmocean.cm.haline
except:
    my_cm = pyplot.cm.viridis
#

def colorize(arr):
    # helper function to replace fipy's colormap; map data to [0,1].
    return (arr - arr.min())/(arr.max() - arr.min())

s = 5
pts = np.array([ [np.cos(th), np.sin(th)] for th in 2*np.pi/s*np.arange(s) ])

cellSize = 0.1

mesh = gen_mesh.gen_mesh(pts,cellSize)

flow = fipy.CellVariable(name = r"$u$ (flow)", mesh=mesh, value=0.)

flow.constrain(0., mesh.exteriorFaces) # dirichlet boundary condition

forcing = fipy.CellVariable(mesh=mesh, name='pressure gradient', value=2.)

eq = (-fipy.DiffusionTerm(coeff=1.) == forcing) # set up poisson eqn
eq.solve(var=flow)

viewer = fipy.Viewer(vars=flow, datamin=0., datamax=flow.value.max()) # vis

# blerg
fig = pyplot.gcf()
ax = fig.axes[0]

ax.collections[0].set_facecolors(my_cm(colorize(flow.value))) # for fun
ax.collections[0].set_edgecolors(None)

# more blerg
fig.axes[1].remove()
fig.tight_layout()
ax.axis('square')

fig.set_figheight(6)
fig.set_figwidth(6)
fig.tight_layout()

####

fig.savefig('pentagon_fe_flow.png', dpi=80, bbox_inches='tight')
