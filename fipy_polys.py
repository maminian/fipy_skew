import fipy
import gen_mesh

import numpy as np
from matplotlib import pyplot

import cmocean

def colorize(arr):
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

ax.collections[0].set_facecolors(cmocean.cm.haline(colorize(flow.value))) # for fun
ax.collections[0].set_edgecolors(None)

# more blerg
fig.axes[1].remove()
fig.tight_layout()
ax.axis('square')

fig.set_figheight(6)
fig.set_figwidth(6)
fig.tight_layout()

####
