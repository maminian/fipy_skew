import fipy
import gen_mesh
import utils

import numpy as np
from matplotlib import pyplot

try:
    import cmocean
    my_cm = cmocean.cm.matter_r
except:
    my_cm = pyplot.cm.viridis
#

hw = 2          # half-width; 1 for square
trim = 1./2     # size of corner trim

pts = np.array([
[-hw, 1-trim],
[-hw+trim,1],
[hw-trim,1],
[hw,1-trim],
[hw,-1+trim],
[hw-trim,-1],
[-hw+trim,-1],
[-hw,-1+trim]
])

cellSize = 0.1

mesh = gen_mesh.gen_mesh(pts,cellSize)

flow = fipy.CellVariable(name = r"$u$ (flow)", mesh=mesh, value=0.)

flow.constrain(0., mesh.exteriorFaces) # dirichlet boundary condition

forcing = fipy.CellVariable(mesh=mesh, name='pressure gradient', value=2.)

eq = (-fipy.DiffusionTerm(coeff=1.) == forcing) # set up poisson eqn
eq.solve(var=flow)

# antialiased=True makes the element faces (triangle edges) visible with 
# gaps between triangles. You can remove this kwarg and do something 
# like ax.collections[0].set_edgecolors('w'), but note that edgecolor
# transparency on its own doesn't seem to be supported in matplotlib 3.1.0.

fig,ax = utils.vis_fe_2d(flow, antialiased=True, cmap=my_cm)

fig.set_figheight(6)
fig.set_figwidth(10)
ax.axis('equal')

fig.tight_layout()

####

fig.show()
pyplot.ion()

