import fipy

from fipy import CellVariable, Gmsh2D, TransientTerm, DiffusionTerm, Viewer
from fipy.tools import numerix

import numpy as np
from matplotlib import pyplot

import cmocean  # just for colormap

# laplace(phi) = -2; phi = 0 on boundary
# triangular domain

cellSize = 0.1
tHeight = np.sqrt(3)

# weird variable substitution, can this be done more transparently?
tmesh = Gmsh2D('''
cellSize = %(cellSize)g;
tHeight = %(tHeight)g;
Point(1) = {-1,0,0,cellSize};
Point(2) = {1,0,0,cellSize};
Point(3) = {0,tHeight,0,cellSize};
Line(4) = {1,2};
Line(5) = {2,3};
Line(6) = {3,1};
Line Polygon(7) = {4,5,6};
Plane Surface(8) = {7};
''' % locals())


tphi = CellVariable(name = "solution variable", mesh=tmesh, value=0.)

tphi.constrain(0., tmesh.exteriorFaces) # dirichlet boundary condition

forcing = CellVariable(mesh=tmesh, name='pressure', value=-2)

eq = DiffusionTerm(coeff=1.) == forcing # set up poisson eqn
eq.solve(var=tphi)

viewer = Viewer(vars=tphi, datamin=0., datamax=tphi.value.max()) # vis

# blerg
fig = pyplot.gcf()
ax = fig.axes[0]

ax.collections[0].set_facecolors(cmocean.cm.ice(9./2*tphi.value)) # for fun
ax.collections[0].set_edgecolors([0.8,0.8,0.8,0.1])

# more blerg
fig.axes[1].remove()
fig.tight_layout()
ax.axis('square')

fig.set_figheight(8)
fig.set_figwidth(8)
fig.tight_layout()

#fig.savefig('triangle_poiseuille_fipy.png', dpi=120, bbox_inches='tight')

