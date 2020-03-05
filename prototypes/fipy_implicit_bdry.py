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

# this took a lot of fiddling... going back to square one with 
# these parameterizations.

s = 0.1 # shape parameter
th = np.linspace(0,2*np.pi, 201)[:-1]   # no overlaps allowed

# apparently the negative branch gets the interior...
rsqv = 1-s - np.sqrt((1-s)**2 - 4*s*(1-s)*np.cos(4*th))
rsqv /= (2*s*np.cos(4*th))

xx = np.sqrt(rsqv)*np.cos(th)
yy = np.sqrt(rsqv)*np.sin(th)

# bizarro issues coming up
#xx += 1e-3*np.random.randn(len(xx))
#yy += 1e-3*np.random.randn(len(xx))

pts_raw = np.vstack( [xx,yy] ).T


# trim issues related to removable singularities.
n = len(xx)
dists = np.array( [np.linalg.norm(pts_raw[i] - pts_raw[(i+1)%n]) for i in range(n)] )
keeps = np.where( dists < 1.8*np.median(dists) )[0]

pts = pts_raw[keeps]

#
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
fig.set_figwidth(8)
ax.axis('equal')

fig.tight_layout()

####

fig.show()
pyplot.ion()

