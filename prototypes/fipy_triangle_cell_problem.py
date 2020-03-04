import fipy

from fipy import CellVariable, Gmsh2D, TransientTerm, DiffusionTerm, Viewer
from fipy.tools import numerix

import numpy as np
from matplotlib import pyplot

try:
    import cmocean  # just for colormap
    my_cm = cmocean.cm.ice
    my_cm2 = cmocean.cm.curl
except:
    my_cm = pyplot.cm.Blues
    my_cm2 = pyplot.cm.bwr
#

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

## TODO - better plot command not using their shitty Viewer().

#########

# laplace(phi) = -2; phi = 0 on boundary
# triangular domain

cellSize = 0.1/2.**2
tHeight = np.sqrt(3)
radius = 1.

if True:
    # weird variable substitution, can this be done more transparently?
    tmesh = Gmsh2D('''
    cellSize = %(cellSize)g;
    tHeight = %(tHeight)g;
    Point(1) = {-tHeight,-1,0,cellSize};
    Point(2) = {+tHeight,-1,0,cellSize};
    Point(3) = {0,2,0,cellSize};
    Line(4) = {1,2};
    Line(5) = {2,3};
    Line(6) = {3,1};
    Line Polygon(7) = {4,5,6};
    Plane Surface(8) = {7};
    ''' % locals())
else:
    tmesh = Gmsh2D('''
    cellSize = %(cellSize)g;
    radius = %(radius)g;
    Point(1) = {0,0,0, cellSize};
    Point(2) = {-radius,0,0,cellSize};
    Point(3) = {0,radius,0,cellSize};
    Point(4) = {radius,0,0,cellSize};
    Point(5) = {0,-radius,0,cellSize};
    Circle(6) = {2,1,3};
    Circle(7) = {3,1,4};
    Circle(8) = {4,1,5};
    Circle(9) = {5,1,2};
    Line Loop(10) = {6,7,8,9};
    Plane Surface(11) = {10};
    ''' % locals())
#

flow = CellVariable(name = "solution variable", mesh=tmesh, value=0.)

flow.constrain(0., tmesh.exteriorFaces) # dirichlet boundary condition

forcing = CellVariable(mesh=tmesh, name='pressure gradient', value=2.)

eq = (-DiffusionTerm(coeff=1.) == forcing) # set up poisson eqn
eq.solve(var=flow)

viewer = Viewer(vars=flow, datamin=0., datamax=flow.value.max()) # vis

# blerg
fig = pyplot.gcf()
ax = fig.axes[0]

ax.collections[0].set_facecolors(my_cm(9./2*flow.value)) # for fun
ax.collections[0].set_edgecolors([0.8,0.8,0.8,0.05])

# more blerg
fig.axes[1].remove()
fig.tight_layout()
ax.axis('square')

fig.set_figheight(6)
fig.set_figwidth(6)
fig.tight_layout()

#######

# part 2 - use the flow as the driver for a Neumann problem (cell problem)
# laplace(theta) = tphi, normal_deriv(theta) = 0.
theta = CellVariable(name = r"$g_1$ (cell problem)", mesh=tmesh, value=0.)

# does this set Neumann boundary conditions?
theta.faceGrad.constrain([0], tmesh.exteriorFaces)

# assign arbitrary value to a single thing
# adjusted after the fact by mean-zero solvability condition at next level.
theta.constrain(0, [1 if i==0 else 0 for i in range(len(tmesh.facesLeft))])

# subtract mean from flow
mz_flow = CellVariable(name = "mean-zero flow", mesh=tmesh, value=0.)
mz_flow.value = flow.value - c_average(flow)

# set up cell equation; theta in some texts; g_1 in my thesis.
cell_eq = (-DiffusionTerm(coeff=1.) == mz_flow)

# solve (?)
cell_eq.solve(var=theta)

# adjust theta
mz_theta = CellVariable(name = "mean-zero cell", mesh=tmesh, value=0.)
mz_theta.value = theta.value - c_average(theta)

# visualize
thmin = theta.value.min()
thmax = theta.value.max()

thrange = max(abs(thmin), abs(thmax))

th_viewer = Viewer(vars=theta, datamin=-thrange, datamax=thrange)

fig = pyplot.gcf()
ax = fig.axes[0]

ax.collections[0].set_facecolors(my_cm2((theta.value -thmin)/(thmax-thmin)))
ax.collections[0].set_edgecolors(None)

fig.axes[1].remove()

cellX,cellY = tmesh.cellCenters.value

# look at level curves of the cell problem. Should look perp to boundary.
ax.tricontour(cellX, cellY, theta.value, 11, colors='w')

ax.axis('square')
fig.set_figwidth(6)
fig.set_figheight(6)

fig.tight_layout()

######################

# set up and solve g_2 problem.

ug1 = mz_flow*mz_theta

g2_driver = CellVariable(name = r"$g_2$ driver", mesh=tmesh, value=0.)
g2_driver.value = 2*(ug1.value - c_average(ug1))

g2 = CellVariable(name=r'$g_2$', mesh=tmesh, value=0.)

# set up cell equation; theta in some texts; g_1 in my thesis.
g2_eq = (-DiffusionTerm(coeff=1.) == g2_driver)

# solve (?)
g2_eq.solve(var=g2)

g2min = g2.value.min()
g2max = g2.value.max()

g2range = max(abs(g2min), abs(g2max))

th_viewer = Viewer(vars=g2, datamin=-g2range, datamax=g2range)

fig = pyplot.gcf()
ax = fig.axes[0]

ax.collections[0].set_facecolors(pyplot.cm.PRGn((g2.value -g2min)/(g2max-g2min)))
ax.collections[0].set_edgecolors(None)

fig.axes[1].remove()

cellX,cellY = tmesh.cellCenters.value

# look at level curves of the cell problem. Should look perp to boundary.
ax.tricontour(cellX, cellY, g2.value, 21, colors='k')

ax.axis('square')
fig.set_figwidth(6)
fig.set_figheight(6)

fig.tight_layout()

####

print('========================')

keff = c_average(ug1)
print('keff coefficient: %.7e'%keff)


sknum = 3 * c_average(mz_flow*g2)
skdenom = (2.*keff)**(1.5)
ltskew = sknum/skdenom

print('skew long time coefficient: %.7e'%ltskew)
fig.show()
