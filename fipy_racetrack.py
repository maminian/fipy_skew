# Demo showing off a numerical solution 
# of the flow in a racetrack (smooth boundary)
#

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

#
# another hacky approach to getting boundary points.
# take a random point collection, then do rootfinding in the r direction.
#
# secant method with small step sizes seems to do good enough.
#

def bfunc(x,y, lam=0.5, s=0.5):
    return 1. - ( (1-lam**4)/(1-lam**2*s**2) * (x**2 + s**2*y**2) ) - ((lam**2*(lam**2-s**2))/(1-lam**2*s**2)) * (x**4 - 6*x**2*y**2 + y**4)
#


s = 0.3     # shape parameter
lam = 0.5   # aspect ratio
dt = 0.1   # for secant method

n = 201
th = np.linspace(0,2*np.pi, n+1)[:-1]
ri = 0.2*np.ones(n)
ro = 0.5*ri

#fi = bfunc(ri*np.cos(th), ri*np.sin(th), lam,s)
xi,yi = ri*np.cos(th), ri*np.sin(th)
fo = bfunc(ro*np.cos(th), ro*np.sin(th), lam,s)

nit = 200

xhist = []
yhist = []
for i in range(n):
    # fixed iteration of secant method.
    xi,yi = ri*np.cos(th), ri*np.sin(th)
    
    xhist.append(xi)
    yhist.append(yi)
    
    fi = bfunc(xi,yi,lam,s)

    dirs = ( fi - fo )/(ri-ro)
    
    # update
    rtemp = np.array(ri)
    ri = ri - dt*fi/dirs
    ro = rtemp
    fo = np.array(fi)
    
    xo = np.array(xi)
    yo = np.array(yi)
#

#fig,ax = pyplot.subplots(1,1)

#for i in range(4):
#    ax.scatter(xhist[i],yhist[i])
##

#ax.plot(xhist[-1], yhist[-1], c='k', marker='.', markersize=10, lw=0.5)
#fig.show()
#pyplot.ion()

# y-first intentional.
pts = np.vstack([yhist[-1], xhist[-1]]).T

if True:

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

