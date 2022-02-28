# Demo showing how one can feed 
# mesh-valued functions from fipy into one 
# another to successively solve the 
# moment equations
# 
# (ex: solving the cell problem Laplace(theta) = u, Neumann boundaries)
#

import fipy

from fipy import CellVariable, Gmsh2D, TransientTerm, DiffusionTerm

import numpy as np
from matplotlib import pyplot


import gen_mesh
import utils



import cmocean  # just for colormap
import colorcet
my_cm = colorcet.cm.kbc
my_cm2 = colorcet.cm.gwv
my_cm3 = colorcet.cm.cwr


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

#s = 8
#pts = np.array([ [np.cos(th), np.sin(th)] for th in 2*np.pi/s*np.arange(s) ])

import demo09
pts = demo09.coords
pts = pts - np.mean(pts, axis=0)
pts = pts/(np.std(pts, axis=0))

Pe = 10**4

########################

t_short = 10.**np.linspace(-1+np.log10(Pe**-2),1+np.log10(Pe**-1), 41)
t_long = 10.**np.linspace(0,2, 41)

asymptotics = utils.Sk_Asymptotics(pts)
_,_,_,Sk_asymp_short = asymptotics.compute_ST_asymptotics(t_short,Pe=Pe)
_,_,Sk_asymp_long = asymptotics.compute_LT_asymptotics(t_long,Pe=Pe)

####
# big vis

fig,ax = pyplot.subplots(1,3, figsize=(15,5), sharex=True, sharey=True, constrained_layout=True)


utils.vis_fe_2d(asymptotics.flow, antialiased=True, cmap=my_cm, ax=ax[0], cbar=False)
utils.vis_fe_2d(asymptotics.g1, antialiased=True, cmap=my_cm2, ax=ax[1], cbar=False, cstyle='divergent')
utils.vis_fe_2d(asymptotics.g2, antialiased=True, cmap=my_cm3, ax=ax[2], cbar=False, cstyle='divergent')


#

cellX,cellY = asymptotics.mesh.cellCenters.value
ax[0].tricontour(cellX, cellY, asymptotics.flow.value, 11, colors='k')
ax[1].tricontour(cellX, cellY, asymptotics.g1.value, 11, colors='k')
ax[2].tricontour(cellX, cellY, asymptotics.g2.value, 11, colors='k')

#
ax[0].axis('square')
ax[1].axis('square')
ax[2].axis('square')
fig.show()

###

fig2,ax2 = pyplot.subplots(1,1, figsize=(8,4), constrained_layout=True)
ax2.plot(t_short, Sk_asymp_short, lw=2, ls='--')
ax2.plot(t_long, Sk_asymp_long, lw=2, ls='--')

ax2.text(
0.05,0.95,
r"$S^{G} = %.4f$"%(asymptotics.SG) + "\n" + r"$S_{long} = %.4f$"%asymptotics.SLONG, 
ha='left', va='top', 
transform=ax2.transAxes
)

ax2.set_xscale('log')
ax2.grid()

fig2.show()

pyplot.ion()

#fig.savefig('spatial_solutions.png', dpi=120, bbox_inches='tight')
