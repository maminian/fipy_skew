# Demo showing how one can feed 
# mesh-valued functions from fipy into one 
# another to successively solve the 
# moment equations
# 
# (ex: solving the cell problem Laplace(theta) = u, Neumann boundaries)
#

import numpy as np
from matplotlib import pyplot
import os

import utils

import colorcet
my_cm = colorcet.cm.kbc
my_cm2 = colorcet.cm.gwv
my_cm3 = colorcet.cm.cwr

##################

#s = 8
#pts = np.array([ [np.cos(th), np.sin(th)] for th in 2*np.pi/s*np.arange(s) ])

if True:
    import demo09
    pts = demo09.coords
    pts = pts - np.mean(pts, axis=0)
    pts = pts/(np.std(pts, axis=0))
else:
    lam = 1.0
    #pts = [[-1/lam,-1],[1/lam,-1],[1/lam,1],[-1/lam,1]]
    th = np.linspace(0,2*np.pi, 200)
    x = 1/lam*np.cos(th[:-1])
    y = np.sin(th[:-1])
    pts = np.vstack([x,y]).T

Pe = 10**4
basename = 'tapir2'
fname = os.path.join('images','%s.png'%basename)

########################

t_short = 10.**np.linspace(-1+np.log10(Pe**-2),1+np.log10(Pe**-1), 41)
t_long = 10.**np.linspace(0,2, 41)

asymptotics = utils.Sk_Asymptotics(pts)
_,_,_,Sk_asymp_short = asymptotics.compute_ST_asymptotics(t_short,Pe=Pe)
_,_,Sk_asymp_long = asymptotics.compute_LT_asymptotics(t_long,Pe=Pe)

####
# big vis

fig = pyplot.figure(constrained_layout=True)
mosaic = '''
ABC
DDD
'''
ax = fig.subplot_mosaic(mosaic)

###

utils.vis_fe_2d(asymptotics.flow, antialiased=True, cmap=my_cm, ax=ax["A"], cbar=False)
utils.vis_fe_2d(asymptotics.g1, antialiased=True, cmap=my_cm2, ax=ax["B"], cbar=False, cstyle='divergent')
utils.vis_fe_2d(asymptotics.g2, antialiased=True, cmap=my_cm3, ax=ax["C"], cbar=False, cstyle='divergent')


#

cellX,cellY = asymptotics.mesh.cellCenters.value
ax["A"].tricontour(cellX, cellY, asymptotics.flow.value, 11, colors='k')
ax["B"].tricontour(cellX, cellY, asymptotics.g1.value, 11, colors='k')
ax["C"].tricontour(cellX, cellY, asymptotics.g2.value, 11, colors='k')

#
ax["A"].axis('equal')
ax["B"].axis('equal')
ax["C"].axis('equal')

##

ax["D"].plot(t_short, Sk_asymp_short, lw=2, ls='--')
ax["D"].plot(t_long, Sk_asymp_long, lw=2, ls='--')

ax["D"].text(
0.02,0.02,
r"$S^{G} = %.4f$"%(asymptotics.SG) + "\n" + r"$S_{long} = %.4f$"%asymptotics.SLONG, 
ha='left', va='bottom', 
transform=ax["D"].transAxes,
fontsize=10,
bbox={'facecolor':'#eee', 'edgecolor':'#bbb', 'boxstyle':'round', 'lw':1}
)

ax["D"].set_xscale('log')
ax["D"].grid()

#
if False:
    import os
    fig.savefig(fname)

fig.show()
pyplot.ion()

