# Demo of Brownian particles in a pentagon domain.
# Uses the cell finding/searching tools. 
# Currently doing absorbing boundary conditions 
# (particles become inactive after exiting the domain)
#
# Note: most of the print statements have to do with 
# diagnostics relating to optimization of finding 
# cell containing particle.
# (this optimization is a TODO as of Feb 21 2022.)
#


import numpy as np
import fe_flow

from matplotlib import pyplot
from mpl_toolkits import mplot3d

import utils

import cmocean  # colormap

##

def redraw(myax, pts, color, d=2):
    if len(myax.collections)>0:
        myax.collections[0].remove()
    if d==2:
        myax.scatter(pts[:,1], pts[:,2], c=color, s=10, edgecolor=None)
    else:
        myax.scatter(pts[:,0], pts[:,1], pts[:,2], c=color, s=10, edgecolor=None)
    #
    return
#

# number of particles?
n = 10000
tsteps = 100
dt = 1e-3
Pe = 1e4

vis = True

mycm = cmocean.cm.thermal

# prescribe boundary - here, a pentagon.
s = 5
bdry = np.array([ [np.cos(th), np.sin(th)] for th in 2*np.pi/s*np.arange(s) ])

fef = fe_flow.fe_flow(bdry, verbosity=1)

# cross-sectional distribution
transv = -1 + 2*np.random.rand(n,2)
#transv = np.zeros( (n,2) )

# lengthwise distribution
longitud = np.zeros( (n,1) )

# pts
pts = np.hstack([longitud, transv])
#pt_loc = np.zeros(pts, dtype=np.int64)  #pointers to cell contained.

# get initial list of pointers for cells for each particle.
mask,idx = utils.locate_cell(pts[:,1:], fef.mesh, c_mats = fef.c_mats)
pt_loc = idx


pt_hist = np.zeros( (n,3,tsteps) )
pt_hist[:,:,0] = pts


#
if vis:
    #fig = pyplot.figure( figsize=(12,8) )
    #ax = fig.add_subplot(111, projection='3d')
    fig,ax = pyplot.subplots(1,1, figsize=(8,8))
    ax.axis('equal')

    # plot the boundary
    wrap = np.arange(s+1)%s
    ax.plot(fef.boundary[wrap,0], fef.boundary[wrap,1], c='r')

    redraw(ax, pts, mycm(np.zeros(n)))

    fig.show()
    pyplot.ion()
# done


active_mask = np.ones(n, dtype=bool)

for i in range(1,tsteps):

    mask,idx = fef.locate_cell2(pts[active_mask,1:], pt_loc)
    # turn off points exiting the boundary that are still being tracked.
#    active_mask[active_mask][np.logical_not(mask)] = False
    # ugh can't think through this
    acount = 0
    for ia,a in enumerate(active_mask):
        if not a:
            continue
        else:
            active_mask[ia] = mask[acount]
            acount += 1
    #

    pt_loc = idx[mask]

    u = fef.get_flow(pts[active_mask,1:], exterior=-fef.flow_average, pointers=pt_loc)
    u += fef.flow_average

    # every so many iterations, update the suspected pointers manually here.
    # worry about what to do for exterior points later.




    adv = np.vstack( [ Pe*u*dt, np.zeros( (2,len(u)) ) ] ).T

    # plotting after diffusion is a little harder to understand
    if i%1==0:
        print('%i of %i'%(i+1,tsteps))

        if vis:
            norm_flow = u/fef.flow_lab.value.max()
            is_pos = np.array(norm_flow > 1e-8, dtype=float)
            redraw(ax, pts[active_mask], mycm(norm_flow))
            pyplot.pause(0.01)
    #
    diff = np.sqrt(2*dt)*np.random.randn(len(u),3)

    pts[active_mask] += adv + diff

    # store history
    pt_hist[:,:,i] = pts

#
