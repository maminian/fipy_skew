import numpy as np
import fe_flow

from matplotlib import pyplot
from mpl_toolkits import mplot3d

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
tsteps = 10
dt = 1e-3
Pe = 1e4

mycm = cmocean.cm.thermal

# prescribe boundary - here, a pentagon.
s = 5
bdry = np.array([ [np.cos(th), np.sin(th)] for th in 2*np.pi/s*np.arange(s) ])

fef = fe_flow.fe_flow(bdry, verbosity=1)

# cross-sectional distribution
transv = -1 + 2*np.random.rand(n,2)

# lengthwise distribution
longitud = np.zeros( (n,1) )

# pts
pts = np.hstack([longitud, transv])

pt_hist = np.zeros( (n,3,tsteps) )
pt_hist[:,:,0] = pts


#
#fig = pyplot.figure( figsize=(12,8) )
#ax = fig.add_subplot(111, projection='3d')
fig,ax = pyplot.subplots(1,1, figsize=(8,8))
ax.axis('equal')

# plot the boundary
wrap = np.arange(s+1)%s
ax.plot(fef.boundary[wrap,0], fef.boundary[wrap,1], c='r')

redraw(ax, pts, mycm(np.zeros(n)))


# done

fig.show()
pyplot.ion()

for i in range(1,tsteps):
    
    u = fef.get_flow(pts[:,1:], exterior=-fef.flow_average) + fef.flow_average
        adv = np.vstack( [ Pe*u*dt, np.zeros( (2,n) ) ] ).T
    
    # plotting after diffusion is a little harder to understand
    if i%1==0:
        print('%i of %i'%(i+1,tsteps))
        
        norm_flow = u/fef.flow_lab.value.max()
        is_pos = np.array(norm_flow > 1e-8, dtype=float)
        redraw(ax, pts, mycm(norm_flow))
        pyplot.pause(0.01)
    #
    diff = np.sqrt(2*dt)*np.random.randn(n,3)
    
    pts += adv + diff
    
    # store history
    pt_hist[:,:,i] = pts
    
    

#


