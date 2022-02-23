# Demo in which I attempt to put everything together 
# in a bona fide advection diffusion.

import numpy as np
import fe_flow

from matplotlib import pyplot

import utils

#import cmocean  # colormap

import colorcet

import os

ANIMATE = True
SAVEFRAMES = False
FRAMESDIR = 'frames_demo08'
if not os.path.exists(FRAMESDIR):
    os.mkdir(FRAMESDIR)

N = 1000    # total particles
Pe = 1e3    # Peclet
times = np.linspace(0,1,100)    # sample times


TEMPLATE_OUTPUT = os.path.join(FRAMESDIR, 'frame_%s.png')
##

#############################

# prescribe boundary
ths = np.pi + np.linspace(0,np.pi/2,20)
bdry = [ [np.cos(th), np.sin(th)] for th in ths]
bdry = bdry + [[0.,-0.2],[-0.2,-0.2],[-0.2,0]]
bdry = np.array(bdry)

bdry = np.array([
[-2,1],
[-2, 0],
[-1.,-0.5],
[-1., -0.8],
[-1.0, -1],
[1.0, -1],
[1., -0.8],
[1.,-0.5],
[2, 0],
[2,1]
])


#############################

fef = fe_flow.fe_flow(bdry, cell_size=0.1, verbosity=1)
exterior_faces = np.where( fef.mesh.exteriorFaces.value )[0]

if ANIMATE:
    fig,ax = utils.vis_fe_2d(fef.flow, cbar=False, cmap=colorcet.cm.bmw)
    fig.set_figwidth(8)
    fig.set_figheight(8)

    pyplot.pause(0.1)


#    edgecolors = np.tile([0.2,0.2,0.2,1],(fef.mesh.faceCenters.value.shape[1],1))
    #edgecolors[exterior_faces] = [0,0,0,1]
    # pyplot ordering of faces isn't the same as gmesh.


    ax.collections[0].set_facecolor('k')
#    ax.collections[0].set_edgecolors(edgecolors)
#    ax.collections[0].set_linewidth(0.05)
    ax.axis('equal')
    for v in ax.spines.values():
        v.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_facecolor('k')
    fig.set_facecolor('k')

    fig.tight_layout()

    fig.show()
    pyplot.ion()



#x0o = np.array([-0.5,-0.1])
#x1o = np.array([-6.,-1])

#x0,x1 = np.array(x0o), np.array(x1o)

# rejection (re)sampling method to get 
# initial condition within domain.
zr,yr = np.max(bdry, axis=0)
zl,yl = np.min(bdry, axis=0)

X0 = np.random.rand(N,3)
X0[:,0] = 0
X0[:,1] = zl + (zr-zl)*X0[:,1]
X0[:,2] = yl + (yr-yl)*X0[:,2]

mask,active_cid = utils.locate_cell(X0[:,1:], fef.mesh, c_mats=fef.c_mats)
bads = np.logical_not(mask)
idx = np.where(bads)[0]
while not all(mask):
    _temp = np.random.rand(len(idx),3)
    _temp[:,1] = zl + (zr-zl)*_temp[:,1]
    _temp[:,2] = yl + (yr-yl)*_temp[:,2]

    submask,_ = utils.locate_cell(_temp[:,1:], fef.mesh, c_mats=fef.c_mats)
    X0[idx[submask]] = _temp[submask]
    mask[idx[submask]] = True

# full repeat identifying active cells.
# TODO: do this within the rejection sampling loop
mask,active_cid = utils.locate_cell(X0[:,1:], fef.mesh, c_mats=fef.c_mats)

# main loop
t = 0.
jj=0

X = np.array(X0) # why have copy of initial cond?

#import pdb
for i,ti in enumerate(times[:-1]):

    # get params
    dt = t[i+1] - t[i]
    dW = np.sqrt(2*dt)*np.random.randn(N,3)

#    mask,active_cid = utils.locate_cell(X0[:,1:], fef.mesh, c_mats=fef.c_mats)
    velos = fef.flow.value[active_cid]

    # Evolve equations.
    X[:,0] += dt*Pe*velos + dW[:,0]
    #X[:,1] += dW[:,1]
    #X[:,2] += dW[:,2]
    YZ = X[:,1:] + dW[:,1:]
    
    mask,active_cid = utils.locate_cell(YZ, fef.mesh, c_mats=fef.c_mats)
    # determine if any particle have exited the domain
    while not all(mask):
        bads = np.logical_not(mask)
        idx = np.where(bad)[0]
        # For each particle we know that exited the domain,
        for ii in idx:
            acid = active_cid[ii]   #active cell related to particle's prior position.
            d = np.linalg.norm(YZ[ii] - X[ii,1:])
            
            # get relative time to crossing of face
            esses = fef.argintersection(acid,X[ii,1:],YZ[ii])
            # get all neighboring faces
            faces = fef.mesh.cellFaceIDs.data[:,acid]
            # 0<=s<=1 corresponds to the physically relevant trajectory.
            valid_mask = np.logical_and( esses>=0, esses<=1. )
            valid_mask = np.logical_and( valid_mask, np.logical_not(faces==active_face) )

            # while the particle still has "energy" and will cross some face,
            while any(valid_mask):
                # find index of face with first valid intersection
                j = (np.where(valid_mask)[0])[ np.argmin(esses[valid_mask]) ]
                di = esses[j]*np.linalg.norm(YZ[ii]-X[ii,1:])
                s = di/d
                
                x1i = X[ii,1:] + esses[j]*(YZ[ii] - X[ii,1:])
                
                di = np.linalg.norm(x1i-X[ii,1:])
                
                # here is that face index! wow.
                # TODO - CONTINUE OFF HERE
                active_face = fef.mesh.cellFaceIDs.data[:,acid][j]
                if active_face in exterior_faces:
                    # get the normal reflection.
                    veci = X[ii,1:] - x1i
                    normal = fef.mesh.faceNormals.data[:,active_face]
                    newvec = veci - 2*normal*np.dot(normal, veci)
                    # update x1
                    x1 = x1i + newvec
                    # throw in an arrow, why not.
            #        ax.quiver(x1i[0],x1i[1], -normal[0], -normal[1], color='k')
                else:
                    face_cells = fef.mesh.faceCellIDs.data[:,active_face]
                    active_cid = np.setdiff1d(face_cells, active_cid)[0]
                
############################################

    if sum(valid_mask)==0:
        # TODO: move forward and finish.
        di = np.linalg.norm(x1-x0)
        x0 = np.array(x1)
        t=1.
        break
    else:
        # local index of face with first valid intersection
        j = (np.where(valid_mask)[0])[ np.argmin(esses[valid_mask]) ]

        
        # update time
        di = esses[j]*np.linalg.norm(x1-x0)
        s = di/d
        t += s

#            next_cell = np.setdiff1d(face_cells, active_cid)[0]
        x1i = x0 + esses[j]*(x1 - x0)
        
        di = np.linalg.norm(x1i-x0)
    #

    # if faceID is not a boundary, update remaining time and target point
    # and go to beginning.
    
    # associated faceID
    active_face = fef.mesh.cellFaceIDs.data[:,active_cid][j]
    if active_face in exterior_faces:
        # get the normal reflection.
        veci = x1 - x1i
        normal = fef.mesh.faceNormals.data[:,active_face]
        newvec = veci - 2*normal*np.dot(normal, veci)
        # update x1
        x1 = x1i + newvec
        # throw in an arrow, why not.
#        ax.quiver(x1i[0],x1i[1], -normal[0], -normal[1], color='k')
    else:
        face_cells = fef.mesh.faceCellIDs.data[:,active_face]
        active_cid = np.setdiff1d(face_cells, active_cid)[0]
    #
    
#    x1 = np.array(x1i)
    

    print(x0,x1i,x1,t)
    
    if ANIMATE:
        #pathcolor = colorcet.cm.colorwheel(((jj*8)%256)/256.)
        pathcolor = pyplot.cm.twilight(((jj*8)%256)/256.)
        
        ax.plot([x0[0], x1i[0]], [x0[1],x1i[1]], c=pathcolor, marker='.', markersize=8, linewidth=2)
    #    ax.text(x1i[0],x1i[1], r't=%.3f'%t, fontsize=11, ha='left', va='top')
        

        if hasattr(txt,'remove'):
            txt.remove()
        txt = ax.text(x1i[0],x1i[1], r'$t=%.3f$'%t, c='w', fontsize=12)
        
        if SAVEFRAMES:
            fig.savefig(TEMPLATE_OUTPUT%str(jj).zfill(4), dpi=120, bbox_inches='tight')
        pyplot.pause(0.001)
    
    
    # move forward to face.
    x0 = np.array(x1i)

    jj += 1
#

