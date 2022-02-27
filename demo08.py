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
#t = 0.
jj=0

X = np.array(X0) # why have copy of initial cond?

for i,ti in enumerate(times[:-1]):

    # get params
    dt = times[i+1] - times[i]
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
        idx = np.where(bads)[0]
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
#            valid_mask = np.logical_and( valid_mask, np.logical_not(faces==active_face) )

            # while the particle still has "energy" and will cross some face,
            x1 = np.array(YZ[ii])
            x1i = np.array(X[ii,1:])
            while any(valid_mask):
                # find index of face with first valid intersection
                j = (np.where(valid_mask)[0])[ np.argmin(esses[valid_mask]) ]
                di = esses[j]*np.linalg.norm(x1-x1i)
                s = di/d
                
                x1i_temp = x1i + esses[j]*(x1 - x1i)
                
                di = np.linalg.norm(x1i_temp-x1i)
                
                active_face = fef.mesh.cellFaceIDs.data[:,acid][j]
                if active_face in exterior_faces:
                    # get the normal reflection.
                    veci = X[ii,1:] - x1i
                    normal = fef.mesh.faceNormals.data[:,active_face]
                    newvec = veci - 2*normal*np.dot(normal, veci)
                    # update x1
                    x1 = x1i + newvec
                else:
                    x1 = x1i
                #
                
                # Recompute valid_mask based on the
                mask,temp_cid = utils.locate_cell(x1, fef.mesh, c_mats=fef.c_mats)
                esses = fef.argintersection(temp_cid,x1i,x1)
                
                # get all neighboring faces
                faces = fef.mesh.cellFaceIDs.data[:,temp_cid]
                # 0<=s<=1 corresponds to the physically relevant trajectory.
                valid_mask = np.logical_and( esses>=0, esses<=1. )
            # end while for 
            YZ[ii] = x1
        #
        # recompute particles and verify 
        # TODO: rewrite code to only re-locate on the particles necessary.
        mask,acid = utils.locate_cell(YZ, fef.mesh, c_mats=fef.c_mats)
        print(sum(mask))
    # end while loop for particles that exicted.

# end for loop


