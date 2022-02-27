# Attempting to plug in duck doordinates 
# into the general simulation.

import numpy as np
import fe_flow

from matplotlib import pyplot

import utils

#import cmocean  # colormap

import colorcet

import os

from scipy import stats

ANIMATE = False
SAVEFRAMES = True
FRAMESDIR = 'frames_demo10'
if not os.path.exists(FRAMESDIR):
    os.mkdir(FRAMESDIR)

N = 1000    # total particles
Pe = 1e4    # Peclet
dtmax = 1e-2
#times = np.linspace(0,1,100)    # sample times
times = 10.**np.linspace(-8,-1,21)
times = np.concatenate([np.array([0]), times])

fig_p,ax_p = pyplot.subplots(1,1, figsize=(8,6), constrained_layout=True)

if ANIMATE:
    # TODO: doesn't work
    fig_p.show()
    fig_p.canvas.draw()
    pyplot.ion()

TEMPLATE_OUTPUT = os.path.join(FRAMESDIR, 'frame_%s.png')
##

#############################

import demo10
bdry = demo10.coords
bdry = bdry - np.mean(bdry, axis=0)
bdry = bdry / np.std(bdry, axis=0)
bdry = bdry*2

#############################

fef = fe_flow.fe_flow(bdry, cell_size=0.5, verbosity=1)
exterior_faces = np.where( fef.mesh.exteriorFaces.value )[0]


# rejection (re)sampling method to get 
# initial condition within domain.
zr,yr = np.max(bdry, axis=0)
zl,yl = np.min(bdry, axis=0)

X0 = np.random.rand(N,3)

X0[:,0] = np.zeros(N)
X0[:,1] = zl + (zr-zl)*X0[:,1]
X0[:,2] = yl + (yr-yl)*X0[:,2]

mask,active_cells = utils.locate_cell(X0[:,1:], fef.mesh, c_mats=fef.c_mats)
bads = np.logical_not(mask)
idx = np.where(bads)[0]
while not all(mask):
    _temp = np.random.rand(len(idx),3)
    _temp[:,0] = 0
    _temp[:,1] = zl + (zr-zl)*_temp[:,1]
    _temp[:,2] = yl + (yr-yl)*_temp[:,2]

    submask,_ = utils.locate_cell(_temp[:,1:], fef.mesh, c_mats=fef.c_mats)
    X0[idx[submask]] = _temp[submask]
    mask[idx[submask]] = True

# full repeat identifying active cells.
# TODO: do this within the rejection sampling loop
mask,active_cells = utils.locate_cell(X0[:,1:], fef.mesh, c_mats=fef.c_mats)

# main loop
#t = 0.
jj=0
i=0

X = np.array(X0) # why have copy of initial cond?

mean_x = np.zeros(np.shape(times))
var_x = np.zeros(np.shape(times))
sk_x = np.zeros(np.shape(times))
med_x = np.zeros(np.shape(times))

mean_x[jj],var_x[jj],sk_x[jj],med_x[jj] = utils.compute_stats(X[:,0])

if SAVEFRAMES:
    ax_p.cla()
    
#    ax_p.scatter(X[:,1], X[:,2], c='k', s=4)
    ax_p.scatter(X[:,1], X[:,2], c=X[:,0], s=4)
    utils.vis_fe_mesh_2d(fef.flow, ax=ax_p, c='#999', linewidth=0.5)
    
    ax_p.set_xlim([bdry[:,0].min()-0.5, bdry[:,0].max()+0.5])
    ax_p.set_xlim([bdry[:,1].min()-0.5, bdry[:,1].max()+0.5])
    ax_p.axis('square')
    ax_p.text( 0.05,0.95,r"$t=%.4e$"%times[i], transform=ax_p.transAxes, ha='left', va='top')
    fig_p.savefig(TEMPLATE_OUTPUT%jj)


jj+=1

times_internal,save_times = utils.pad_times(times, dtmax)

for i in range(1,len(times_internal)):
    print("Start of iter %i, time %.3e"%(i,times_internal[i]))


    # get params
    dt = times_internal[i] - times_internal[i-1]
    dW = np.sqrt(2*dt)*np.random.randn(N,3)

    velos = fef.flow.value[active_cells]

    # Evolve equations.
    X[:,0] += dt*Pe*velos + dW[:,0]
    YZ = X[:,1:] + dW[:,1:]
    
    # Did they end up somewhere within the mesh?
    mask,_ = utils.locate_cell(YZ, fef.mesh, c_mats=fef.c_mats)

    mask_to_vis = np.array(mask)

    # What cell did they start out in?
    _,active_cells = utils.locate_cell(X[:,1:], fef.mesh, c_mats=fef.c_mats)

    out_of_bounds_idxs = np.where(np.logical_not(mask))[0]

    for ii in out_of_bounds_idxs:
        out_of_bounds = True

        x0 = np.array( X[ii,1:] )
        x1 = np.array( YZ[ii] )
        tmp_exclude_faces = []

        # 1. Find face crossing which is valid.
        active_cell = active_cells[ii]
        while out_of_bounds:
            xorig = np.array(X[ii,1:])
            face_cross_times = fef.argintersection(active_cell, x0, x1)
            
            face_ids = fef.mesh.cellFaceIDs.data[:,active_cell]

            # get valid crossings (boolean for each face)
            valid_crossings = [0<fct and fct<=1 and fid not in tmp_exclude_faces for fct,fid in zip(face_cross_times,face_ids)]
            tmp_exclude_faces = []
            
            # There may be two positive crossings; want the smaller of them.
            if sum(valid_crossings)>1:
                valid_crossings = np.logical_and(valid_crossings, face_cross_times < max(face_cross_times))
            
            if any(valid_crossings):
                
                face_id_ptr = np.where(valid_crossings)[0][0]   # array position for face
                face_cross_id = face_ids[ face_id_ptr ]         # global identifier for face
                
                x0_temp = x0 + (x1-x0)*face_cross_times[face_id_ptr] # cartesian point
                
                # 2. Check if we passed in to another cell first.
                # If face crossing was into another cell, assign the crossing point 
                # at the face to x0; reset, and exclude the face as a valid face;
                # reassign the active cell; then go to 1.

                # what cells does this face lie on?
                cell_neighbors = fef.mesh.faceCellIDs[:,face_cross_id].data
                if not (face_cross_id in exterior_faces):
                    neighbor_cell = cell_neighbors[0] if cell_neighbors[0] != active_cell else cell_neighbors[1]
                    x0 = x0_temp
                    tmp_exclude_faces = [face_cross_id]
                    active_cell = neighbor_cell
                    active_cells[ii] = neighbor_cell
                else:
                # 3. If face crossing which is valid is an exterior face,
                # then calculate the reflection and reassign x1. Assign the crossing point 
                # to x0. Reset and exclude disallowed reflection faces to exclude the crossing face.

                    veci = x1-x0_temp
                    normal = fef.mesh.faceNormals.data[:,face_cross_id]
                    newvec = veci - 2*normal*np.dot(normal, veci)
                    # update x0 to be boundary point.
                    x0 = x0_temp
                    # update x1
                    x1 = x0_temp + newvec
                    tmp_exclude_faces = [face_cross_id]
                #
            else:
                out_of_bounds = False
            # end if
        # end while for single particle out-of-bounds.

        # 4. Done!
        YZ[ii] = x1
        mask,_ = utils.locate_cell(x1, fef.mesh, c_mats = fef.c_mats)
        if not mask:
            print('what the fuck', ii, x1)
            import pdb
            pdb.set_trace()
    # end for loop of out-of-bounds particles.
    X[:,1:] = YZ
    
    # TODO: do we need to do a complete repeat of this???
#    mask,active_cells = utils.locate_cell(X[:,1:], fef.mesh, c_mats=fef.c_mats)
    
    
    if SAVEFRAMES and save_times[i]:
        print("\tSaved a frame, ")
        print("\t mean: %.3e; std: %.3e; skew: %.3e"%(mean_x[jj],np.sqrt(var_x[jj]),sk_x[jj]))
        ax_p.cla()

#        ax_p.scatter(X[:,1], X[:,2], c='k', s=4)
        ax_p.scatter(X[:,1], X[:,2], c=mask_to_vis, s=4)
        utils.vis_fe_mesh_2d(fef.flow, ax=ax_p, c='#999', linewidth=0.5)
        
        ax_p.set_xlim([bdry[:,0].min()-0.5, bdry[:,0].max()+0.5])
        ax_p.set_ylim([bdry[:,1].min()-0.5, bdry[:,1].max()+0.5])
        ax_p.axis('equal')
        
        ax_p.text( 0.05,0.95,r"$t=%.4e$"%times_internal[i], transform=ax_p.transAxes, ha='left', va='top')
        fig_p.savefig(TEMPLATE_OUTPUT%jj)
    if ANIMATE and save_times[i]:
        # TODO: doesn't work
        fig_p.canvas.draw()
    
    if save_times[i]:
        # Calculate statistics
        # Statistics on X.
        mean_x[jj],var_x[jj],sk_x[jj],med_x[jj] = utils.compute_stats(X[:,0])
        jj += 1
    
# end for loop

# vis...
from matplotlib import pyplot
fig,ax = pyplot.subplots(1,2, figsize=(12,6))

ax[0].plot(times,sk_x, marker='.', markersize=2)
ax[0].set_xscale('log')
utils.vis_fe_2d(fef.flow, cbar=False, ax=ax[1])
utils.vis_fe_mesh_2d(fef.flow, c='#555', linewidth=0.1, ax=ax[1])
ax[1].axis('equal')

ax[0].set_xlabel('time')
ax[0].set_ylabel('Sk')

fig.show()
pyplot.ion()

