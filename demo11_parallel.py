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

import multiprocessing

pyplot.style.use('dark_background')


###############

NPROCS = 12 # number of processors to parallelize the dynamics

SAVEFRAMES = True
FRAMESDIR = 'frames_demo11_p_aug11'
if not os.path.exists(FRAMESDIR):
    os.mkdir(FRAMESDIR)

N = 10000    # total particles
Pe = 1e4    # Peclet
dtmax = 1e-3
#times = np.linspace(0,1,100)    # sample times
times = 10.**np.linspace(-8,0,9)
times = np.concatenate([np.array([0]), times, np.arange(1.5,10.1,0.5)])
#times = np.arange(0,1,1e-2)
#times = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


fig_p,ax_p = pyplot.subplots(1,1, figsize=(8,6), constrained_layout=True)


TEMPLATE_OUTPUT = os.path.join(FRAMESDIR, 'frame_%s.png')
##

#############################

import demo09 as myshape
bdry = myshape.coords
bdry = bdry - np.mean(bdry, axis=0)
bdry = bdry / np.std(bdry, axis=0)
bdry = bdry*2

#############################

fef = fe_flow.fe_flow(bdry, cell_size=0.2, verbosity=1)
exterior_faces = np.where( fef.mesh.exteriorFaces.value )[0]


# TODO: don't do a shit job of this. Just sample randomly based on 
# cell number and position randomly in barycentric coordinates.

#
# rejection (re)sampling method to get 
# initial condition within domain.
def sample_interior_uniform(fe_obj, N):
    '''
    Generates an (N,3) array of (x,y,z) coordinates 
    where (y,z) are sampled on the interior of the given finite element mesh.
    '''
    bdry_face_indexes = fe_obj.mesh.exteriorFaces.value
    bdry_vertex_indexes = fe_obj.mesh.faceVertexIDs.data[:,bdry_face_indexes]
    bdry_vertex_indexes = np.unique( bdry_vertex_indexes.flatten() )
    
    bdry_xy = fe_obj.mesh.vertexCoords[:, bdry_vertex_indexes]
    
    zr,yr = bdry_xy.max(axis=1)
    zl,yl = bdry_xy.min(axis=1)

    X0 = np.random.rand(N,3)

    X0[:,0] = np.zeros(N)
    X0[:,1] = zl + (zr-zl)*X0[:,1]
    X0[:,2] = yl + (yr-yl)*X0[:,2]

    mask,active_cells = utils.locate_cell(X0[:,1:], fe_obj.mesh, c_mats=fe_obj.c_mats)
    bads = np.logical_not(mask)
    idx = np.where(bads)[0]
    while not all(mask):
        _temp = np.random.rand(len(idx),3)
        _temp[:,0] = 0
        _temp[:,1] = zl + (zr-zl)*_temp[:,1]
        _temp[:,2] = yl + (yr-yl)*_temp[:,2]

        submask,_ = utils.locate_cell(_temp[:,1:], fe_obj.mesh, c_mats=fe_obj.c_mats)
        X0[idx[submask]] = _temp[submask]
        mask[idx[submask]] = True
    return X0
#

######################
def stepforward(inputs):
    '''
    Marches forward the particles according to the mesh, at the 
    prescribed set of times.
    '''
    XYZ, fe_obj, times = inputs

    dts = np.diff(times)
    _mask, _active_cells = utils.locate_cell(XYZ[:,1:], fe_obj.mesh, c_mats=fe_obj.c_mats)

    _N = np.shape(XYZ)[0]
    
    for _i,dt in enumerate(dts):
        dW = np.sqrt(2*dt)*np.random.normal(0, 1, (_N,3))
        velos = fe_obj.flow.value[_active_cells]
        XYZ[:,0] = XYZ[:,0] + dt*Pe*velos + dW[:,0]
        
        # tentative step forward for (y,z) prior to boundary conditions
        YZ = XYZ[:,1:] + dW[:,1:]
        _mask,_ = utils.locate_cell(YZ, fe_obj.mesh, c_mats=fe_obj.c_mats)
        
        
        # TODO: what is this nightmare code
        # (loop over all particles to apply boundary conditions)
        out_of_bounds_idxs = np.where(np.logical_not(_mask))[0]

        for ii in out_of_bounds_idxs:
            out_of_bounds = True

            x0 = np.array( XYZ[ii,1:] )
            x1 = np.array( YZ[ii] )
            tmp_exclude_faces = []

            # 1. Find face crossing which is valid.
            active_cell = _active_cells[ii]
            while out_of_bounds:
                xorig = np.array(XYZ[ii,1:])
                face_cross_times = fe_obj.argintersection(active_cell, x0, x1)
                
                face_ids = fe_obj.mesh.cellFaceIDs.data[:,active_cell]

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
                    cell_neighbors = fe_obj.mesh.faceCellIDs[:,face_cross_id].data
                    if not (face_cross_id in exterior_faces):
                        neighbor_cell = cell_neighbors[0] if cell_neighbors[0] != active_cell else cell_neighbors[1]
                        x0 = x0_temp
                        tmp_exclude_faces = [face_cross_id] # avoid "reflecting" a second time off the same face (floating point issues)
                        active_cell = neighbor_cell
                        _active_cells[ii] = neighbor_cell
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
            
    #        _mask,_ = utils.locate_cell(x1, fe_obj.mesh, c_mats = fe_obj.c_mats)
    #        if not mask:
    #            print('what the fuck', ii, x1)
    #            import pdb
    #            pdb.set_trace()
        # end for loop of out-of-bounds particles.
        
        # update XYZ
        XYZ[:,1:] = YZ
        
        # update active cells for next velocity
        #_,_active_cells = utils.locate_cell(X[:,1:], fef.mesh, c_mats=fef.c_mats)
    #
    return XYZ

def get_bounds(fe_obj):
    bdry_face_indexes = fe_obj.mesh.exteriorFaces.value
    bdry_vertex_indexes = fe_obj.mesh.faceVertexIDs.data[:,bdry_face_indexes]
    bdry_vertex_indexes = np.unique( bdry_vertex_indexes.flatten() )
    
    bdry_xy = fe_obj.mesh.vertexCoords[:, bdry_vertex_indexes]
    
    zr,yr = bdry_xy.max(axis=1)
    zl,yl = bdry_xy.min(axis=1)
    return zl,zr,yl,yr
    
def plot_state(ax_p, fe_obj, time_val):
    ax_p.cla()
    zl,zr,yl,yr = get_bounds(fe_obj)

#        ax_p.scatter(X[:,1], X[:,2], c='k', s=4)
    ax_p.scatter(X[:,1], X[:,2], c=X[:,0], s=8, edgecolor='#999', linewidth=0.5)
    utils.vis_fe_mesh_2d(fef.flow, ax=ax_p, c='#333', linewidth=0.5)
    
    utils.vis_fe_mesh_boundary(fe_obj, color='#aaf', ax=ax_p, linewidth=1)
    
    ax_p.set_xlim([zl-0.5, zr+0.5])
    ax_p.set_ylim([yl-0.5, yr+0.5])
    ax_p.axis('equal')
    
    ax_p.text( 0.05,0.95,r"$t=%.2e$"%time_val, transform=ax_p.transAxes, ha='left', va='top')
    return



#############



X0 = sample_interior_uniform(fef, N)
# full repeat identifying active cells.
mask,active_cells = utils.locate_cell(X0[:,1:], fef.mesh, c_mats=fef.c_mats)

times_internal, save_times = utils.pad_times2(times, dtmax)

_pool = multiprocessing.Pool(NPROCS)

######### 
# Setting up partitioning for parallel timestepping
cutoffs = np.linspace(0, X0.shape[0], NPROCS+1)
cutoffs = np.array(cutoffs, dtype=int)
subsets = [np.arange(cutoffs[i],cutoffs[i+1]) for i in range(len(cutoffs)-1)]

###########

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
#    plot_state(ax_p, fef, times_internal[0])
    plot_state(ax_p, fef, times[0])
    fig_p.savefig(TEMPLATE_OUTPUT % str(jj).zfill(5))
jj+=1


for i in range(len(times_internal)):
    print("Start of iter %i, time %.3e -- %.3e"%(i, times_internal[i][0], times_internal[i][-1]))

    inputs = [[np.array(X[s,:]), fef, times_internal[i]] for s in subsets]
    Xlist = _pool.map(stepforward, inputs)
    X = np.vstack(Xlist)
    
    # TODO: do we need to do a complete repeat of this???
#    mask,active_cells = utils.locate_cell(X[:,1:], fef.mesh, c_mats=fef.c_mats)
    
    if save_times[i]:
        # Calculate statistics
        # Statistics on X.
        mean_x[jj],var_x[jj],sk_x[jj],med_x[jj] = utils.compute_stats(X[:,0])
        print("\t mean: %.3f; std: %.3f; skew: %.3f"%(mean_x[jj],np.sqrt(var_x[jj]),sk_x[jj]))
        
        if SAVEFRAMES:
            print("\tSaved frame %i"%jj)
            

            plot_state(ax_p, fef, times[i])

            fig_p.savefig(TEMPLATE_OUTPUT % str(jj).zfill(5))
        
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
fig.savefig(os.path.join(FRAMESDIR, 'diagram.png'))
pyplot.ion()

