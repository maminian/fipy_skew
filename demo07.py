# Demo of a weirdo domain.

import numpy as np
import fe_flow

from matplotlib import pyplot

import utils

#import cmocean  # colormap

import colorcet

import os

ANIMATE = True
SAVEFRAMES = False
FRAMESDIR = 'frames'
if not os.path.exists(FRAMESDIR):
    os.mkdir(FRAMESDIR)

TEMPLATE_OUTPUT = os.path.join(FRAMESDIR, 'reflections_%s.png')
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


    edgecolors = np.tile([0.2,0.2,0.2,1],(fef.mesh.faceCenters.value.shape[1],1))
    #edgecolors[exterior_faces] = [0,0,0,1]
    # pyplot ordering of faces isn't the same as gmesh.


    ax.collections[0].set_facecolor('k')
    ax.collections[0].set_edgecolors(edgecolors)
    ax.collections[0].set_linewidth(0.05)
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



x0o = np.array([-0.5,-0.1])
x1o = np.array([-6.,-1])

x0,x1 = np.array(x0o), np.array(x1o)

vec = x1-x0
d = np.linalg.norm(vec)

mask,active_cid = utils.locate_cell(x0, fef.mesh, c_mats=fef.c_mats)


# for loop roughly should look like....
# while remaining time positive,
t = 0.

jj=0

active_face=np.nan

txt = None

#import pdb
while t<1:
    # identify current vector and cell.
#    pdb.set_trace()
    # use cell exit algorithm with current position and expected final.
    esses = fef.argintersection(active_cid,x0,x1)

    # identify faceID of exit, if any (just do it if none)
    faces = fef.mesh.cellFaceIDs.data[:,active_cid]
    
    # exclude non-physical times.
    valid_mask = np.logical_and( esses>=0, esses<=1. )
    
    # exclude the face we're sitting on (if any)
    valid_mask = np.logical_and( valid_mask, np.logical_not(faces==active_face) )
    
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

