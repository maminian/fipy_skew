# wow

def anticlockwise_order(points):
    '''
    returns the integer ordering of a collection of
    points so that they are arranged in an anti-clockwise order.

    degenerate cases (e.g. three points on top of each other)
    are not handled. This code may work for "nice" cases with n>3,
    but no guarantees can be made.

    Inputs:
        points: float array shape (n,2)

    Outputs:
        idx: integer array shape (n,) so that points[idx] will arrange
            the same points in anti-clockwise order, as determined by
            the angle relative to the center of the points (using np.arctan2)
    '''
    import numpy as np

    center = np.mean(points, axis=0)
    pc = np.array(points) - center

    angles = np.arctan2(pc[:,1], pc[:,0])

    idx = np.argsort(angles)

    return idx
#

def vis_fe_2d(fipy_cell_var, cbar=True, **kwargs):
    '''
    A slightly more flexible visualization tool for fipy.CellVariable()
    solutions using matplotlib's pyplot.tripcolor() function.


    Inputs:
        fipy_cell_var: A fipy.CellVariable(), presumably the solution to
            a PDE on a triangular mesh.
    Outputs:
        Dependent on the keyword arguments (see below).
        Typically None, if a matplotlib axes handle is provided,
        otherwise a matplotlib figure and axis pair.

    Optional inputs:
        args:
            cbar : Boolean. If True, the default colobar is generated
                using pyplot's colorbar methods. (default: True)
        kwargs:
            ax : if provided, the plotting will be done in the provided
                matplotlib axis object.
            The remainder of kwargs are passed on to tripcolor's named
            and keyword arguments. Note the internal call to
            tripcolor() will look like:

            tripcolor(triangulation, cmap=pyplot.cm.viridis, facecolors=fipy_cell_var.value, **kwargs)

            The kwargs 'cmap' will be read and popped from
            the input to vis_fe_2d. The rest get passed directly to the
            kwargs of tripcolor.
    '''

    import numpy as np
    from matplotlib import tri,pyplot

    # pop out/set default values
    if 'ax' in kwargs.keys():
        ax = kwargs.pop('ax')
        fig = ax.get_figure()
        flag_ax = False
    else:
        fig,ax = pyplot.subplots(1,1)
        flag_ax = True
    #

    if 'cmap' in kwargs.keys():
        mycm = kwargs.pop('cmap')
    else:
        mycm = pyplot.cm.viridis
    #

    # create matplotlib triangulation object.
    cell_to_face = fipy_cell_var.mesh.cellFaceIDs       # 3 by ncells
    face_to_vertex = fipy_cell_var.mesh.faceVertexIDs   # 2 by nedges
    vertex_coords = fipy_cell_var.mesh.vertexCoords     # 2 by nvertex

    # input looks like x coords, y coords, integer triples for triangle vertex pointers.
    # however, points need to be in anticlockwise order.
    xx,yy = vertex_coords

    # don't understand exactly what face_to_vertex is doing;
    # this implementation will work but it's not optimal.
#    idx_pre = face_to_vertex[:,cell_to_face.T]
    idx_pre = [ np.unique(face_to_vertex[:,ci]) for ci in cell_to_face.T ]

    # ensure ordering in anti-clockwise order for matplotlib.tri.Triangulation.
    idx = np.array(idx_pre, dtype=int)
    for j,triple in enumerate(idx_pre):
        coords = np.vstack( [ xx[triple], yy[triple] ]).T
        o = anticlockwise_order(coords)
        idx[j] = triple[o]
    #

    triang = tri.Triangulation(xx, yy, idx)

    triobj = ax.tripcolor(
        triang,
        facecolors = fipy_cell_var.value,
        cmap = mycm,
        vmin = np.nanmin( fipy_cell_var.value ),     # doesn't handle NaN well
        vmax = np.nanmax( fipy_cell_var.value ),
        **kwargs
    )

    if cbar:
        fig.colorbar(triobj)
    #

    # the end
    if flag_ax:
        return fig,ax
    else:
        return
    #
#

def locate_cell(points, gmesh, **kwargs):
    '''
    Given a gmesh object (collection of vertices, faces, cells),
    generate a collection of pointers indicating which cell each
    point lies in. Degenerate cases are not given any special consideration.

    Input:
        points: numpy array of shape (2,) or (n,2); either a singleton
            or a collection of points in 2D organized by row.
        gmesh: a gmesh object which is a triangularization of a domain.
            You may also input a CellVariable; the mesh for that variable
            contains the same information and is used internally.

    Outputs:
        mask: boolean array shape (n,) indicating whether the points
            lie in *any* of the triangles. You should use this as a
            preprocessing step to handle exceptions, etc.
        idx: Either an integer, or a numpy integer array shape (n,)
            indicating which cell one or more points lie within.
            Note that pointers for which mask==False are nonsense
            (the mask indicates they are outside the array, so not handled.)

    Optional inputs:
        c_mats: if specified, presumed to be precomputed coefficient
            arrays used in evaluating new points. Big cost savings on
            repeated calls!
    '''
    import numpy as np
    import fipy
    import utils

    # Make sure the mesh is soemthing we know we can work with.
    if isinstance(gmesh, fipy.variables.cellVariable.CellVariable):
        mg = gmesh.mesh
    elif isinstance(gmesh, fipy.meshes.gmshMesh.Gmsh2D):
        mg = gmesh
    else:
        raise Exception('Cannot handle input mesh of type %s'%str(type(gmesh)))
    #

    pshape = np.shape(points)
    if pshape==(2,):
        singleton = True
        n=1
        pts = [points]
    elif len(pshape)==2 and pshape[1]==2:
        singleton = False
        n = pshape[0]
        pts = points
    else:
        raise Exception('Cannot handle input points of shape ', np.shape(points))
    #

    ######
    #
    # If user precomputed coefficient arrays, set a flag to true to branch the main loop.
    #
    if 'c_mats' in kwargs.keys():
        c_precomputed = True
        c_mats = kwargs.get('c_mats')
    else:
        c_precomputed = False
        # normally c_mats is shape (ncell,3,3). It doesn't really matter what's set here.
        c_mats = np.zeros( (0,3,3) )
    #

    ncell = mg.cellFaceIDs.shape[1]

    #########
    #
    #
    # TODO - class which precomputes these outside this function.
    # TODO - get rid of this "np.unique" and figure out why the faces have
    # inconsistent ordering around each triangle which makes this a pain.
    if not c_precomputed:
        vci = [np.unique(mg.faceVertexIDs[:,ci]) for ci in mg.cellFaceIDs.T]
        X = mg.vertexCoords[0,vci]
        Y = mg.vertexCoords[1,vci]

        # TODO - repackage everything in a nicer way that I'm not constantly
        # doing transposes, elementwise operations, etc.
        #Cmats = np.array([utils._get_cell_coeffs(np.vstack([xi,yi]).T) for xi,yi in zip(X,Y)])
    #


    # existing code is set up to loop over *cells*, not points.
    # so, the algorithm will look like
    # 1. Get collection of undecided particles (monotone decreasing over loops)
    # 2. Evaluate at current cell
    # 3. Update associated idx entries
    # 4. If all idx have been evaluated, exit; else go to 1.
    # 5. If all cells have been evaluated, mark any remaining points as np.nan
    #    (they do not lie within the mesh).
    active_pt_mask = np.ones(n, dtype=bool)
    idx = np.zeros(n, dtype=int)

    for i in range(ncell):
        if c_precomputed:
            Cmat = c_mats[i]
        else:
            Cmat = utils._get_cell_coeffs(np.vstack([ X[i],Y[i] ]).T)
        #
        
        # get the active set of points
        pt_subset = points[active_pt_mask]
        pt_subset_ptr = np.where(active_pt_mask)[0]

        # get which points lie in the current cell
        flags = utils._in_Carr(pt_subset, Cmat)
        # update index
        idx[pt_subset_ptr[flags]] = i
        # update active point mask with points accounted for
        active_pt_mask[pt_subset_ptr[flags]] = False   # god no

        if any(active_pt_mask):
            # if any points have yet to be assigned, continue
            continue
        else:
            # all points have been accounted for; we can break.
            break
    #

    # check for any points which haven't been assigned a cell; mark as NaN
#    idx[active_pt_mask] = np.nan
    interior_mask = np.logical_not(active_pt_mask)

    # done - return a scalar if a singleton was fed; else the idx array.
    if singleton:
        return interior_mask[0],idx[0]
    else:
        return interior_mask,idx
#

# unneeded for algorithms below
#
#def l(xv,yv,coeffs):
#    return coeffs[0] + coeffs[1]*xv + coeffs[2]*yv

def _get_cell_coeffs(point_coll):
    '''
    Given a triangle parameterized by the coordinates of
    its vertices, construct the coefficient matrix needed
    to evaluate whether a point is in the interior.

    It's expected this will be used in conjunction with in_interior().
    This matrix does not have much use otherwise.

    Inputs:
        point_coll: numpy array shape (3,2)
    Outputs:
        coeff_arr: numpy array shape (3,3)
    '''
    import numpy as np

    coeff_arr = np.zeros((3,3))
    for i,o in enumerate( [[0,1,2],[1,2,0],[0,2,1]] ):
        pc = point_coll[o]
        det0 = np.linalg.det(pc[[0,1]])
        det1 = np.linalg.det(pc[[0,2]])
        det2 = np.linalg.det(pc[[1,2]])
        detA = det0 - det1 + det2
        c0 = det0/detA
        c1 = -(pc[1,1] - pc[0,1])/detA
        c2 = (pc[1,0] - pc[0,0])/detA
        coeff_arr[i] = [c0,c1,c2]
    #
    return coeff_arr
#

def _in_Carr(point, Carr):
    '''
    Given one or more points, evaluate whether each point lies in the
    interior of a given triangle parameterized by its coefficient array,
    constructed by get_cell_coeffs().

    Inputs:
        point: numpy array shape (2,) or (n,2)
        Carr: numpy array shape (3,3)
    Outputs:
        flags: boolean numpy array of shape (n,) indicating
            whether each point lies in the given triangle.
            Note in the singleton case this is an *array* of shape (1,),
            not a scalar boolean.
    '''
    import numpy as np
    if len(np.shape(point))==1:
        point = [point] # make 2d array if given a singleton
    #

    Pall = np.array( [ [1.,p[0],p[1]] for p in point] )
    flags = np.all( np.dot(Carr, Pall.T) >= 0, axis=0)
    return flags
#
