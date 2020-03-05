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
        facecolors=fipy_cell_var.value, 
        cmap=mycm, 
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

