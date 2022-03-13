# wow
import numpy as np
from scipy import stats


def pad_times(times, dtmax):
    '''
    Given 1d array times and dtmax, constructs two new 
    arrays which treat "times" as checkpoints with internal 
    timesteps which respect the maximum timestep.
    
    Inputs:
        times : 1d array
        dtmax : positive scalar
    Outputs:
        times_internal : 1d array
        save_time : 1d boolean (if True, we'd like to record the solution at this point)
    '''
    t=times[0]
    times_internal = [t]
    save_time = [True]
    for i in range(1,len(times)):
        while (times[i]-t)>dtmax:
            t += dtmax
            times_internal.append(t)
            save_time.append(False)
        times_internal.append(times[i])
        save_time.append(True)
    return np.array(times_internal), np.array(save_time)
#

def smooth_boundary(arr_1d, nsteps=10, gamma=0.0):
    '''
    does a few steps of the smoothing procedure x[i] = (1-gamma)x[i] + gamma(x[i-1]+x[i+1])/2
    '''
    x = np.array(arr_1d)
    for k in range(nsteps):
        y = np.array(x)
        for i in range(len(arr_1d)):
            x[i] = (1-gamma)*y[i] + gamma*(y[(i+1)%len(arr_1d)] + y[(i-1)%len(arr_1d)])/2
    return x
#

def compute_stats(arr_1d):
    '''
    Calculates summary statistics of a one-dimensional array.
    
    Inputs:
        arr_1d: expected a one-dimensional object; list; numpy array, etc.
    Outputs:
        mean,var,skew,median: scalar statistics computed by scipy.stats.describe and np.median
    '''
    st = stats.describe(arr_1d)
    median = np.median(arr_1d)
    
    return st.mean,st.variance,st.skewness,median
#

def neighbor_order(points,n_neighbors=12):
    '''
    gives an order for which the points are ordered based on distance
    to neighbors.
    
    uses sklearn.
    
    Super unstable.
    Actually just junk, really.
    '''
    from sklearn import neighbors
    
    points = points - np.mean(points, axis=0)
    
    nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn.fit(points)
    avail = list(range(len(points)))
    nbrs = nn.kneighbors()[1]
    nbrs = {i:nbrs[i] for i in range(len(nbrs))}
    avail.remove(0)
    o = [0]
    kk=0
    while len(avail)>1:
        nb = nbrs[kk]
        for ii in nb:
            if ii in avail:
                avail.remove(ii)
                o.append(ii)
                kk = ii
    #
    return o
#

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
                
            cmap : if provided, the provided colormap object will be used. 
                Care should be taken for visualization approach (i.e., 
                linear scaling vs divergent colormaps); see 'cstyle' kwarg below.
                Default: pyplot.cm.viridis
                
            cstyle : if provided, the vmin and vmax values for tripcolor()
                will be defined differently (default: 'linear')
                    'linear' : vmin=fipy_cell_var.value.min(); similar with vmax.
                        This is the default behavior for most plotting commands.
                    'divergent' : vmin and vmax are defined such that the center 
                        value of the colormap corresponds to the value 0 (zero). 
                        This is useful when the context of where "zero" is matters.
            
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
    
    if 'cstyle' in kwargs.keys():
        cstyle = kwargs.pop('cstyle')
    else:
        cstyle = 'linear'
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

    fvals = fipy_cell_var.value
    if cstyle.lower()=='divergent':
        vbnd = max( abs(np.nanmin( fvals )), abs(np.nanmax( fvals )))
        # handle the case of vbnd==0; bad stuff happens later.
        if vbnd==0.: 
            vbnd = 0.1
        vminv = -vbnd
        vmaxv = vbnd
        
    elif cstyle.lower()=='linear':
#        vminv = np.nanmin( fvals )
#        vmaxv = np.nanmax( fvals )
        # try something more robust to possible outliers.
        vminv,vmaxv = np.nanquantile( fvals, [0.025, 0.975])
    else:
        # print warning; default to lienar.
        print('Unrecognized cstyle %s; defaulting to linear colormap assumption.'%cstyle)
#        vminv = np.nanmin( fvals )
#        vmaxv = np.nanmax( fvals )
        vminv,vmaxv = np.nanquantile( fvals, [0.025, 0.975])
    #
    
    triobj = ax.tripcolor(
        triang,
        facecolors = fvals,
        cmap = mycm,
        vmin = vminv,     # doesn't handle NaN well
        vmax = vmaxv,
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

def vis_fe_mesh_2d(fipy_cell_var, ax=None, **kwargs):
    '''
    Invokes pyplot.triplot to draw the grid mesh only.
    
    kwargs passed to pyplot.triplot directly.
    '''
    import numpy as np
    from matplotlib import tri,pyplot
    
    if ax is None:
        fig,ax = pyplot.subplots(1,1)
        flag_ax=True
    else:
        flag_ax=False
    
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

    triobj = ax.triplot(
        triang,
        **kwargs
    )
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
        pts = np.array([points])
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
        pt_subset = pts[active_pt_mask]
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

class Sk_Asymptotics:
    '''
    Class to compute and store the results of computing 
    predicted asymptotics of mass statistics.
    
    Initialize with an n-by-2 numpy array parameterizing the 
    boundary.
    
    Built off of demo12.py as a prototype.
    '''
    def __init__(self, boundary, cell_size_rel=0.005):
        import gen_mesh
        import numpy as np

        import fipy
#        from fipy import CellVariable, Gmsh2D, TransientTerm, DiffusionTerm
        
        assert len(np.shape(boundary))==2
        assert np.shape(boundary)[1] == 2
        self.boundary = boundary
        
        #self.fef = fe_flow.fe_flow(boundary, cell_size=cell_size, verbosity=0)
        
#        oneish = np.sqrt(10)/3. - 0.054  # something vaguely irrational
        domain_scale = np.max(boundary) - np.min(boundary)
        cellSize = cell_size_rel*domain_scale

        mesh = gen_mesh.gen_mesh(boundary,cellSize)
        self.mesh = mesh


        flow = fipy.CellVariable(name = r"$u$ (flow)", mesh=mesh, value=0.)

        flow.constrain(0., mesh.exteriorFaces) # dirichlet boundary condition

        forcing = fipy.CellVariable(mesh=mesh, name='pressure gradient', value=2.)

        eq = (-fipy.DiffusionTerm(coeff=1.) == forcing) # set up poisson eqn
        eq.solve(var=flow)


        #######

        # part 2 - use the flow as the driver for a Neumann problem (cell problem)
        # laplace(theta) = tphi, normal_deriv(theta) = 0.
        theta = fipy.CellVariable(name = r"$g_1$ (cell problem)", mesh=mesh, value=0.)

        # does this set Neumann boundary conditions?
        theta.faceGrad.constrain([0], mesh.exteriorFaces)

        # assign arbitrary value to a single thing
        # adjusted after the fact by mean-zero solvability condition at next level.
        theta.constrain(0, [1 if i==0 else 0 for i in range(len(mesh.facesLeft))])

        # subtract mean from flow
        mz_flow = fipy.CellVariable(name = "mean-zero flow", mesh=mesh, value=0.)
        mz_flow.value = flow.value - self.c_average(flow)

        # set up cell equation; theta in some texts; g_1 in my thesis.
        cell_eq = (-fipy.DiffusionTerm(coeff=1.) == mz_flow)

        # solve (?)
        cell_eq.solve(var=theta)

        # adjust theta
        mz_theta = fipy.CellVariable(name = "mean-zero cell", mesh=mesh, value=0.)
        mz_theta.value = theta.value - self.c_average(theta)


        # look at level curves of the cell problem. Should look perp to boundary.
        #ax.tricontour(cellX, cellY, mz_theta.value, 11, colors='w')

        ######################

        # set up and solve g_2 problem.

        ug1 = mz_flow*mz_theta

        g2_driver = fipy.CellVariable(name = r"$g_2$ driver", mesh=mesh, value=0.)
        g2_driver.value = 2*(ug1.value - self.c_average(ug1))

        g2 = fipy.CellVariable(name=r'$g_2$', mesh=mesh, value=0.)

        # set up cell equation; theta in some texts; g_1 in my thesis.
        g2_eq = (-fipy.DiffusionTerm(coeff=1.) == g2_driver)

        # solve (?)
        g2_eq.solve(var=g2)

        g2min = g2.value.min()
        g2max = g2.value.max()

        g2range = max(abs(g2min), abs(g2max))

        self.flow = flow
        self.mz_flow = mz_flow
        self.theta = theta
        self.g1 = mz_theta
        self.g2 = g2

        # look at level curves of the cell problem. Should look perp to boundary.


        ###########

        # Calculate the asymptotics formulas from "Mass distribution and skewness..."
        # 
        # Short-time (geometric/advective-only skewness)
        # compute <u~>, <u~^2>, <u^2>, <u^3>, then the asymptotics are
        #
        # M_1 ~ 0,
        # M_2 ~ 2t + Pe**2*<u^2>*t^2 - 2/3*Pe**2*<u~>*t^3
        # M_3 ~ Pe^3*<u^3>*t^3 - (<u~^2> - 2<u~>^2)*t^4
        #

        util1_avg = self.c_average(flow)
        util2_avg = self.c_average(flow**2)
        u2_avg = self.c_average(mz_flow**2)
        u3_avg = self.c_average(mz_flow**3)

        SG = u3_avg/(u2_avg**(3/2))

        # Long time skewness
        # 
        # Compute g1, g2 (done above) and <u*g_1>, <u*g_2>. 
        # Then skewness decays like
        # 3*Pe**3*<u*g_2>/(2 + 2*Pe**2*<u*g_1>)**(3/2) * t**(-1/2).
        #
        # Note the denominator term 2*Pe**2*<u*g_1> is the enhancement 
        # to the molecular diffusivity, which in these units is 2.
        #

        ug1_avg = self.c_average(mz_flow*mz_theta)
        ug2_avg = self.c_average(mz_flow*g2)

        SLONG = 3*ug2_avg/(2*ug1_avg)**(3/2)
        
        self.area = self.mesh.cellVolumes.sum()
        self.ly, self.lz = np.max( self.boundary, axis=0 ) - np.min( self.boundary, axis=0 )
        
        self.util1_avg = util1_avg
        self.util2_avg = util2_avg
        self.u2_avg = u2_avg
        self.u3_avg = u3_avg
        
        self.ug1_avg = ug1_avg
        self.ug2_avg = ug2_avg
        
        self.SG = SG
        self.SLONG = SLONG

        return
        
    def c_integrate(self,cvar):
        '''
        Use fipy routines to integrate an object (naive riemann sum)
        
        Input: cvar
            A fipy.variables.cellVariable.CellVariable
        Output: a scalar.
        '''
        # is there a higher-order version of this?
        # does it make sense to do anything more sophisticated?
        return (cvar.mesh.cellVolumes * cvar.value).sum()
    #

    def c_average(self,cvar):
        '''
        Use fipy routines to find the average an object; roughly,
        
        c_integrate(cvar)/Volume(cvar.mesh)
        
        This should be invariant to constants 
        (average of a constant function is the scalar constant)
        
        Input: cvar
            A fipy.variables.cellVariable.CellVariable
        Output: a scalar; the domain average of the object.
        
        '''
        # There's also a built-in cvar.cellVolumeAverage
        # This can be done more efficiently, but this won't be used often.
        return self.c_integrate(cvar)/(cvar.mesh.cellVolumes.sum())
    #
    
    def compute_ST_asymptotics(self, t, Pe=10**4):
    
        M1_asymp = np.zeros(t.shape)
        M2_asymp = 2*t + Pe**2*self.u2_avg*t**2 - 2/3*Pe**2*self.util1_avg*t**3
        M3_asymp = Pe**3*self.u3_avg*t**3 - (self.util2_avg - 2*self.util1_avg**2)*t**4

        Sk_asymp_short = M3_asymp/(M2_asymp)**(3/2) # right power?
        return M1_asymp,M2_asymp,M3_asymp,Sk_asymp_short
        
    def compute_LT_asymptotics(self, t, Pe=10**4):

        keff_long = 2*t + 2*Pe**2*self.ug1_avg*t
        M3_asymp_long = 3*Pe**3*self.ug2_avg*t
        Sk_asymp_long = M3_asymp_long/(keff_long)**(3/2)
        
        return keff_long, M3_asymp_long, Sk_asymp_long
        
    def return_statistics(self):
        '''
        Returns a collection of scalar statistics related to asymptotics for the cross section...
            0. Area of the cross section
            1. largest span in the z coordinate (horizontal coord)
            2. largest span in the y zoordinate (vertical coord)
            3. Cross-sectional average of flow (lab-frame)
            4. Average of u*g1, which is part of the enhanced diffusion kappa*(1 + Pe**2 * <ug1>)
            5. Geometric skewness; coefficient for skewness in advective time scales
            6. Long time skewness; coefficient for skewness decay to zero of the form LG*t**-1/2
        '''
        return ( self.area, self.lz, self.ly, self.util1_avg, self.ug1_avg, self.SG, self.SLONG )
