import numpy as np
import utils
#import fipy
#import gen_mesh

class fe_flow:
    def __init__(self, boundary, cell_size=0.1, **kwargs):
        verb = kwargs.get('verbosity',0)

        self.boundary = boundary
        self.cell_size = cell_size

        if verb>0: print('Generating mesh... ', end='')
        self.generate_mesh()
        if verb>0: print('done.')

        if verb>0: print('Solving flow on mesh... ', end='')
        self.solve_flow()
        if verb>0: print('done.')

        if verb>0: print('Precalculating arrays for cell membership... ', end='')
        self.calculate_coeffs()
        if verb>0: print('done.')

        if verb>0: print('Precalculating cell neighborhoods... ', end='')
        self.calculate_cell_nbrs()
        if verb>0: print('done.')

        if verb>0: print('\nfe_flow object ready to use for simulation.')
        return
    #

    def generate_mesh(self):
        import gen_mesh
        self.mesh = gen_mesh.gen_mesh(self.boundary, self.cell_size)
        self.ncell = self.mesh.cellFaceIDs.shape[1]
        return
    #

    def solve_flow(self):
        '''
        Assuming the object has already generated a mesh,
        solve the flow problem in the domain using fipy.

        The flow is the solution to the Poisson problem
        -laplace(u)=2, u(x)=0 on the boundary.

        Note both the lab-frame and mean-flow frames are stored
        in self.lab_flow and self.flow, respectively.
        '''
        import fipy

        lab_flow = fipy.CellVariable(name = "lab_flow", mesh=self.mesh, value=0.)
        lab_flow.constrain(0., self.mesh.exteriorFaces) # dirichlet boundary condition

        forcing = fipy.CellVariable(mesh=self.mesh, name='pressure gradient', value=2.)

        eq = (-fipy.DiffusionTerm(coeff=1.) == forcing) # set up poisson eqn
        eq.solve(var=lab_flow)

        self.flow_lab = lab_flow    #yeah i know

        self.flow_average = self.c_average(self.flow_lab)

        self.flow = self.flow_lab - self.flow_average
        return
    #

    def calculate_coeffs(self):
        '''
        Given a mesh, precompute a collection of (3,3) arrays needed to
        test whether a point lies in the interior of the associated triangle, 
        and to help in calculating intersections.
        '''
        import numpy as np
        import utils

        #X = mg.vertexCoords[0,vci]
        #Y = mg.vertexCoords[1,vci]

#        self.c_mats = np.array([utils._get_cell_coeffs(np.vstack([xi,yi]).T) for xi,yi in zip(X,Y)])
        self.c_mats = np.array([self.cellFaceParams(i) for i in range(self.mesh.cellCenters.value.shape[1])])

        return
    #
    
    def hp_function(self,pt,faceID,cellID):
        '''
        Calculate the value of the linear function determining 
        which side of a face a point lies on. This does not use 
        precomputed coefficients, and requires an explicit 
        face (line segment) pointer and cell pointer. 
        
        It's assumed the faceID is one of the faces of the corresponding 
        cellID.
        
        The linear function is assumed 0 along the face, 
        and 1 at the indicated cell center.
        
        Outputs: scalar value of the respective half-plane function.
        '''
        import numpy as np
        cc = self.mesh.cellCenters.value[:,cellID] # cell center
        fc = self.mesh.faceCenters.value[:,faceID] # face center
        normal = self.mesh.faceNormals.data[:,faceID] # face normal
        A = np.array([[cc[0],cc[1],1.],[fc[0],fc[1],1],[-normal[1],normal[0],0]])
        coeff = np.linalg.solve(A,np.array([1.,0,0]))

        return coeff[0]*pt[0] + coeff[1]*pt[1] + coeff[2]
    #
    
    def cellFaceParams(self,cellID):
        '''
        Computes the parameter matrix for the linear functions 
        corresponding to the faces of the input cellID. 
        
        Ordering of the faces has the same order as indicated 
        in the mesh given by self.mesh.cellFaceIDs.
        
        Outputs: params; numpy array of shape (3,3)  Each row 
            corresponds to the parameters of the linear function 
            associated with the face.
        '''
        import numpy as np
        
        cc = self.mesh.cellCenters.value[:,cellID] # cell center
        faceIDs = self.mesh.cellFaceIDs.data[:,cellID]
        params = np.zeros((3,3), dtype=float)
        
        for i,fi in enumerate(faceIDs):
            fc = self.mesh.faceCenters.value[:,fi]
            normal = self.mesh.faceNormals.data[:,fi]
            mat = np.array([[1.,cc[0],cc[1]], [1,fc[0],fc[1]], [0,-normal[1],normal[0]]])
            parami = np.linalg.solve(mat, [1.,0,0])
            params[i] = parami
        #
        return params
    #
    
    def _solve_s(param, x0, x1):
        '''
        Solve for the parameterized time (fraction of length along a trajectory)
        for which a line from x0 to x1 intersects the linear 
        function defined by input param.
        
        Inputs:
            param : list or array length 3 of floats for the function 
                param[0] + param[1]*x + param[2]*y
            x0 : list or array length 2 of floats of initial position
            x1 : list or array length 2 of floats of final position
        Outputs:
            s : scalar indicating parameterized time of intersection.
                A value outside [0,1] indicates a non-physical intersection 
                for the trajectory from x0 to x1.
        '''
        numer = param[1]*x0[0] + param[2]*x0[1] + param[0]
        denom = param[1]*(x0[0]-x1[0]) + param[2]*(x0[1]-x1[1])
        return numer/denom
    #
    
    def argintersection(self,cellID,x0,x1):
        '''
        returns times of intersections for the linear trajectory from x0 to x1, 
        for the lines associated to the faces of cell defined by cellID.
        
        Typically we look for smallest nonnegative t in this interval 
        (first "real" intersection).
        
        Inputs:
            cellID : integer pointer indicating the cell of interest
            x0 : list or array length 2 of floats of initial position
            x1 : list or array length 2 of floats of final position
            
        Outputs:
            esses : numpy array shape (3,) of intersection times.
        '''
        import numpy as np
        A = self.cellFaceParams(cellID)
        return np.array([self._solve_s(ai,x0,x1) for ai in A])
    #

    def calculate_cell_nbrs(self):
        '''
        Precalculates approximate ordering of neighbor cells for every finite element cell.
        Should vastly improve lookup time for flows for particles if prior
        cell position is saved, since motion is continuous in time.
        '''
        from scipy import spatial
        import numpy as np
        D = spatial.distance_matrix(self.mesh.cellCenters.value.T, self.mesh.cellCenters.value.T)
        nbrs = np.zeros(D.shape, dtype=np.int)

        for j,row in enumerate(D):
            nbrs[j] = np.argsort(row)
        self.cell_nbrs = nbrs
        return
    #

    def get_flow(self, particles, exterior=0., pointers=None):
        '''
        get flow values for all particles in tandem.
        particles outside the domain are prescribed a flow
        according to by the *arg "exterior" (0 by default).

        Input:
            particles: numpy array of shape (n,2); n particles in two dimensions.
        Output:
            flows: numpy array of shape (n,) containing flow values
                associated with the coordinates of the particles.
        Optional arguments:
            exterior: scalar value; how to handle flow values outside
                the domain. Depending on the application this might take
                on different values. Default: 0.
        '''
        import numpy as np
        import utils

        n = np.shape(particles)[0]

        # identify the cell every particle lies in. If it does not lie in
        # any cells (mask==False), it's assigned to the default value "exterior".
        if type(pointers) == type(None):    #ugh
            mask,idx = utils.locate_cell(particles, self.mesh, c_mats = self.c_mats)
        else:
            #mask,idx = utils.locate_cell(particles, self.mesh, c_mats = self.c_mats)
            mask,idx = self.locate_cell2(particles, pointers)
        #

        flows = exterior*np.ones(n)
        flows[mask] = self.flow.value[idx[mask]]

        return flows
    #

    def c_integrate(self,cvar):
        # is there a higher-order version of this?
        # does it make sense to do anything more sophisticated?
        return (cvar.mesh.cellVolumes * cvar.value).sum()
    #

    def c_average(self,cvar):
        # There's also a built-in cvar.cellVolumeAverage
        # This can be done more efficiently, but this won't be used often.
        return self.c_integrate(cvar)/(cvar.mesh.cellVolumes.sum())
    #

    ####


    def locate_cell2(self, particles, pointers):
        '''
        Given a gmesh object (collection of vertices, faces, cells),
        generate a collection of pointers indicating which cell each
        point lies in.

        This version is distinguished by
            (a) primary loop is over points, not cells;
            (b) requires a list of indexes of previous cells each
                particle was. Used in conjunction with precomputed
                cell_nbrs list, lookups should be much faster.
            (c) utilizes previously calculated quantities in fe_flow() class.

        Input:
            particles: numpy array of shape (2,) or (n,2); either a singleton
                or a collection of points in 2D organized by row.
            pointers: numpy array of shape (n,) of pointers to the
                cells every particle previously lied.

        Outputs:
            mask: boolean array shape (n,) indicating whether the points
                lie in *any* of the triangles. You should use this as a
                preprocessing step to handle exceptions, etc.
            idx: Either an integer, or a numpy integer array shape (n,)
                indicating which cell one or more points lie within.
                Note that pointers for which mask==False are nonsense
                (the mask indicates they are outside the array, so not handled.)

        '''
        import numpy as np
        import fipy
        import utils

        pshape = np.shape(particles)
        if pshape==(2,):
            singleton = True
            n=1
            pts = [points]
        elif len(pshape)==2 and pshape[1]==2:
            singleton = False
            n = pshape[0]
            pts = particles
        else:
            raise Exception('Cannot handle input particles of shape ', np.shape(particles))
        #

        ncell = self.mesh.cellFaceIDs.shape[1]
        idx = np.array(pointers)
        interior_mask = np.ones(len(pointers), dtype=bool)

        search_stats = np.zeros(len(pointers), dtype=int)

        # LOOP OVER PARTICLES
        for i,o,particle in zip( range(len(particles)), pointers, particles ):
            # get which points lie in the current cell
            flag = False
            for ii,j in enumerate( self.cell_nbrs[o] ):
                flag = utils._in_Carr(particle, self.c_mats[j])
                if flag:    # is this the correct cell?
                    idx[i] = j
                    # print(ii)
                    break
            #
            if not flag:
                # couldn't find a cell! Must be in the exterior.
                interior_mask[i] = False
            #
            search_stats[i] = ii
        # end for

        stats = (np.mean(search_stats), np.quantile(search_stats,0.05), np.quantile(search_stats,0.50), np.quantile(search_stats,0.95))
        print('Search stats: mean: %.1f \t 5pct: %.1f 50pct: %.1f 95pct: %.1f'%stats)

        # done - return a scalar if a singleton was fed; else the idx array.
        if singleton:
            return interior_mask[0],idx[0]
        else:
            return interior_mask,idx
    #
#
