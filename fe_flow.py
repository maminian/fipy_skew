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
        test whether a point lies in the interior of the associated triangle.
        '''
        import numpy as np
        import utils

        vci = [np.unique(self.mesh.faceVertexIDs[:,ci]) for ci in self.mesh.cellFaceIDs.T]
        X,Y = self.mesh.vertexCoords[:,vci]
        #X = mg.vertexCoords[0,vci]
        #Y = mg.vertexCoords[1,vci]

        # TODO - repackage everything in a nicer way that I'm not constantly
        # doing transposes, elementwise operations, etc.
        self.c_mats = np.array([utils._get_cell_coeffs(np.vstack([xi,yi]).T) for xi,yi in zip(X,Y)])

        return
    #

    def get_flow(self, particles, exterior=0.):
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
        mask,idx = utils.locate_cell(particles, self.mesh, c_mats = self.c_mats)

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
#
