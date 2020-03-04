# goal: generate a mesh for an arbitrary 
# closed region in 2D based on an ordered collection 
# of points along its boundary. 
#
# the first point doesn't need to be repeated 
# to the last point; this will be done internally.

# build input successively.
def gen_mesh(pts,cellSize):

    import fipy
    
    gmsh_arg = 'cellSize = %.8f; \n '%cellSize    # %.8f could cause truncation issues in edge cases
    idx = 0
    for p in pts:
        idx += 1
        gmsh_arg += 'Point(%i) = {%.8f,%.8f,0,%.8f}; \n '%(idx, p[0], p[1], cellSize)
    #

    line_ptrs = [str(jj+1+idx) for jj in range(len(pts))]
    for j in range(len(pts)):
        idx += 1
        gmsh_arg += 'Line(%i) = {%i,%i};\n'%(idx, 1+(j%len(pts)), 1+((j+1)%len(pts)))
        
#        line_ptrs += str(idx)
    #

    idx += 1
    gmsh_arg += 'Line Polygon(%i) = '%(idx,)
    gmsh_arg += '{' + ','.join(line_ptrs) + '}; \n'

    idx += 1
    gmsh_arg += 'Plane Surface(%i) = {%i};\n'%(idx, idx-1)

    mesh = fipy.Gmsh2D(gmsh_arg)

    return mesh
#

if __name__=="__main__":
    import numpy as np
    import fipy
    from matplotlib import pyplot
    
    import fipy
    
    # triangle
    pts = np.array([[-1,0],[1,0],[0,np.sqrt(3)]])
    
    # square
    pts = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])

    # octo
    s = 8
    pts = np.array([ [np.cos(th), np.sin(th)] for th in 2*np.pi/s*np.arange(s) ])

    cellSize = 0.1
    
    mesh = gen_mesh(pts,cellSize)
    dummy = fipy.CellVariable(mesh=mesh)
    
    fipy.Viewer(dummy, datamin=-1, datamax=1)
    
    fig = pyplot.gcf()
    ax = fig.axes[0]
    
    ax.collections[0].set_facecolors([0,0,0,0])
    ax.collections[0].set_edgecolors('k')
    
    ax.axis('square')
    #    tmesh = Gmsh2D('''
    #    cellSize = %(cellSize)g;
    #    tHeight = %(tHeight)g;
    #    Point(1) = {-tHeight,-1,0,cellSize};
    #    Point(2) = {+tHeight,-1,0,cellSize};
    #    Point(3) = {0,2,0,cellSize};
    #    Line(4) = {1,2};
    #    Line(5) = {2,3};
    #    Line(6) = {3,1};
    #    Line Polygon(7) = {4,5,6};
    #    Plane Surface(8) = {7};
    #    ''' % locals())
