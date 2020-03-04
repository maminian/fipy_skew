import numpy as np
from matplotlib import pyplot

def l(xv,yv,coeffs):
    return coeffs[0] + coeffs[1]*xv + coeffs[2]*yv

def get_coeffs(point_coll):
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

def in_interior(point, Carr):
    import numpy as np
    if len(np.shape(point))==1:
        point = [point] # make 2d array if given a singleton
    #
    
    Pall = np.array( [ [1.,p[0],p[1]] for p in point] )
    return np.all( np.dot(Carr, Pall.T) >= 0, axis=0)
#

triangle = np.random.randn(3,2)
coeff_arr = get_coeffs(triangle)

test_points = np.random.randn(400,2)
flags = in_interior( test_points, coeff_arr )

# visualize
fig,ax = pyplot.subplots(1,1)

# plot triangle
for pair in [[0,1],[0,2],[1,2]]:
    ax.plot(triangle[pair,0], triangle[pair,1], c='k', lw=2, marker='o')
#

ax.scatter(test_points[:,0], test_points[:,1], c=flags, cmap=pyplot.cm.RdGy, s=20, alpha=0.9)

fig.show()
