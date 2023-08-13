import numpy as np

def generate_triangle(lam, e):
    '''
    Triangular domain with vertex coords:
    
        (-1/lam, -1)
        (1/lam, -1)
        (2q/lam, 1), q=(1-e)/e
    
    Which are then normalized to have incircle diameter 2, center 0.
        lam: ratio of height to base (between 0 and 1)
        e: relative positioning of third vertex in the x-coordinate.
            e=1: isosceles triangle
            e=1/2: right triangle
            e to 0: oblong (one angle greater than pi/2)
            Technically -1<e<0 is valid but is a reflection of 0<e<1.
    '''
    import triangles
    
    q = (1-e)/e
    pts = np.zeros((3,2), dtype=float)
    pts[0] = [-1/lam,-1]
    pts[1] = [1/lam,-1]
    pts[2] = [2*q/lam,1]
    
    # normalize
    pts = triangles.normalize_triangle(pts)
    return pts
    
    
def generate_ellipse(lam):
    th = np.linspace(0,2*np.pi, 100)
    x = 1/lam*np.cos(th[:-1])
    y = np.sin(th[:-1])
    pts = np.vstack([x,y]).T
    return pts

def generate_rectangle(lam):
    pts = np.array([
    [-1./lam, -1],
    [1./lam, -1],
    [1./lam, 1],
    [-1./lam, 1]
    ])
    return pts

def generate_trapezoid(lam,q):
    if q<1e-4:
        # assume we have q==0 and avoid duplicate point on boundary.
        pts = np.array([
        [-1./lam,1],
        [0,-1],
        [1/lam,1]
        ])
        pts += _RANDN_JITTER_MAG*np.random.randn(3,2)
    else:
        pts = np.array([
        [-1./lam,1],
        [-q/lam,-1],
        [q/lam,-1],
        [1/lam,1]
        ])
        pts += _RANDN_JITTER_MAG*np.random.randn(4,2)
    return pts


