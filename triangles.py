import numpy as np

def incircle_diameter(pts):
    '''
    Calculates radius of incircle.
    See https://en.wikipedia.org/wiki/Incircle_and_excircles#Radius
    (Accessed Aug 11, 2023)
    
    Input: pts; array shape (3,2)
    Output: scalar; incircle diameter.
    '''
    p2 = np.zeros((pts.shape[0]+1, pts.shape[1]))
    p2[:len(pts)] = pts
    p2[-1] = pts[0]
    dists = np.linalg.norm( np.diff(p2, axis=0), 2, axis=1 )
    s = sum(dists)/2
    return 2*np.sqrt((s-dists[0])*(s-dists[1])*(s-dists[2])/s)
    
def incircle_center(pts):
    '''
    Calculates center of incircle.
    See https://en.wikipedia.org/wiki/Incircle_and_excircles#Cartesian_coordinates
    (Accessed Aug 11, 2023)
    
    Inputs: pts; array shape (3,2)
    Output: array shape (2,)
    '''
    p2 = np.zeros((pts.shape[0]+1, pts.shape[1]))
    p2[:len(pts)] = pts
    p2[-1] = pts[0]
    dists = np.linalg.norm( np.diff(p2, axis=0), 2, axis=1 ) # c, a, b
    dists = np.roll(dists, -1) # a, b, c
    center = np.dot(pts.T, dists)/sum(dists)
    return center

def normalize_triangle(pts):
    '''
    outputs a*(pts-c)
    such that the incircle center is (0,0) and the incircle diameter is 2.
    '''
    c = incircle_center(pts)
    d = incircle_diameter(pts)
    return 2/d*(pts-c)

def sketch(pts, ax=None):
    from matplotlib import patches
    
    if ax is None:
        from matplotlib import pyplot as plt
        flag = True
        fig,ax = plt.subplots()
    else:
        flag = False
    
    p2 = np.tile(pts, (2,1))
    
    ax.plot(p2[:4,0], p2[:4,1], marker='o', lw=3, c='#808')
    
    # circle center
    c = incircle_center(pts)
    d = incircle_diameter(pts)
    # circle diam
    cpatch = patches.Circle(c, radius=d/2, facecolor=[1,1,1,0], edgecolor='#000', linewidth=3)
    
    for p in np.roll(pts, -1, axis=0):
        v = (p-c)/np.linalg.norm(p-c)
        ax.plot([c[0], p[0]], [c[1], p[1]], c='r', ls='dotted', linewidth=1)
        ax.plot([c[0], c[0] + d/2*v[0]], [c[1], c[1] + d/2*v[1]], c='r', ls='-', linewidth=1.5)
    
    ax.add_patch(cpatch)
    ax.set_aspect('equal')
    if flag:
        return fig,ax
    else:
        return None
    
#

if __name__=="__main__":
    from matplotlib import ticker
    from matplotlib import pyplot as plt
    
    plt.rcParams.update({'font.size':16})
    
    np.random.seed(271828)
    pts = np.random.normal(0, 1, (3,2))
    pts2 = normalize_triangle(pts)
    
    fig,ax = plt.subplots(1,2, figsize=(8,4), sharex=True, sharey=True, constrained_layout=True)
    
    sketch(pts, ax[0])
    sketch(pts, ax[1])
    for l in ax[1].lines:
        l.set_alpha(0.1)
    for c in ax[1].collections:
        c.set_alpha(0.1)
    for p in ax[1].patches:
        p.set_alpha(0.1)
    
    sketch(pts2, ax[1])
    for axi in ax:
        for xy in ['xaxis', 'yaxis']:
            getattr(axi,xy).set_major_locator(ticker.MultipleLocator(1))
        axi.grid(zorder=-1000, alpha=0.4)
    
    '''
    # arrows pointing from old vertices to new
    for i in range(3):
        pp = pts[i]
        v = pts2[i] - pp
        ax[1].arrow(pp[0], pp[1], v[0], v[1], 
            color='#666', linewidth=1, head_width=0.2, length_includes_head=True)
    '''
    
    
    ax[0].set(xlabel=r'$x$', ylabel=r'$y$')
    ax[1].set(xlabel=r'$x$')
    
    fig.savefig('triangle_norm_vis.pdf', bbox_inches='tight')
    fig.savefig('triangle_norm_vis.png', bbox_inches='tight')
    fig.show()
    
