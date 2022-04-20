
from matplotlib import pyplot, patches
import numpy as np

def generate_trapezoid_loop(lam,q):
    pts = np.array([
    [-1./lam,1],
    [-q/lam,-1],
    [q/lam,-1],
    [1/lam,1],
    # repeat first point
    [-1./lam,1]
    ])

    return pts

def make_arrow_patch(coord1, coord2):
    return patches.FancyArrowPatch(
        posA=coord1, 
        posB=coord2,
        arrowstyle='<|-|>',
        facecolor='k',
        edgecolor=None,
        shrinkA=0,shrinkB=0, mutation_scale=20,
        zorder=100
    )
def add_arrow(axi, coord1, coord2, zorder=0):
    axi.add_artist( make_arrow_patch(coord1,coord2) )
    return
######

pr = [0.2, 0.5]
pair1 = [0.25, 0.9]
pair2 = [0.3,0.1]

#####

fig = pyplot.figure(constrained_layout=True, figsize=(12,4))

ax = fig.subplot_mosaic('''
AAB
AAC
''')

for axn,pair in zip(['A','B','C'], [pr,pair1,pair2], ):
    # essential
    X = generate_trapezoid_loop(*pair)
    ax[axn].plot(X[:,0], X[:,1], marker='.', markersize=10, c='k')
    
    # polish
    ax[axn].fill(X[:,0], X[:,1], c='#ddd')
    ax[axn].text(-5.9,-1.45, r'$(\lambda,q) = (%.2f,%.2f)$'%tuple(pair), ha='left', va='bottom', fontsize=12 )
    ax[axn].text(-5.9,1.45, axn, ha='left', va='top', fontsize=16, fontweight='bold')
    
    ax[axn].set_ylim([-1.5,1.5])
    ax[axn].set_xlim([-6,6])
    
    ax[axn].set_xticks([-5,0,5])
    ax[axn].set_yticks([-1,0,1])
    ax[axn].grid(c='#ddd', lw=1.5, zorder=-100)
    

# fancy arrow thing to illustrate parameters.

# horizontal span arrow
#ax["A"].quiver(-1/pr[0], 1.1, 2/pr[0], 0, units='x', scale=1)

add_arrow(ax["A"], [-1/pr[0], 1.1], [1/pr[0], 1.1])
ax["A"].text(0, 1.12, r'$2/\lambda$', ha='center', va='bottom', fontsize=14, zorder=100)

add_arrow(ax["A"], [-pr[1]/pr[0], -1.1], [pr[1]/pr[0], -1.1])
ax["A"].text(0, -1.12, r'$2q/\lambda$', ha='center', va='top', fontsize=14, zorder=100)

ax["A"].set_xlabel(r"$z$")
ax["A"].set_ylabel(r"$y$")

#
fig.savefig('trapezoid_param_schematic.pdf',
    metadata = {
        'Creator' : __file__,
        'Author': 'Manuchehr Aminian'
        }
)

fig.show()
pyplot.ion()

