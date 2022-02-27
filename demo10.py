# Purpose:
#
# Investigate pulling pixels from a hand drawn png.
#

from matplotlib import pyplot
import numpy as np
import utils

from PIL import Image

with Image.open('wombat.png') as f:
#with Image.open('rock.png') as f:
    img = np.asarray(f)

#img = img[::-1,:,0].T
img = img[:,:,0]>0
img = np.array(img, dtype=int)

border_img = np.gradient(img, axis=0)!=0
border = np.where(border_img)

coords = np.vstack(border[::-1]).T

# TODO: ordering based on nearest neighbors.
#o = utils.anticlockwise_order(coords)
#coords = coords[o]
from sklearn import metrics
D = metrics.pairwise_distances(coords)
keeps = np.zeros(len(coords), dtype=bool)
keeps[0] = True
for i in range(1,len(coords)):
    if not any(np.logical_and(D[i][keeps]>0, D[i][keeps]<10)):
        keeps[i] = True
#
coords = coords[keeps]
coords = coords[ utils.anticlockwise_order(coords) ]
coords = utils.smooth_boundary(coords)

if __name__ == "__main__":
    from matplotlib import pyplot

    fig,ax = pyplot.subplots(1,1)
    ax.imshow(img)
    
    ax.scatter(coords[:,0], coords[:,1], c='k')
    
    fig.show()
    pyplot.ion()

