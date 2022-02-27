# Purpose:
#
# Investigate ways of loading an arbitrary sketch in an svg
# (with many limitations...)
#
# The expectation is to load a sequence of pixels with their coordinates 
# given, for example.
#
# Following https://stackoverflow.com/a/15857847
# the svg can be loaded using a sub-module from the xml module
# 
# THIS ONLY SEEMS TO WORK FOR PATHS
# ALSO UNCLEAR HOW TO ACTUALLY GRAB WHAT WE WANT FROM WHAT'S STORED THERE
# SINCE THE COORDINATES ARE PARAMETERIZING SPLINES

import xml
import re
from matplotlib import pyplot
import numpy as np
import utils

prog = re.compile('([0-9\.\-]{1,},[0-9\.\-]{1,})')


doc = xml.dom.minidom.parse('wombat2.svg')
path_strings = [path.getAttribute('d') for path in doc.getElementsByTagName('path')]
#doc.unlink()

coords_strings = prog.findall(path_strings[0])

coords = [[float(s) for s in row.split(',')] for row in coords_strings]
coords = np.array(coords)
coords = coords[1:] # first coord has to do with some other property?

# NOTE: values are relative in reference to prior point. So, need to 
# do a cumulative sum here.
coords = np.cumsum(coords, axis=0)
coords[:,1] = coords[:,1]*-1
# this won't work except with convex-ish bodies
# However, gmsh might not care about order.
#o = utils.anticlockwise_order(coords)
#coords = coords[o]
#coords = np.cumsum(coords, axis=0)

if __name__=="__main__":
    fig,ax = pyplot.subplots(1,1)
    ax.plot(coords[:,0], coords[:,1], c='k', lw=1, marker='.')
    ax.scatter(coords[:,0], coords[:,1], c=range(len(coords)) )

    fig.show()

