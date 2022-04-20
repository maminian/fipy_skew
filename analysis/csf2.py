import pandas
from matplotlib import pyplot, tri
import numpy as np
import datetime
import os

#pyplot.style.use('ggplot')

fig,ax = pyplot.subplots(1,1, figsize=(8,8), constrained_layout=True)

color_g = '#55f'
color_l = '#0ff'
HATCHME = False # hatches are misbehaving.

#####3

data_file = '../cross_section_features/trapezoid_asymptotics_april19_1.csv'
data_file_basename = os.path.basename(data_file)
df = pandas.read_csv(data_file)

# Insertion of triangulation made beforehand.
triang = tri.Triangulation(df['lambda'].values, df['q'].values)

region1 = np.logical_and(-0.1<df['skew_g'],df['skew_g']<0.1)
region2 = np.logical_and(-0.1<df['skew_l'],df['skew_l']<0.1)

# contours

skg_contours = ax.tricontour(triang,df['skew_g'], levels=[-0.1,0,0.1], vmin=-0.15,vmax=0.15, colors='k', linewidths=3, zorder=100)

skl_contours = ax.tricontour(triang,df['skew_l'], levels=[-0.1,0,0.1], vmin=-0.15,vmax=0.15, colors='k', linewidths=3, zorder=100)

##

# regions

ax.tricontourf(df['lambda'],df['q'],df['skew_g'], levels=[-0.1,0,0.1], vmin=-0.15,vmax=0.15, colors=color_g, alpha=0.8,zorder=10)

ax.tricontourf(df['lambda'],df['q'],df['skew_l'], levels=[-0.1,0,0.1], vmin=-0.15,vmax=0.15, colors=color_l, alpha=0.5,zorder=20)

ax.grid(c='#ddd', lw=2)

if HATCHME:
    #
    # with hatches...

    # TODO: hatches don't seem to respect the linear interpolation 
    # used by the rest of the contour tools. Can I get this to work?

    # TODO: bizarre behavior with tricontourf hatches being dependent on middle value 
    # in "levels" parameter. Does not align precisely with interpolated contours 
    # produced by tricontour. (likely due to the values region1/region2)
    fuzzy_thresh = 0.3
    ax.tricontourf(triang, region1, levels=[0,fuzzy_thresh,2], colors=None, hatches=['','\\\\'], alpha=0)
    ax.tricontourf(triang, region2, levels=[0,fuzzy_thresh,2], colors=None, hatches=['','//'], alpha=0)


#

if True:
    fig.savefig('short_long_intersection.pdf',
    metadata = {
        'Creator' : __file__ + " using data " + data_file_basename,
        'Author': 'Manuchehr Aminian'
        }
    )

fig.show()

