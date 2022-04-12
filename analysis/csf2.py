import pandas
from matplotlib import pyplot
import numpy as np

pyplot.style.use('ggplot')

fig,ax = pyplot.subplots(1,1, figsize=(8,8), constrained_layout=True)

df = pandas.read_csv('../cross_section_features/trapezoid_asymptotics2.csv')

# contours

skg_contours = ax.tricontour(df['lambda'],df['q'],df['skew_g'], 
levels=[-0.1,0,0.1], vmin=-0.15,vmax=0.15, 
colors='k', linewidths=3)

skl_contours = ax.tricontour(df['lambda'],df['q'],df['skew_l'], 
levels=[-0.1,0,0.1], vmin=-0.15,vmax=0.15, 
colors='k', linewidths=3)

##

# regions

ax.tricontourf(df['lambda'],df['q'],df['skew_g'], 
levels=[-0.1,0,0.1], vmin=-0.15,vmax=0.15, 
colors='#008', alpha=0.4)

ax.tricontourf(df['lambda'],df['q'],df['skew_l'], 
levels=[-0.1,0,0.1], vmin=-0.15,vmax=0.15, 
colors='#800', alpha=0.4)

#
# with hatches...

ax.tricontourf(df['lambda'],df['q'],np.logical_and(-0.1<df['skew_l'],df['skew_l']<0.1), levels=[0,1,2], vmin=-0.15,vmax=0.15, colors=None, hatches=['','\\\\'], alpha=0)

ax.tricontourf(df['lambda'],df['q'],np.logical_and(-0.1<df['skew_g'],df['skew_g']<0.1), levels=[0,1,2], vmin=-0.15,vmax=0.15, colors=None, hatches=['','//'], alpha=0)

fig.show()

