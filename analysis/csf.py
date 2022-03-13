import pandas
from matplotlib import pyplot
import numpy as np

pyplot.style.use('ggplot')

# todo: make folder reference generic to operating system
df = pandas.read_csv('../cross_section_features/trapezoid_asymptotics.csv')

bads = abs(df['skew_l'])>20
_temp = df['skew_l'].values
_temp[bads] = np.nan
df['skew_l'] = _temp

###

def vis_value(name, cmap, diverging=True, valrange=None, contour_levels=None):
    fig,ax = pyplot.subplots(1,1, figsize=(8,6), constrained_layout=True)

    if valrange is not None:
        valmin,valmax = valrange
    elif diverging:
        valrange = max(abs(df[name].min()), abs(df[name]).max())
        valmin,valmax = -valrange,valrange
    else:
        valmin,valmax=df[name].min(),df[name].max()
    
    #contour_levels = np.linspace(valmin,valmax,11)
    mask = ~df[name].isna()
    
    if contour_levels is None:
        contour_levels = np.linspace(valmin,valmax,11)
    
    contours = ax.tricontour(df['lambda'][mask], df['q'][mask], df[name][mask], levels=contour_levels, colors='k', linewidths=1,  extend='min',vmin=valmin, vmax=valmax)
    
    cobj = ax.tricontourf(df['lambda'][mask], df['q'][mask], df[name][mask], levels=contour_levels, cmap=pyplot.cm.PRGn, extend='min', vmin=valmin, vmax=valmax)

    ax.clabel(contours, contours.levels, inline=True, fmt=r'$%.1f$', fontsize=12)

    ax.set_xlabel(r'Aspect ratio $\lambda$')
    ax.set_ylabel(r'Trapezoid parameter $q$')
    
    ax.tick_params(axis='both', which='both', length=0)
    
    # todo: remove tick lines 
    cbar = fig.colorbar(cobj)
    
    return fig,ax,cbar
#


fig,ax,cbar = vis_value('skew_g', pyplot.cm.PRGn, contour_levels=np.arange(-0.7,0.7,0.1))
fig.show()

fig2,ax2,cbar2 = vis_value('skew_l', pyplot.cm.RdBu, valrange=[-1,1],contour_levels=np.arange(-1,1,0.1))
fig2.show()

pyplot.ion()

