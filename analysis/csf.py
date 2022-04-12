import pandas
from matplotlib import pyplot
import numpy as np

pyplot.style.use('ggplot')

# todo: make folder reference generic to operating system
df = pandas.read_csv('../cross_section_features/trapezoid_asymptotics2.csv')

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
    
    contours = ax.tricontour(df['lambda'][mask], df['q'][mask], df[name][mask], levels=contour_levels, colors='k', linewidths=1,vmin=valmin, vmax=valmax)
    
    cobj = ax.tricontourf(df['lambda'][mask], df['q'][mask], df[name][mask], levels=contour_levels, cmap=cmap, vmin=valmin, vmax=valmax)

    ax.clabel(contours, contours.levels, inline=True, fmt=r'$%.1f$', fontsize=12)

    ax.set_xlabel(r'Aspect ratio $\lambda$')
    ax.set_ylabel(r'Trapezoid parameter $q$')
    
    ax.tick_params(axis='both', which='both', length=0)
    
    # todo: remove tick lines 
    cbar = fig.colorbar(cobj)
    
    return fig,ax,cbar
#


fig,ax,cbar = vis_value('skew_g', pyplot.cm.PRGn, valrange=[-1,1],contour_levels=np.arange(-0.7,0.7,0.1))
fig.show()

fig2,ax2,cbar2 = vis_value('skew_l', pyplot.cm.PRGn, valrange=[-1,1],contour_levels=np.arange(-0.7,0.7,0.1))
fig2.show()

df['log_k_enh'] = np.log10(df['k_enh'])

fig3,ax3,cbar3 = vis_value('log_k_enh', pyplot.cm.viridis, valrange=[-3,0], contour_levels=np.arange(-4,0.5,0.5))
fig3.show()

# Check if trapezoids do the rectangle when q==1 (index==20 mod 21 for trapezoid asymptotics)
fig4,ax4 = pyplot.subplots(1,1, figsize=(6,5), constrained_layout=True)
# compare to direct rectangle sim
df_r = pandas.read_csv('../cross_section_features/rectangle_asymptotics.csv')
ax4.plot(df_r['span_z']/df_r['span_y'], df_r['skew_g'], marker='s', c='k', label=r'Rectangles')

for i in range(6):
    df_sub = df.iloc[4*i::21]
    qv = df_sub['q'].values[0]
    ax4.plot(df_sub['lambda'], df_sub['skew_g'], marker='^', label=r'Trapezoid $q=%.2f$'%qv, c=pyplot.cm.tab10(i))

ax4.set_xlabel(r'Aspect ratio $\lambda$')
ax4.set_ylabel(r'Geometric skewness')
ax4.set_title('Geometric skewness' , loc='left')
ax4.legend()

fig4.show()

#######

if False:
    fig.savefig('skew_g_trapezoids.png')
    fig2.savefig('skew_l_trapezoids.png')
    fig3.savefig('log_k_enh_trapezoids.png')
    fig4.savefig('trap_rect_comparison.png')
#


pyplot.ion()

