import pandas
from matplotlib import pyplot as plt
import numpy as np

from sklearn import neighbors

plt.rcParams.update({'font.size':14})

def knn_uniformize(X,y, xg=np.linspace(0,1,101), yg=np.linspace(0,1,101)):
    '''
    Fits a k neighbors regressor on (X,y) (assumed unstructured)
    and outputs a triple x,y,z to be used with ax.scatter()
    
        knn = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance')
    
    so the regression is a inverse-square weighted combination of 10 nearest 
    neighbors in X.
    '''
    
    knn = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance')
    knn.fit(X, y)
    
    reg_xy = np.meshgrid(xg, yg)
    reg_xy = np.transpose(reg_xy, (1,2,0)).reshape(len(xg)*len(yg),2)
    vals = knn.predict(reg_xy)

    return reg_xy[:,0], reg_xy[:,1], vals, knn
###

df = pandas.read_csv('triangle_asymptotics.csv')
df_orig = df

mask = df.notna().all(axis=1)

df_bads = df[~mask]
df = df[mask]

l1 = df['skew_g'].quantile([0.05, 0.95])
l1 = [-max(abs(l1)), max(abs(l1))]

l2 = df['skew_l'].quantile([0.05,0.95]).values

#
fig,ax = plt.subplots(1,2, figsize=(8,4), constrained_layout=True)


#ax[0].tricontourf(df['lambda'],df['e'], df['skew_g'], levels=np.linspace(*l1,11), cmap=plt.cm.PRGn)

# alternatively...

x_g, y_g, s_g, _ = knn_uniformize(df[['lambda', 'e']].values, df['skew_g'].values)
ax[0].scatter(x_g, y_g, c=s_g, vmin=l1[0], vmax=l1[1], cmap=plt.cm.PRGn, marker='s')

x_l, y_l, s_l, _ = knn_uniformize(df[['lambda', 'e']].values, df['skew_l'].values)
ax[1].scatter(x_l, y_l, c=s_l, vmin=l2[0], vmax=l2[1], cmap=plt.cm.Greys)


for axi in ax:
    axi.set(xlabel=r'$\lambda$', ylabel=r'$e$', aspect='equal', xlim=[0,1], ylim=[0,1])
    #ax[0].scatter(df_bads['lambda'], df_bads['e'], c='r', marker='x')

fig.show()

