import pandas
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams.update({'font.size':14})

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


ax[0].tricontourf(df['lambda'],df['e'], df['skew_g'], levels=np.linspace(*l1,11), cmap=plt.cm.PRGn)

ax[1].tricontourf(df['lambda'],df['e'], df['skew_l'], levels=np.linspace(*l2,11), cmap=plt.cm.Greys)


ax[0].scatter(df_bads['lambda'], df_bads['e'], c='r', marker='x')
ax[1].scatter(df_bads['lambda'], df_bads['e'], c='r', marker='x')

for axi in ax:
    axi.set(xlabel=r'$\lambda$', ylabel=r'$e$', aspect='equal', xlim=[0,1], ylim=[0,1])

fig.show()

