# Demo showing how one can feed 
# mesh-valued functions from fipy into one 
# another to successively solve the 
# moment equations
# 
# (ex: solving the cell problem Laplace(theta) = u, Neumann boundaries)
#

import numpy as np
from matplotlib import pyplot
import os

import utils

import pandas


import multiprocessing

_RANDN_JITTER_MAG = 1e-6    # jittering all boundary points by this times a randn.
NPROC = 4
np.random.seed(0)   # reproducibility

#######

def process_domain(inputs):
    i,name,pts = inputs
    
    # cell_size_rel seems to not do what it ought to, and 
    # results in enormous triangulations for 1/lambda domains order 1 (near squares).
    # just do a dirty hack here to address, and see what happens.
    aratio_inv = pts[-1][0]  # 1/lambda
    lam = 1/aratio_inv
    q = pts[-2][0]/pts[-1][0]
    
    if aratio_inv < 5:
        cell_size_rel=0.020304
    else:
        cell_size_rel=0.0050607
        
    # back to regularly scheduled programming
    try:
        # TODO: weird magic numbers making a problem down the line with 
        # singular matrices. consider some kind of random perturbation, 
        # or repeated retries thereof.
        asymptotics = utils.Sk_Asymptotics(pts, cell_size_rel=cell_size_rel)
        result = asymptotics.return_statistics()
        outputs = [i,name] + list(result)
    except:
        print(i, name, lam, q)
        print('\tREPORTS FAILURE OH NO' )
        
        outputs = [i,name] + [np.nan for _ in range(7)] # number of outputs is 7 as of Apr 19 2022
    print(i, name, lam, q)
    return outputs
##################


def generate_ellipse(lam):
    th = np.linspace(0,2*np.pi, 100)
    x = 1/lam*np.cos(th[:-1])
    y = np.sin(th[:-1])
    pts = np.vstack([x,y]).T
    return pts

def generate_rectangle(lam):
    pts = np.array([
    [-1./lam, -1],
    [1./lam, -1],
    [1./lam, 1],
    [-1./lam, 1]
    ])
    return pts

def generate_trapezoid(lam,q):
    if q<1e-4:
        # assume we have q==0 and avoid duplicate point on boundary.
        pts = np.array([
        [-1./lam,1],
        [0,-1],
        [1/lam,1]
        ])
        pts += _RANDN_JITTER_MAG*np.random.randn(3,2)
    else:
        pts = np.array([
        [-1./lam,1],
        [-q/lam,-1],
        [q/lam,-1],
        [1/lam,1]
        ])
        pts += _RANDN_JITTER_MAG*np.random.randn(4,2)
    return pts


#############

# **must** Initialize pool after defining functions! 
p = multiprocessing.Pool(NPROC)


##

aratios = np.linspace(0,1,51)[1:]
eccentricities = np.linspace(0,1,51)
reps = np.arange(10)

#inputs = [ [i,'ellipse_%s'%str(i).zfill(4),generate_ellipse(a)] for i,a in enumerate(aratios)]
#inputs = [ [i,'ellipse_%s'%str(i).zfill(4),generate_rectangle(a)] for i,a in enumerate(aratios)]
inputs = []
import itertools
lams = []
qs = []
for i,poople in enumerate(itertools.product( aratios, eccentricities, reps )):
    pair = poople[:2]   #don't care about the last value
    inputs.append([i, 'trapezoid_%s'%str(i).zfill(8), generate_trapezoid(*pair)])
    lams.append(pair[0])
    qs.append(pair[1])
outputs = p.map(process_domain, inputs)

###
# Outputs into dataframe...

df = pandas.DataFrame(
    outputs,
    columns=['index', 'name', 'area','span_z', 'span_y', 'u_avg', 'k_enh', 'skew_g', 'skew_l']
)

df['lambda'] = lams
df['q'] = qs

if False:
    df.to_csv('trapezoid_asymptotics.csv', index=None)

