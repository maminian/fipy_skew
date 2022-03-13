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


#######

def process_domain(inputs):
    i,name,pts = inputs

    asymptotics = utils.Sk_Asymptotics(pts, cell_size_rel=0.002)
    result = asymptotics.return_statistics()
    print(i, name)
    return [i,name] + list(result)
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
    else:
        pts = np.array([
        [-1./lam,1],
        [-q/lam,-1],
        [q/lam,-1],
        [1/lam,1]
        ])
    return pts


#############

# **must** Initialize pool after defining functions! 
p = multiprocessing.Pool(4)


##

aratios = np.linspace(0,1,41)[1:]
#inputs = [ [i,'ellipse_%s'%str(i).zfill(4),generate_ellipse(a)] for i,a in enumerate(aratios)]
#inputs = [ [i,'ellipse_%s'%str(i).zfill(4),generate_rectangle(a)] for i,a in enumerate(aratios)]
inputs = []
import itertools
lams = []
qs = []
for i,pair in enumerate(itertools.product( aratios, np.linspace(0,1,11) )):
    inputs.append([i, 'trapezoid_%s'%str(i).zfill(4), generate_trapezoid(*pair)])
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

