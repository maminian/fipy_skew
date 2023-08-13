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
import domains

import pandas

import multiprocessing
import time

_JITTER_MAG = 0.03   # jittering all boundary points by this times a randn.
NREPS = 2
NPROC = 8
np.random.seed(1234)   # reproducibility

#######

def process_domain(inputs):
    i,name,lam,e,pts = inputs
    
    # been a long time - I think this is just arbitrary, and trying to avoid 
    # floating point issues.
    cell_size_rel=0.001030411706
        
    # back to regularly scheduled programming
    try:
        # TODO: weird magic numbers making a problem down the line with 
        # singular matrices. consider some kind of random perturbation, 
        # or repeated retries thereof.
        asymptotics = utils.Sk_Asymptotics(pts, cell_size_rel=cell_size_rel)
        result = asymptotics.return_statistics()
        outputs = [i,name] + list(result)
    except:
        print(i, name, lam, e)
        print('\tREPORTS FAILURE OH NO' )
        
        outputs = [i,name] + [np.nan for _ in range(7)] # number of outputs is 7 as of Apr 19 2022
    print(i, name, lam, e)
    
    return outputs
##################


if __name__=="__main__":

    # **must** Initialize pool after defining functions! 
    p = multiprocessing.Pool(NPROC)


    ##
    # can't get into extreme aspect ratios...
    aratios = np.linspace(0,1,51)[5:]
    eccentricities = np.linspace(0,1,51)[5:]
    reps = np.arange(NREPS)

    inputs = []
    import itertools
    lams = []
    es = []
    for i,poople in enumerate(itertools.product( aratios, eccentricities, reps )):
        pair = poople[:2]   #don't care about the last value
        eps = _JITTER_MAG/2
        pair += np.random.uniform(-eps,eps, 2)
        pair = np.clip(pair, 0,1)
        
        inputs.append([i, 'triangle_%s'%str(i).zfill(8), pair[0], pair[1], domains.generate_triangle(*pair)])
        lams.append(pair[0])
        es.append(pair[1])

    outputs = p.map(process_domain, inputs)

    ###
    # Outputs into dataframe...

    df = pandas.DataFrame(
        outputs,
        columns=['index', 'name', 'area','span_z', 'span_y', 'u_avg', 'k_enh', 'skew_g', 'skew_l']
    )

    df['lambda'] = lams
    df['e'] = es

    if True:
        df.to_csv('triangle_asymptotics.csv', index=None)

