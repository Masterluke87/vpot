import numpy as np


def vpot(geom,elez,ri):
    """
    calculate the external potential at points r_i
    
    sum_A \dfrac{Z_A}{|R_A -r|}
    """
    V = 0.0
    for ZA,RA in zip(elez,geom):
        V += - ZA/(np.linalg.norm(RA-ri,axis=1))
    return V


def vBpot(phi,v):
    """
    VB = - \sum_{mu} V_{mu} phi_mu 
    """
    K = phi.shape[1]
    VB = 0.0 
    for mu in range(K):
            VB += - v[mu] * phi[:,mu] 
    return VB