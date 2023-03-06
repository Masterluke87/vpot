import numpy as np
from scipy.special import erf


def V_Z_ANC(Z : int, a : int,  r: np.array):
    """
    Returns the value of the V_Z function for the given parameters.

    Parameters:
        Z (float): the value of Z
        a (float): the value of a
        b (float): the value of b
        r (float): the value of r

    Returns:
        float: the value of V_Z(Z, a, b, r) = Z^2 * V(a, b, Z*r)
    """

    AB = {1 : -3.6442293856e-01,
          2 : -1.9653418982e-01,
          3 : -1.3433604753e-01,
          4 : -1.0200558466e-01,
          5 : -8.2208091118e-02,
          6 : -6.8842555167e-02,
          7 : -5.9213652850e-02, 
          8 : -5.1947028250e-02,
          9 : -4.6268559218e-02,
          10 : -4.1708913494e-02,
          11 : -3.7967227308e-02,
          12 : -3.4841573775e-02}

    b = AB[a]



    def h(a, b, r):
        erf_term = -r * erf(a*r)
        exp_term = b * np.exp(-a**2 * r**2)
        return erf_term + exp_term

    def h_prime(a, b, r):
        erf_term = -erf(a*r)
        exp_term = -2 * ((a**2) * b + (a / np.sqrt(np.pi))) * r * np.exp(-a**2 * r**2)
        return erf_term + exp_term

    def h_double_prime(a, b, r):
        exp_term = (-2 * (a**2) * b - (4 * a) / np.sqrt(np.pi) 
                    + (4 * (a**4) * b + (4 * (a**3)) / np.sqrt(np.pi)) * r**2) * np.exp(-a**2 * r**2)
        return exp_term


    def V(a, b, r):
        h_prime_term = h_prime(a, b, r)
        h_double_prime_term = h_double_prime(a, b, r)
        v = -0.5 + (h_prime_term / r) + (h_prime_term**2 / 2) + (h_double_prime_term / 2)
        return v



    return Z**2 * V(a, b, Z*r)


def vpotANC(geom,elez,ri,a):
    """
    calculate the ANC external potential at points r_i
    """
    V = 0.0
    for ZA,RA in zip(elez,geom):
        V += V_Z_ANC(ZA,a,np.linalg.norm(RA-ri,axis=1))
    return V


def vpot(geom,elez,ri):
    """
    calculate the exact external potential at points r_i
    
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
