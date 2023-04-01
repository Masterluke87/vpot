from vpot.calc import myMolecule
from vpot.calc.grids import sphericalGrid, blockGrid, pointGrid,sphericalAtomicGrid,blockAtomicGrid,sphericalIndexGrid
from vpot.calc.potential import vpot,vBpot, vpotANC
from vpot.calc import DFTGroundState
from matplotlib import pyplot as plt
from scipy.special import erf
from scipy.optimize import minimize
from vpot.calc.grids import sphericalAtomicGrid
from psi4.driver import qcdb
from ase.data import atomic_numbers


import psi4
import numpy as np
import logging,time

plt.style.use("dark_background")

if __name__ == "__main__":
    nSphere = 590
    nRadial = 300
    prec=2

    M = myMolecule("tests/CH3Cl.xyz","def2-TZVP")

    atomIdx = 2
    M.keepAugmentBasisForIndex(atomIdx)

    a,newOrb = qcdb.BasisSet.pyconstruct(M.psi4Mol.to_dict(),'BASIS',
                                            "def2-TZVP",fitrole='ORBITAL',
                                            other=None,return_dict=True,return_atomlist=False)
    for i in newOrb["shell_map"]:
            del i[2:]

    for c,i in enumerate(M.getOrbitalDict()["shell_map"]):
        if c == atomIdx:
            newOrb["shell_map"][c] += i[2:]

    breakpoint()
    M.setBasisDict(orbitalDict=newOrb,augmentDict=M.getAugmentDict())

    Gb =  sphericalIndexGrid(M,atomIdx,minDist=0.0,maxDist=2.0,nSphere=nSphere,nRadial=nRadial,pruningScheme="None") 
    V1 = Gb.optimizeBasis(potentialType="anc",a=2)
