from vpot.calc import myMolecule, sphericalGrid, blockGrid
from vpot.calc.potential import vpot,vBpot
from matplotlib import pyplot as plt
import numpy as np
import logging
from ase.io import read
import psi4
import os
import sys

def flushAllXYZ():
    A = read("QM7/QM7.xyz",index=":")
    for c,i in enumerate(A):
        flushXYZ(c)
        
def flushXYZ(index : int):
    A = read("QM7/QM7.xyz",index=":")
    if not os.path.exists(f"QM7/{index}/{index}.xyz"):
        os.mkdir(f"QM7/{index}/")
        A[index].write(f"QM7/{index}/{index}.xyz")


def runIndex(index: int):
    flushXYZ(index)
    M  = myMolecule(f"QM7/{index}/{index}.xyz","aug-cc-pv5z")
    Gs = sphericalGrid(M,minDist=0.4,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    np.savez(f"QM7/{index}/input.npz", 
                Vfit=Vs,
                Vpot=M.ao_pot, 
                Nel=M.nElectrons,
                basis="aug-cc-pv5z",
                gridInfo=Gs.gridInfo)
    Gs.printStats(Vs)
    Gs.exportErrorVsDistance(Vs,mode="min",pltLabel="Error",path=f"QM7/{index}/error.svg",plotPot=True)


    psi4.core.set_output_file(f'QM7/{index}/psi.out', False)
    E,wfn = M.runPSI4("HF")
    np.savez(f"QM7/{index}/output.npz", Da=wfn.Da().np, Fa=wfn.Fa().np)

    


if __name__ == "__main__":
    logging.basicConfig(filename='QM7.log', level=logging.INFO,filemode="w")
    psi4.set_memory("16 Gb")
    
    psi4.core.set_num_threads(6)

    index = int(sys.argv[1])

    runIndex(index)
        

        
       

