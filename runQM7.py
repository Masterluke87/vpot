from vpot.calc import myMolecule, sphericalGrid, blockGrid
from vpot.calc.potential import vpot,vBpot
from matplotlib import pyplot as plt
import numpy as np
import logging

def printStats(grid,V):
    logging.info(f"GridInfo : {grid.gridInfo} ")
    logging.info(f"residue  : {np.sum(np.square((vBpot(grid.phi,V.diagonal()) - vpot(grid.mol.geom,grid.mol.elez,grid.points))))}")
    logging.info(f"MeanError: {np.mean(np.square((vBpot(grid.phi,V.diagonal()) - vpot(grid.mol.geom,grid.mol.elez,grid.points))))}")
    logging.info(f"MaxError : {np.max(np.square((vBpot(grid.phi,V.diagonal()) - vpot(grid.mol.geom,grid.mol.elez,grid.points))))}")
    logging.info(f"MinError : {np.min(np.square((vBpot(grid.phi,V.diagonal()) - vpot(grid.mol.geom,grid.mol.elez,grid.points))))}")


def distanceVSerror(grid,V, mode="com",pltLabel=""):
    Error = vBpot(grid.phi,V.diagonal()) - vpot(grid.mol.geom,grid.mol.elez,grid.points)
    if mode=="com":
        Dists = np.array([np.linalg.norm(grid.points - grid.mol.com,axis=1)]).transpose()
    if mode=="min":
        Dists = np.min(np.array([np.linalg.norm(grid.points - x,axis=1) for x in grid.mol.geom]).transpose(),axis=1)    
    plt.plot(Dists,Error,"o",markersize=0.6,label=pltLabel)

def flushXYZ():
    A = read("tests/6-QM7/QM7.xyz",index=":")
    for c,i in enumerate(A):
        os.mkdir(f"QM7/{c}/")
        i.write(f"QM7/{c}/{c}.xyz")


if __name__ == "__main__":
    logging.basicConfig(filename='QM7.log', level=logging.INFO,filemode="w")


    mode="min"
    for i in range(10):
        M = myMolecule(f"QM7/{i}/{i}.xyz","aug-cc-pv5z")
        Gs = sphericalGrid(M,minDist=0.4,maxDist=4.5)   
        Vs = Gs.optimizeBasis()
        np.savez(f"QM7/{i}/input.npz", Vfit=Vs,Vpot=M.ao_pot, basis="aug-cc-pv5z")
        distanceVSerror(Gs,Vs,mode="min")
        plt.savefig(f"QM7/{i}/error.svg")
        plt.clf()

        E,wfn = M.runPSI4("HF")
        np.savez(f"QM7/{i}/output.npz", Da=wfn.Da().np, Fa=wfn.Fa().np)
        

        
        printStats(Gs,Vs)


