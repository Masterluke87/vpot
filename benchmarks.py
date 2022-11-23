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


def testAlrichsBasis():
    """
    def2-QZVPD seems very good
    """
    mode = "min"
    M = myMolecule("tests/6-QM7/1218.xyz","def2-SVP")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    printStats(Gs,Vs)
    distanceVSerror(Gs,Vs,mode=mode,pltLabel="def2-SVP")
    
    M = myMolecule("tests/6-QM7/1218.xyz","def2-TZVP")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    printStats(Gs,Vs)
    distanceVSerror(Gs,Vs,mode=mode,pltLabel="def2-TZVP")


    M = myMolecule("tests/6-QM7/1218.xyz","def2-TZVPD")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    printStats(Gs,Vs)
    distanceVSerror(Gs,Vs,mode=mode,pltLabel="def2-TZVPD")

    M = myMolecule("tests/6-QM7/1218.xyz","def2-QZVP")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    printStats(Gs,Vs)
    distanceVSerror(Gs,Vs,mode=mode,pltLabel="def2-QZVP")

    M = myMolecule("tests/6-QM7/1218.xyz","def2-QZVPD")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    printStats(Gs,Vs)
    distanceVSerror(Gs,Vs,mode=mode,pltLabel="def2-QZVPD")


    if mode=="com":
        Dists = np.array([np.linalg.norm(Gs.points - Gs.mol.com,axis=1)]).transpose()
    if mode=="min":
        Dists = np.min(np.array([np.linalg.norm(Gs.points - x,axis=1) for x in Gs.mol.geom]).transpose(),axis=1)

    plt.plot(Dists,vpot(Gs.mol.geom,Gs.mol.elez,Gs.points)     ,"o",markersize=0.6,label="Potential")
    plt.legend(loc='best',markerscale=4)
    plt.show()


def testDunningBasis():
    """
    def2-QZVPD seems very good
    """
    mode = "min"
    M = myMolecule("tests/6-QM7/1218.xyz","cc-pvdz")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    printStats(Gs,Vs)
    distanceVSerror(Gs,Vs,mode=mode,pltLabel="cc-pvdz")
    
    M = myMolecule("tests/6-QM7/1218.xyz","cc-pvtz")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    printStats(Gs,Vs)
    distanceVSerror(Gs,Vs,mode=mode,pltLabel="cc-pvtz")


    M = myMolecule("tests/6-QM7/1218.xyz","cc-pvqz")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    printStats(Gs,Vs)
    distanceVSerror(Gs,Vs,mode=mode,pltLabel="cc-pvqz")

    M = myMolecule("tests/6-QM7/1218.xyz","cc-pv5z")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    printStats(Gs,Vs)
    distanceVSerror(Gs,Vs,mode=mode,pltLabel="cc-pv5z")

    
    M = myMolecule("tests/6-QM7/1218.xyz","aug-cc-pv5z")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    printStats(Gs,Vs)
    distanceVSerror(Gs,Vs,mode=mode,pltLabel="aug-cc-pv5z")

    if mode=="com":
        Dists = np.array([np.linalg.norm(Gs.points - Gs.mol.com,axis=1)]).transpose()
    if mode=="min":
        Dists = np.min(np.array([np.linalg.norm(Gs.points - x,axis=1) for x in Gs.mol.geom]).transpose(),axis=1)

    plt.plot(Dists,vpot(Gs.mol.geom,Gs.mol.elez,Gs.points)     ,"o",markersize=0.6,label="Potential")
    plt.legend(loc='best',markerscale=4)
    plt.show()

def alrichVSDunning():
    mode = "min"
    M = myMolecule("tests/6-QM7/1218.xyz","aug-cc-pv5z")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    printStats(Gs,Vs)
    distanceVSerror(Gs,Vs,mode=mode,pltLabel="aug-cc-pv5z")

    M = myMolecule("tests/6-QM7/1218.xyz","def2-QZVPD")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    printStats(Gs,Vs)
    distanceVSerror(Gs,Vs,mode=mode,pltLabel="def2-QZVPD")

    plt.show()

def testSphericalGrid():
    mode="min"
    M = myMolecule("tests/6-QM7/1218.xyz","aug-cc-pv5z")
    Gs = sphericalGrid(M,minDist=0.4,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    printStats(Gs,Vs)
    distanceVSerror(Gs,Vs,mode=mode,pltLabel="aug-cc-pv5z")

    # M = myMolecule("tests/6-QM7/1218.xyz","aug-cc-pv5z")
    # Gb = blockGrid(M,maxDist=4.5,minDist=0.4,gridSpacing=0.35)   
    # Vb = Gb.optimizeBasis()
    # printStats(Gb,Vb)
    # distanceVSerror(Gb,Vb,mode=mode,pltLabel="aug-cc-pv5z")

    # # distanceVSerror(Gb,Vs,mode=mode,pltLabel="S on B")

    # logging.info(f"Vs: {Vs.diagonal()}")
    # logging.info(f"Vb: {Vb.diagonal()}")
    # printStats(Gb,Vs)
    # printStats(Gs,Vb)
    plt.show()



if __name__ == "__main__":


    logging.basicConfig(filename='benchmark.log', level=logging.INFO,filemode="w")


    # for i in [26,38,110,170,194,230,266,302,350,
    #          434,590,770,974,1202,1454,1730,2030,
    #          2354,2702,3074,3470,3890,4334,4802,5294,5810]:
    #     M = myMolecule("tests/6-QM7/1.xyz","def2-TZVP")
    #     G = sphericalGrid(M,maxDist=7.5,nSphere=i)
    #     Vs = G.optimizeBasis()
    #     logging.info(f"Grid points: {i}")
    #     printStats(G,Vs)
        
    # testAlrichsBasis()
    # testDunningBasis()
    # testSphericalGrid()
    alrichVSDunning()




    # Gb = blockGrid(M,maxDist=3.5)  
    # Vb = Gb.optimizeBasis()
    # printStats(Gb,Vb)
    # distanceVSerror(Gb,Vb)


    

 