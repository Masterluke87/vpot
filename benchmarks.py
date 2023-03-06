from vpot.calc import myMolecule, sphericalGrid, blockGrid
from vpot.calc.potential import vpot,vBpot, vpotANC
from matplotlib import pyplot as plt
import numpy as np
import logging

def testANCMolecule(prec):
    M = myMolecule("tests/6-QM7/1218.xyz","def2-TZVPD-decon")
    Gs = sphericalGrid(M,minDist=0.01)   
    Vs = Gs.optimizeBasis(potentialType="anc",a=prec)
    Gs.printStats(Vs)
    Gs.exportErrorVsDistance(Vs,pltLabel="def2-SVP",plotPot=True)
    




def testAlrichsBasis():
    """
    def2-QZVPD seems very good
    """
    mode = "min"
    M = myMolecule("tests/6-QM7/1218.xyz","def2-SVP")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    Gs.printStats(Vs)
    Gs.exportErrorVsDistance(Vs,mode=mode,pltLabel="def2-SVP")
    
    M = myMolecule("tests/6-QM7/1218.xyz","def2-TZVP")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    Gs.printStats(Vs)
    Gs.exportErrorVsDistance(Vs,mode=mode,pltLabel="def2-TZVP")


    M = myMolecule("tests/6-QM7/1218.xyz","def2-TZVPD")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    Gs.printStats(Vs)
    Gs.exportErrorVsDistance(Vs,mode=mode,pltLabel="def2-TZVPD")

    M = myMolecule("tests/6-QM7/1218.xyz","def2-QZVP")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    Gs.printStats(Vs)
    Gs.exportErrorVsDistance(Vs,mode=mode,pltLabel="def2-QZVP")

    M = myMolecule("tests/6-QM7/1218.xyz","def2-QZVPD")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    Gs.printStats(Vs)
    Gs.exportErrorVsDistance(Vs,mode=mode,pltLabel="def2-QZVPD")


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
    Gs.printStats(Vs)
    Gs.exportErrorVsDistance(Vs,mode=mode,pltLabel="cc-pvdz")
    
    M = myMolecule("tests/6-QM7/1218.xyz","cc-pvtz")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    Gs.printStats(Vs)
    Gs.exportErrorVsDistance(Vs,mode=mode,pltLabel="cc-pvtz")


    M = myMolecule("tests/6-QM7/1218.xyz","cc-pvqz")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    Gs.printStats(Vs)
    Gs.exportErrorVsDistance(Vs,mode=mode,pltLabel="cc-pvqz")

    M = myMolecule("tests/6-QM7/1218.xyz","cc-pv5z")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    Gs.printStats(Vs)
    Gs.exportErrorVsDistance(Vs,mode=mode,pltLabel="cc-pv5z")

    
    M = myMolecule("tests/6-QM7/1218.xyz","aug-cc-pv5z")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    Gs.printStats(Vs)
    Gs.exportErrorVsDistance(Vs,mode=mode,pltLabel="aug-cc-pv5z")

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
    Gs.printStats(Vs)
    Gs.exportErrorVsDistance(Vs,mode=mode,pltLabel="aug-cc-pv5z")

    M = myMolecule("tests/6-QM7/1218.xyz","def2-QZVPD")
    Gs = sphericalGrid(M,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    Gs.printStats(Vs)
    Gs.exportErrorVsDistance(Vs,mode=mode,pltLabel="def2-QZVPD")

    plt.show()

def testSphericalGrid():
    mode="min"
    M = myMolecule("tests/6-QM7/1218.xyz","aug-cc-pv5z")
    Gs = sphericalGrid(M,minDist=0.4,maxDist=4.5)   
    Vs = Gs.optimizeBasis()
    Gs.printStats(Vs)
    Gs.exportErrorVsDistance(Vs,mode=mode,pltLabel="aug-cc-pv5z")

    # M = myMolecule("tests/6-QM7/1218.xyz","aug-cc-pv5z")
    # Gb = blockGrid(M,maxDist=4.5,minDist=0.4,gridSpacing=0.35)   
    # Vb = Gb.optimizeBasis()
    # Gb.printStats(Vb)
    # Gb.exportErrorVsDistance(Vb,mode=mode,pltLabel="aug-cc-pv5z")

    # # Gb.exportErrorVsDistance(Vs,mode=mode,pltLabel="S on B")

    # logging.info(f"Vs: {Vs.diagonal()}")
    # logging.info(f"Vb: {Vb.diagonal()}")
    # printStats(Gb,Vs)
    # printStats(Gs,Vb)
    plt.show()


def testPotentialIntegration():
    M = myMolecule("tests/6-QM7/1.xyz","def2-TZVP-c")
    Gs = sphericalGrid(M,minDist=0.05,maxDist=20.0,nRadial=200)   
    Vs = Gs.optimizeBasis()
    Gs.printStats(Vs)

    vpotError = 0.0 
    VPOT  = np.zeros(Gs.mol.ao_pot.shape[0])
    VBPOT = np.zeros(Gs.mol.ao_pot.shape[0])

    for i in range(Gs.mol.ao_pot.shape[0]):
        VP = np.sum(Gs.phi[:,i] * vpot(Gs.mol.geom,Gs.mol.elez,Gs.points) * Gs.phi[:,i]*Gs.weights)
        VB = np.sum(Gs.phi[:,i] * vBpot(Gs.phi, Vs.diagonal())            * Gs.phi[:,i] * Gs.weights)
        
        logging.info(f"[{i}] {Gs.mol.ao_pot[i][i]:4.2f} {VP:4.2f} {VB:4.2f}")
        VPOT[i] = VP
        VBPOT[i] = VB
            
    
    logging.info(f"ao_pot - VPOT: {np.sum(np.abs(Gs.mol.ao_pot.diagonal() - VPOT))}")
    logging.info(f"ao_pot - VBPOT: {np.sum(np.abs(Gs.mol.ao_pot.diagonal() - VBPOT))}")
    logging.info(f"VPOT - VBPOT: {np.sum(np.abs(VPOT - VBPOT))}")
    M.runPSI4("HF")
    
def plotPotentialOfAtom():
    M = myMolecule("tests/1-Oxygen/o.xyz","def2-TZVP")
    gridSpace = 0.01
    myCube = {"xmin" : 0.0,
            "xmax" : gridSpace,
            "ymin" : 0.0,
            "ymax" : gridSpace,
            "zmin" : -10.0,
            "zmax" : 10.0
           }
    Gb = blockGrid(M,cube=myCube,gridSpacing=gridSpace,minDist=0.01,maxDist=10.0)

    Gs = sphericalGrid(M,minDist=0.1,nRadial=200)

    Vs = Gs.optimizeBasis()
    Gs.printStats(Vs)

    logging.info(f"{Gb.points}")
    VPOT = vpot(Gb.mol.geom, Gb.mol.elez, Gb.points)

    N = np.sum(VPOT*gridSpace)

    VBPOT = vBpot(Gb.phi,Vs.diagonal())

    x = Gb.points[:,2]
    y = - 8/(np.abs(Gb.points[:,2]))

    plt.plot(Gb.points[:,2],VPOT)
    plt.plot(x,y,"--",lw=3)
    
    
    #plt.plot(Gb.points[:,2],Gb.phi[:,0])
    plt.plot(Gb.points[:,2],VBPOT)
    plt.vlines([-0.5,0.5],-5,5)
    Gs.mol.basisSet.print_detail_out()
    logging.info(np.sum(Gs.phi[:,0]*Gs.phi[:,0]*Gs.weights))
    logging.info(f"{Vs.diagonal()}")


    
    

    
    plt.show()
    
def plotPotentialOfAtomANC():

    def testBlockGridANC(ancPrecision):
        M = myMolecule("tests/1-Oxygen/o.xyz","def2-tzvpd-decon")
        gridSpace = 0.01
        myCube = {"xmin" : 0.0,
                "xmax" : gridSpace,
                "ymin" : 0.0,
                "ymax" : gridSpace,
                "zmin" : -10.0,
                "zmax" : 10.0
                }    

        prec = ancPrecision 
        Gb = blockGrid(M,cube=myCube,gridSpacing=gridSpace,minDist=0.01,maxDist=10.0)
        Vb = Gb.optimizeBasis(potentialType="anc",a=prec)

        Gb.printStats(Vb)


        VPOT = vpot(Gb.mol.geom, Gb.mol.elez, Gb.points)
        VPOTANC = vpotANC(Gb.mol.geom, Gb.mol.elez, Gb.points,a=prec)

        N = np.sum(VPOT*gridSpace)

        VBPOT = vBpot(Gb.phi,Vb.diagonal())

        x = Gb.points[:,2]
        y = - 8/(np.abs(Gb.points[:,2]))

        plt.plot(Gb.points[:,2],VPOT,label="vpot")
        plt.plot(Gb.points[:,2],VPOTANC,label="vpot-anc")
        plt.plot(Gb.points[:,2],VBPOT,label="VBPOT")
        plt.legend()
        plt.vlines([-0.5,0.5],-5,5)
        plt.xlim((-1.0,1.0))
        Gb.mol.basisSet.print_detail_out()
    
        plt.show()

    def testSphericalGridANC(ancPrecision):
        M = myMolecule("tests/1-Oxygen/o.xyz","def2-tzvpd-decon")
        gridSpace = 0.01
        myCube = {"xmin" : 0.0,
                "xmax" : gridSpace,
                "ymin" : 0.0,
                "ymax" : gridSpace,
                "zmin" : -10.0,
                "zmax" : 10.0
                }    
        Gb = blockGrid(M,cube=myCube,gridSpacing=gridSpace,minDist=0.01,maxDist=10.0)


        prec = ancPrecision 
        Gs = sphericalGrid(M,minDist=0.01,maxDist=10.0)
        Vs = Gs.optimizeBasis(potentialType="anc",a=prec)

        Gs.printStats(Vs)


        VPOT = vpot(Gb.mol.geom, Gb.mol.elez, Gb.points)
        VPOTANC = vpotANC(Gb.mol.geom, Gb.mol.elez, Gb.points,a=prec)


        VBPOT = vBpot(Gb.phi,Vs.diagonal())

        plt.plot(Gb.points[:,2],VPOT,label="vpot")
        plt.plot(Gb.points[:,2],VPOTANC,label="vpot-anc")
        plt.plot(Gb.points[:,2],VBPOT,label="VBPOT")
        plt.legend()
        plt.vlines([-0.5,0.5],-5,5)
        plt.xlim((-1.0,1.0))
        Gb.mol.basisSet.print_detail_out()

        plt.show()
        Gs.exportErrorVsDistance(Vs,plotPot=True)
    


    
    """
    testBlockGridANC(1)
    testBlockGridANC(2)
    testBlockGridANC(4)
    """
    testSphericalGridANC(1)
    testSphericalGridANC(2)
    testSphericalGridANC(4)


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
    # alrichVSDunning()



    #testPotentialIntegration()

    #plotPotentialOfAtomANC()
    testANCMolecule(1)
    testANCMolecule(2)


    # Gb = blockGrid(M,maxDist=3.5)  
    # Vb = Gb.optimizeBasis()
    # printStats(Gb,Vb)
    # distanceVSerror(Gb,Vb)


    

 
