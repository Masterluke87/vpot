from vpot.calc.grids import sphericalGrid, blockGrid
from vpot.calc import myMolecule
from vpot.calc.potential import vpot,vBpot, vpotANC
from vpot.calc import DFTGroundState
from matplotlib import pyplot as plt
import psi4
import numpy as np
import logging,time


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

def testIntegration():

    nSphere = 302
    nRadial = 120

    """
    for maxDist in [5.0,10.0,20.0]:
        for nSphere in [590]: #,770,974,1202,1454,1730,2030]:
            for nRadial in [300]:
                M = myMolecule("tests/6-QM7/1.xyz","def2-TZVP")
                Gs = sphericalGrid(M,minDist=0.0,maxDist=maxDist,nRadial=nRadial,nSphere=nSphere,pruningScheme="None")   
                Vs = Gs.optimizeBasis()
                Gs.printStats(Vs)
            
                vpotAnalytic = M.ao_pot
                vpotNumeric  = np.einsum("ji,j,jk,j->ik",Gs.phi,vpot(Gs.mol.geom,Gs.mol.elez,Gs.points),Gs.phi,Gs.weights)

                logging.info(f"MaxDist: {maxDist} Ns: {nSphere} nR: {nRadial} Error: {np.linalg.norm(vpotAnalytic-vpotNumeric)} Max: {np.max(np.abs(vpotAnalytic-vpotNumeric))}")

    """

    gridSpacing = 0.5
    for gridSpacing in [0.2,0.1,0.075]:
        M = myMolecule("tests/6-QM7/1.xyz","def2-TZVP")
        Gb = blockGrid(M,gridSpacing,0.01,4.5)
        vpotAnalytic = M.ao_pot
        vpotNumeric  = np.einsum("ji,j,jk->ik",Gb.phi,vpot(Gb.mol.geom,Gb.mol.elez,Gb.points),Gb.phi)*gridSpacing**3
        logging.info(f"GridSpacing: {gridSpacing} Error: {np.linalg.norm(vpotAnalytic-vpotNumeric)} Max: {np.max(np.abs(vpotAnalytic-vpotNumeric))}")

    return vpotAnalytic,vpotNumeric


def testMyDFT():
    """
    Run that and --> grep "Functional" benchmark.log
    """
    psi4.set_memory("12Gb")
    psi4.set_num_threads(8)
    

    M = myMolecule("tests/6-QM7/1.xyz","def2-TZVP")
    for func in ["HF","PBE","B3LYP","TPSS"]:
        MyDFT,_ = DFTGroundState(M,func)
        EPsi,_ = M.runPSI4(func)
        logging.info(f"Functional: {func} Psi4: {EPsi:16.12f}, MyDFT: {MyDFT:16.12f} Diff: {EPsi-MyDFT:16.12f}")

    M = myMolecule("tests/6-QM7/1218.xyz","def2-TZVP")
    for func in ["HF","PBE","B3LYP","TPSS"]:
        MyDFT,_ = DFTGroundState(M,func)
        EPsi,_ = M.runPSI4(func)
        logging.info(f"Functional: {func} Psi4: {EPsi:16.12f}, MyDFT: {MyDFT:16.12f} Diff: {EPsi-MyDFT:16.12f}")


def testPotentialDFT():
    """
    1. Push a numerically evaluated AO POT into the DFT algo
    2. Add noise to AO-POT -> works
    """
    summary = open("testPotentialDFT.txt","w")
    summary.write("{:10s} {:16s} {:16s} {:10s} {:10s} {:10s}\n".format("Noise","PSI4","MyDFT(NumPot)","DIFF","potError","maxPotErr"))

    psi4.set_memory("12Gb")
    psi4.set_num_threads(8)
    

    for noise in [0.0,1.0E-6,1.0E-4,1.0E-3,1.0E-1]:
        M = myMolecule("tests/6-QM7/1.xyz","def2-TZVP")
        Gs = sphericalGrid(M,minDist=0.0,maxDist=10.0,nRadial=300,nSphere=590,pruningScheme="None")
        VPOT = np.einsum("ji,j,jk,j->ik",Gs.phi,vpot(Gs.mol.geom,Gs.mol.elez,Gs.points),Gs.phi,Gs.weights)

        N     = np.random.normal(0.0,noise,(M.basisSet.nbf(),M.basisSet.nbf()))
        VPOT += (np.tril(N) + np.triu(N.T, 1))

        MyDFT,_ = DFTGroundState(M,"PBE",AOPOT=VPOT)
        EPsi,_ = M.runPSI4("PBE")

        summary.write(f"{noise:.3E} {EPsi:16.12f} {MyDFT:16.12f} {EPsi-MyDFT:.3e} {np.linalg.norm(M.ao_pot-VPOT):.3e} {np.max(np.abs(M.ao_pot-VPOT)):.3e}\n")
        

def testANCvsExactPotential():
    psi4.set_memory("24Gb")
    psi4.set_num_threads(8)
    summary = open("tests/testANCvsExactPotential.txt","w")
    summary.write("{:^16s} {:^10s} {:^16s} {:^16s} {:^16s} {:^16s} {:^16s} {:^16s} {:^16s} {:^16s} {:^16s}\n".format("Basis","a","E_ANC_exact",
    "E_ANC_basis","E_exact","basErr","basMax","ancErr","ancMax","potErr","potMax"))

    basisSet = "def2-SVP-decon"

    for prec in [1,2,3,4,5,6]:
        for basisSet in ["def2-SVP","def2-TZVP","def2-QZVP","def2-SVP-decon","def2-TZVP-decon","def2-QZVP-decon"]:
            M = myMolecule("tests/6-QM7/1.xyz", basisSet)
            Gs = sphericalGrid(M,minDist=0.0,maxDist=10.0,nRadial=300,nSphere=590,pruningScheme="None")
            VMat = Gs.optimizeBasis(potentialType="anc",a=prec)
            Gs.printStats(VMat)
            

            VANCexact = np.einsum("ji,j,jk,j->ik",Gs.phi,vpotANC(Gs.mol.geom,Gs.mol.elez,Gs.points,prec),Gs.phi,Gs.weights)
            VANCbasis = np.einsum("ji,j,jk,j->ik",Gs.phi,vBpot(Gs.phi,VMat.diagonal()),Gs.phi,Gs.weights)

            MyDFTexact,_ = DFTGroundState(M,"PBE",AOPOT=VANCexact)
            MyDFTbasis,_ = DFTGroundState(M,"PBE",AOPOT=VANCbasis)
            EPsi,_ = M.runPSI4("PBE")

            basErr = np.linalg.norm(VANCexact-VANCbasis)
            basMax = np.max(np.abs(VANCexact-VANCbasis))
            ancErr = np.linalg.norm(VANCexact-M.ao_pot)
            ancMax = np.max(np.abs(VANCexact-M.ao_pot))
            potErr = np.linalg.norm(VANCbasis-M.ao_pot)
            potMax = np.max(np.abs(VANCbasis-M.ao_pot))
            

            summary.write(f"{basisSet:^16s} {prec:^10d} {MyDFTexact:^16.6f} {MyDFTbasis:^16.6f} {EPsi:^16.6f} {basErr:^16.6f} {basMax:^16.6f} {ancErr:^16.6f} {ancMax:^16.6f} {potErr:^16.6f} {potMax:^16.6f}\n")
            summary.flush()

    """  
    summary = open("testANCvsExactPotential.txt","w")
    summary.write("{:10s} {:16s} {:16s} {:10s} {:10s} {:10s}\n".format("a","PSI4","MyDFT(NumPot)","DIFF","potError","maxPotErr"))


    for prec in [2,4,6,8,10,12]:
        M = myMolecule("tests/6-QM7/1.xyz","def2-TZVP")
        Gs = sphericalGrid(M,minDist=0.0,maxDist=10.0,nRadial=300,nSphere=590,pruningScheme="None")

        VANCexact = np.einsum("ji,j,jk,j->ik",Gs.phi,vpotANC(Gs.mol.geom,Gs.mol.elez,Gs.points,prec),Gs.phi,Gs.weights)
        MyDFT,_ = DFTGroundState(M,"PBE",AOPOT=VANC)



        EPsi,_ = M.runPSI4("PBE")

        summary.write(f"{prec:10d} {EPsi:16.12f} {MyDFT:16.12f} {EPsi-MyDFT:.3e} {np.linalg.norm(M.ao_pot-VANCexact):.3e} {np.max(np.abs(M.ao_pot-VANCexact)):.3e}\n")
    """


def testSphericalAtomicGrid():
    M = myMolecule("tests/CH3Cl.xyz","def2-TZVP")
    Gas = sphericalAtomicGrid(M,"Cl",minDist=0.0,maxDist=1.2,nRadial=300,nSphere=590,pruningScheme="None") 
    Vas = Gas.optimizeBasis(potentialType="anc",a=2)
    Gas.printStats(Vas)
    Gas.exportErrorVsDistance(Vas)
    
    
def testSpericalAtomicGridCoeff(atomType="C"):
    from scipy.optimize import minimize
    from vpot.calc.grids import sphericalAtomicGrid
    from psi4.driver import qcdb
    
    def getCoeffsAndExps(basisDict,atomType):
        out = []
        for i in basisDict["shell_map"]:
            if i[0] == atomType:
                for j in i[2:]:
                    for k in j[1:]:
                        out.append(k)
        return(out)

    def optmizeBasis(x0,Gs,atomType):
        optmizeBasis.counter+=1
        xmod = x0
        counter = 0
        a,newBasis = qcdb.BasisSet.pyconstruct(Gs.mol.psi4Mol.to_dict(),'BASIS', 
                                               "def2-SVP",fitrole='ORBITAL',
                                              other=None,return_dict=True,return_atomlist=False)

        for i in newBasis["shell_map"]:
            del i[2:]

        for c,i in enumerate(Gs.mol.basisDict["shell_map"]):
            if i[0] == atomType:
                for j in i[2:]:
                    newBas = []
                    newBas.append(j[0])
                    for k in j[1:]:
                        newBas.append((xmod[counter],xmod[counter+1]))
                        counter+=2
                    newBasis["shell_map"][c] += [newBas]

        Gs.mol.setBasisDict(newBasis,quiet=True)
        logging.info(getCoeffsAndExps(M.basisDict,atomType))

        Gs.projectBasis()
        Vs = Gs.optimizeBasis(potentialType="anc",a=2)
        Error = Gs.getMSError(Vs)

        logging.info(f"ERROR FROM AUGMENTBASIS: {Error}")
        logging.info(getCoeffsAndExps(M.basisDict,atomType))

        if (optmizeBasis.counter %1)==0:
            print(f"Count: {optmizeBasis.counter}, Error {Error}")
        return Error

    
    atomType = atomType.upper()
    M = myMolecule("tests/CH3Cl.xyz","")
    Gas = sphericalAtomicGrid(M,atomType,minDist=0.0,maxDist=1.2,nRadial=300,nSphere=590,pruningScheme="None") 
    Vas = Gas.optimizeBasis(potentialType="anc",a=2)
    Gas.printStats(Vas)
    Gas.exportErrorVsDistance(Vas)
    
    
    xinit = np.array(getCoeffsAndExps(M.basisDict,atomType))
    xinit = xinit.flatten()
    
    bounds = [(0.1,100000.0) if (c%2==0) else (None,None) for c,x in enumerate(xinit) ]

    optmizeBasis.counter=0
    result = minimize(optmizeBasis,xinit,args=(Gas,atomType),bounds=bounds)
    
    print(result)

    return result,M
    
    
    

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

    #testIntegration()

    #testPotentialIntegration()

    #plotPotentialOfAtomANC()
    #testANCMolecule(1)
    #testANCMolecule(2)
    #testANCvsExactPotential()

    #testMyDFT()
    #testPotentialDFT()
    # Gb = blockGrid(M,maxDist=3.5)  
    # Vb = Gb.optimizeBasis()
    # printStats(Gb,Vb)
    # distanceVSerror(Gb,Vb)


    

 
