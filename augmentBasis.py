from vpot.calc import myMolecule, sphericalGrid, blockGrid, pointGrid
from vpot.calc.potential import vpot,vBpot, vpotANC
import numpy as np
from psi4.driver import qcdb
import logging,time
from scipy.optimize import minimize
from matplotlib import pylab as plt

plt.style.use('dark_background')


def getCoeffsAndExps(basisDict):
    out = []
    for i in basisDict["shell_map"]:
        for j in i[2:]:
            for k in j[1:]:
                out.append(k)
    return(out)



def optmizeBasis(x0,M):
    optmizeBasis.counter+=1
    xmod = x0
    counter = 0
    a,newBasis = qcdb.BasisSet.pyconstruct(M.psi4Mol.to_dict(),'BASIS', 
                                           "def2-SVP",fitrole='ORBITAL',
                                          other=None,return_dict=True,return_atomlist=False)

    for i in newBasis["shell_map"]:
        del i[2:]
        
    for c,i in enumerate(M.basisDict["shell_map"]):
        for j in i[2:]:
            newBas = []
            newBas.append(j[0])
            for k in j[1:]:
                newBas.append((xmod[counter],xmod[counter+1]))
                counter+=2
            newBasis["shell_map"][c] += [newBas]
    
    M.setBasisDict(newBasis,quiet=True)
    logging.info(getCoeffsAndExps(M.basisDict))
    
    Gs=sphericalGrid(M,minDist=0.0,maxDist=1.2,nRadial=300,nSphere=590,pruningScheme="None") 
    Vs = Gs.optimizeBasis(potentialType="anc",a=2)
    Error = Gs.getMSError(Vs)

    logging.info(f"ERROR FROM AUGMENTBASIS: {Error}")
    logging.info(getCoeffsAndExps(M.basisDict))
    
    if (optmizeBasis.counter %100)==0:
        print(f"Count: {optmizeBasis.counter}, Error {Error}")
    return Error


if __name__ == "__main__":
    
    logging.basicConfig(filename='augmentBasis.log', level=logging.INFO,filemode="w")

    M = myMolecule("./tests/CH3Cl.xyz","")
    Gs=sphericalGrid(M,minDist=0.0,maxDist=1.2,nRadial=300,nSphere=590,pruningScheme="None") 

    Vs = Gs.optimizeBasis(potentialType="anc",a=2)
    Gs.printStats(Vs,output="print")
    Gs.exportErrorVsDistance(Vs)

    counter = 0
    for i in M.basisDict["shell_map"]:
        for j in i[2:]:
            for k in j[1:]:
                counter += 1 
            
    print(f"Basis set has {counter} degrees of freedom")

    xinit = np.array(getCoeffsAndExps(M.basisDict))
    xinit[:,0] = xinit[:,0]
    xinit = xinit.flatten()
    
    bounds = [(0.1,100000.0) if (c%2==0) else (None,None) for c,x in enumerate(xinit) ]

    optmizeBasis.counter=0
    
    result = minimize(optmizeBasis,xinit,args=(M),bounds=bounds)
    print(result)
