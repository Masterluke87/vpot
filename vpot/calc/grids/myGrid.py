import psi4
import logging
import numpy as np
from matplotlib import pyplot as plt
import time
from vpot.calc.mol import myMolecule
from vpot.calc.potential import vpot,vBpot,vpotANC
from ase.data.colors import jmol_colors
from ase.data import chemical_symbols


class myGrid(object):
    def __init__(self, mol : myMolecule):
        self.mol = mol
        self.gridInfo = { }

        #This should be an np.array, which hold the xyz coordinates
        #of the grid points. Shape (N,3)
        self.points = None
        #This guy contains the values of the K basis functions at each point
        #Shape (N,K)
        self.phi    = None

        #We save also the potential we would like to fit
        #Until now this is either the true potential or the 
        #analytic norm conserving potential

        self.vpot  = None

    def updateMol(self,newMol):
        self.mol = newMol
        self._projectBasis()

    def _projectBasis(self):
        """
        This function uses the self.mol object and the basis set in it to project the basis functions onto
        the grid point
        """

        basis_extents = psi4.core.BasisExtents(self.mol.basisSet, 0.0)
        
        xs = psi4.core.Vector.from_array(self.points[:,0])
        ys = psi4.core.Vector.from_array(self.points[:,1])
        zs = psi4.core.Vector.from_array(self.points[:,2])
        ws = psi4.core.Vector.from_array(self.weights)

        blockopoints = psi4.core.BlockOPoints(xs, ys, zs, ws, basis_extents)
        max_points = blockopoints.npoints()
        max_functions = self.mol.basisSet.nbf()
        funcs = psi4.core.BasisFunctions(self.mol.basisSet, max_points, max_functions)
        
        funcs.compute_functions(blockopoints)
        self.phi = np.array(funcs.basis_values()['PHI'])

    def getMSError(self,V):
        return np.mean(np.square((vBpot(self.phi,V.diagonal()) - self.vpot )))

    def getMaxError(self,V):
        return np.max(np.square((vBpot(self.phi,V.diagonal()) - self.vpot )))

    def optimizeBasisWeights(self,expPref=[1.0,1.0],potentialType:str="anc",a:int=2):
        start = time.perf_counter()
        
        """
        we try to solve
        A.X = B
        phi.C = Vpot
        
        
        we try to use the the weights from 
        """
        
        Dist    = np.array([np.linalg.norm(self.points[:,:3] - x,axis=1) for x in self.mol.geom]).transpose()
        Dmin    = np.min(Dist,axis=1)
        
        #W = np.sqrt(0.5*(1-np.exp(-(Dmin/2)**5))+0.5)
        W = expPref[1]*np.exp(-(Dmin*expPref[0])**2)
        if (potentialType == "exact"):
            #W  = np.sqrt(self.weights)
            Aw = (self.phi.T*W).T
            Bw = -vpot(self.mol.geom,self.mol.elez,self.points) * W
            VMat, resis,_,_ = np.linalg.lstsq(Aw, Bw,rcond=-1)
            self.vpot = -vpot(self.mol.geom, self.mol.elez,self.points)

        elif (potentialType == "anc"):
            #W  = np.sqrt(self.weights)
            Aw =  (self.phi.T*W).T
            Bw = -vpotANC(self.mol.geom,self.mol.elez,self.points , a) *W
            VMat, resis,_,_ = np.linalg.lstsq(Aw,Bw,rcond=-1)
            self.vpot = vpotANC(self.mol.geom, self.mol.elez, self.points , a)

        else:
            raise(f"Unknown potential {potentialType}")

        mints = psi4.core.MintsHelper(self.mol.basisSet)
        X =mints.ao_overlap().np
        for c1,i in enumerate(VMat):
            X[c1,c1] = i

        logging.info(f"resis : {resis} ")
        logging.info(f"optimizeBasis time: {time.perf_counter()-start:10.2f} s")
        return X



    def optimizeBasis(self,potentialType = "exact",a:int = 2):
        start = time.perf_counter()

        if (type(potentialType)==str):
            if (potentialType == "exact"):
                VMat, resis,_,_ = np.linalg.lstsq(self.phi, -vpot(self.mol.geom,
                                                      self.mol.elez,
                                                      self.points),rcond=-1)
                self.vpot = -vpot(self.mol.geom, self.mol.elez,self.points)

            elif (potentialType == "anc"):
                VMat, resis,_,_ = np.linalg.lstsq(self.phi, -vpotANC(self.mol.geom,
                                                           self.mol.elez,
                                                           self.points , a),rcond=-1)
                self.vpot = vpotANC(self.mol.geom, self.mol.elez, self.points , a)

        elif (type(potentialType)==np.ndarray):
            VMat, resis,_,_ = np.linalg.lstsq(self.phi, -potentialType,rcond=-1)
            self.vpot = potentialType
        else:
            raise TypeError("Unknown Potential type")


        mints = psi4.core.MintsHelper(self.mol.basisSet)
        X =mints.ao_overlap().np
        for c1,i in enumerate(VMat):
            X[c1,c1] = i

        logging.info(f"resis : {resis} ")
        logging.info(f"optimizeBasis time: {time.perf_counter()-start:10.2f} s")
        return X

    def exportErrorVsDistance(self,V,mode :str ="min", pltLabel: str= "", path : str ="",plotPot:bool=False):
        
        Error = vBpot(self.phi,V.diagonal()) - self.vpot
        if mode=="com":
            Dists = np.array([np.linalg.norm(self.points - self.mol.com,axis=1)]).transpose()
            plt.plot(Dists,Error,"o",markersize=0.6,label=pltLabel)
        if mode=="min":
            Dists = np.min(np.array([np.linalg.norm(self.points - x,axis=1) for x in self.mol.geom]).transpose(),axis=1)
            DminARG = np.argmin(np.array([np.linalg.norm(self.points - x,axis=1) for x in self.mol.geom]).transpose(),axis=1)
            DminZ = np.array([int(self.mol.elez[x]) for x in DminARG])
            for i in set(DminZ):
                idx = np.where(DminZ==i)
                plt.plot(Dists[idx],Error[idx],"o",color=jmol_colors[i],markersize=0.6,label=chemical_symbols[i])
            plt.legend()
        
        plt.ylabel("Error wrt. exact potential [a.u.]")
        plt.xlabel("Distance to neareast nuclei [a.u.]")
      
        if plotPot:
            ax2 = plt.gca().twinx()
            ax2.plot(Dists,self.vpot,"o",color="red",markersize=0.6,label="Potential")           
            ax2.set_ylabel("Potential [a.u.]")
            plt.legend()
            
        if path:
            plt.savefig(f"{path}",dpi=300)
            plt.clf()
        else:
            plt.show()
        
    def printStats(self, V, output="logger"):
        if output=="logger":
            logging.info(f"GridInfo : {self.gridInfo} ")
            logging.info(f"residue  : {np.sum(np.square((vBpot(self.phi,V.diagonal())  - self.vpot )))}")
            logging.info(f"MeanError: {np.mean(np.square((vBpot(self.phi,V.diagonal()) - self.vpot )))}")
            logging.info(f"MaxError : {np.max(np.square((vBpot(self.phi,V.diagonal())  - self.vpot )))}")
            logging.info(f"MinError : {np.min(np.square((vBpot(self.phi,V.diagonal())  - self.vpot )))}")
        elif output=="print":
            print(f"GridInfo : {self.gridInfo} ")
            print(f"residue  : {np.sum(np.square((vBpot(self.phi,V.diagonal())  - self.vpot )))}")
            print(f"MeanError: {np.mean(np.square((vBpot(self.phi,V.diagonal()) - self.vpot )))}")
            print(f"MaxError : {np.max(np.square((vBpot(self.phi,V.diagonal())  - self.vpot )))}")
            print(f"MinError : {np.min(np.square((vBpot(self.phi,V.diagonal())  - self.vpot )))}")
        else:
            pass



