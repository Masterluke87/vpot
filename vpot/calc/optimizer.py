from vpot.calc import myMolecule
from vpot.calc.grids import sphericalGrid, blockGrid, pointGrid,sphericalAtomicGrid,blockAtomicGrid,sphericalIndexGrid
from vpot.calc.potential import vpot,vBpot, vpotANC
from vpot.calc.dft import DFTGroundState, DFTGroundStateRKS, getSADGuess
from matplotlib import pyplot as plt
import numpy as np
import os
from matplotlib.colors import SymLogNorm,CenteredNorm
from scipy.spatial import KDTree


def generateFullPotential(M,Gs,Vsoft):
    kdt = KDTree(M.geom)
    nndist,nnidx = kdt.query(Gs.points)
    Vhard = Vsoft.copy()

    for i in range(len(Gs.mol.geom)):
        idx = nnidx==i
        Vhard [idx] +=  vpot([Gs.mol.geom[i]],[Gs.mol.elez[i]],Gs.points[idx])
    return Vhard

def generateSoftPotential(M,Gs):
    from scipy.spatial import KDTree
    kdt = KDTree(M.geom)
    nndist,nnidx = kdt.query(Gs.points)

    VpotTOT = np.zeros(Gs.points.shape[0])

    for i in range(len(Gs.mol.geom)):
        center = [c for c,x in enumerate(Gs.mol.geom) if c!=i]
        #get all point as close a 1.2 Angstrom around center i
        idx = nnidx==i
        VpotTOT[idx] =  vpot(Gs.mol.geom[center],Gs.mol.elez[center],Gs.points[idx])
    return VpotTOT


class threeStepOptimizer(object):
    
    def unwrapMatrix(self, M):
        dim = sum([len(x) for x in M])
        
        newMat = np.zeros((dim,dim))
        
        counter = 0
        for j in M:
            for i in j.diagonal():
                newMat[counter][counter] = i
                counter+=1
                
        return newMat
        
    
    def plotAugmentBasis(self,idx1=0,idx2=1):
        M = myMolecule(self.pathToMolecule,"",labelAtoms=True)
        Gb =  sphericalGrid(M,minDist=0.0,maxDist=4.0,nSphere=nSphere,nRadial=nRadial,pruningScheme="None") 
        Vb = Gb.optimizeBasis(potentialType="anc",a=2)
        
        for c,i in enumerate(Vb.diagonal()):
            Vb[c][c] = 0.0 
  
        counter = 0
        for c2,j in enumerate(self.Va):
            for i in j.diagonal():
                Vb[counter][counter] = i
                counter+=1

        Gb.printStats(Vb,output="print")
        Gb.exportErrorVsDistance(Vb)

        P1 = M.geom[idx1]
        P2 = M.geom[idx2]

        v = P2-P1
        r = np.arange(-0.5,np.linalg.norm(v)+0.5,0.01)

        L = np.array([P1 + i*v/np.linalg.norm(v) for i in r])

        VPOT = vpot(Gb.mol.geom,Gb.mol.elez,L)
        VANC = vpotANC(Gb.mol.geom,Gb.mol.elez,L,self.prec)
        GL2 = pointGrid(M,L)
        VBAS = vBpot(GL2.phi,Vb.diagonal())

        R = VANC - VBAS

        plt.plot(r,R,label="Residue")
        plt.plot(r,VPOT,label="VPOT")
        plt.plot(r,VANC,label="VANC",color="green")
        plt.plot(r,VBAS,label="VBAS",color="red")
        plt.legend()
        plt.ylim(-300,20)


        
    
    def optimizeAugmentBasis(self):
        Va = [] 
        M = myMolecule(self.pathToMolecule,"",labelAtoms=True)
        
        for atomIdx in range(len(M.geom)):
            M = myMolecule(self.pathToMolecule,"",labelAtoms=True)
            M.keepAugmentBasisForIndex(atomIdx)
            M.setBasisDict(orbitalDict=None,augmentDict=M.getAugmentDict(),quiet=True)

            Ga =  sphericalIndexGrid(M,atomIdx,minDist=0.0,maxDist=1.0,nSphere=self.nSphere,nRadial=self.nRadial,pruningScheme="None") 
            Va.append(Ga.optimizeBasis(potentialType="anc",a=2))

        self.Va = Va
        return Va
    
    def plotOrbitalBasis(self,idx1=0,idx2=1):
        Ma = myMolecule(self.pathToMolecule,"",augmentBasis=True,labelAtoms=True)
        Mo = myMolecule(self.pathToMolecule,self.orbitalBasisSet,augmentBasis=False,labelAtoms=True)
        
        Ga = sphericalGrid(Ma,minDist=0.0,maxDist=4.0,nSphere=self.nSphere,nRadial=self.nRadial,pruningScheme="None") 
        Go = sphericalGrid(Mo,minDist=0.0,maxDist=4.0,nSphere=self.nSphere,nRadial=self.nRadial,pruningScheme="None") 
        
        VANC = vpotANC(Ga.mol.geom,Ga.mol.elez,Ga.points,self.prec)
        VBAS = vBpot(Ga.phi,self.unwrapMatrix(self.Va).diagonal())
        Ra = VANC - VBAS

        Vo = Go.optimizeBasis(potentialType=Ra)
        Go.printStats(Vo,output="print")
        Go.exportErrorVsDistance(Vo)

        P1 = Ma.geom[idx1]
        P2 = Ma.geom[idx2]

        v = P2-P1
        r = np.arange(-0.5,np.linalg.norm(v)+0.5,0.01)

        L = np.array([P1 + i*v/np.linalg.norm(v) for i in r])

        VANC = vpotANC(Ga.mol.geom,Ga.mol.elez,L,self.prec)
        GL2 = pointGrid(Ma,L)
        GL3 = pointGrid(Mo,L)

        VBAS = vBpot(GL2.phi,self.unwrapMatrix(self.Va).diagonal())
        RBAS = vBpot(GL3.phi,Vo.diagonal())
        R = VANC - VBAS



        plt.plot(r,VBAS,label="VBAS",color="red")
        plt.plot(r,VANC,label="VANC",color="green")
        plt.plot(r,R,label="R",color="blue")
        plt.plot(r,RBAS,label="RBAS",color="yellow")


        plt.legend()
        plt.ylim((-200,40))
        plt.show()

    
    def optimizeOrbitalBasis(self):
        Ma   = myMolecule(self.pathToMolecule,self.orbitalBasisSet,augmentBasis=False,labelAtoms=True)
        Gbb  = sphericalGrid(Ma,minDist=0.0,maxDist=4.0,nSphere=self.nSphere,nRadial=self.nRadial,pruningScheme="None") 
        

        Vbb= Gbb.optimizeBasis(potentialType=self.residue)
        
        
        V = [] 
        for c,i in enumerate(Ma.geom):
            Vl = []
            for c2,j in enumerate(Vbb.diagonal()):
                if Ma.basisSet.function_to_center(c2) == c:
                    Vl.append(j)
            V.append(Vl)
            
            
        self.Vo = V
            
        return V
        
    def plotTotalBasis(self,idx1=0,idx2=1):
        Mt = myMolecule(self.pathToMolecule,self.orbitalBasisSet,augmentBasis=True,labelAtoms=True)

        
        P1 = Mt.geom[idx1]
        P2 = Mt.geom[idx2]

        v = P2-P1
        r = np.arange(-0.5,np.linalg.norm(v)+0.5,0.01)

        L = np.array([P1 + i*v/np.linalg.norm(v) for i in r])

        GLt = pointGrid(Mt,L)
        VBAS = vBpot(GLt.phi,self.Vt.diagonal())
        VANC = vpotANC(GLt.mol.geom,GLt.mol.elez,L,self.prec)
        VPOT = vpot(GLt.mol.geom,GLt.mol.elez,L)

        plt.plot(r,VBAS,label="VBAS",color="red")
        plt.plot(r,VANC,label="VANC",color="green")
        plt.plot(r,VPOT,label="VPOT",color="blue")




        plt.legend()
        plt.ylim((1.1*np.min(VANC),10))
        plt.show()
        
    def optimizeTotalBasis(self):
        Mt = myMolecule(self.pathToMolecule,self.orbitalBasisSet,augmentBasis=True,labelAtoms=True)
        Gt = sphericalGrid(Mt,minDist=0.0,maxDist=4.0,nSphere=self.nSphere,nRadial=self.nRadial,pruningScheme="None") 
        Vt = Gt.optimizeBasis(potentialType="anc",a=2)
        
        for c,i in enumerate(Vt.diagonal()):
            Vt[c][c] = 0.0 

        counter = 0
        for c,i in enumerate(Mt.geom):
            for j in self.Va[c].diagonal():
                Vt[counter][counter] = j
                counter +=1
            for j in self.Vo[c]:
                Vt[counter][counter] = j 
                counter +=1
                
        self.Vt = Vt

    def calculateResidue(self,Va):
        M  = myMolecule(self.pathToMolecule,"",labelAtoms=True)
        Gb =  sphericalGrid(M,minDist=0.0,maxDist=4.0,nSphere=self.nSphere,nRadial=self.nRadial,pruningScheme="None") 
        
        
        VANC = vpotANC(Gb.mol.geom,Gb.mol.elez,Gb.points,self.prec)
        VBAS = vBpot(Gb.phi,self.unwrapMatrix(self.Va).diagonal())
        Ra = VANC - VBAS
                
        self.residue = Ra
        
        return Ra
    
    def plotAllNeigborsTotalBasis(self):
        M = myMolecule(self.pathToMolecule,self.orbitalBasisSet,augmentBasis=True,labelAtoms=True)

        for i,j in M.getCloseNeighbors():
            self.plotTotalBasis(i,j)



    def __init__(self,pathToMolecule,orbitalBasisSet):
        
        #Start with the augmented basis optimization:
        self.nSphere = 590
        self.nRadial = 300
        self.prec=2
        
        self.pathToMolecule = pathToMolecule
        self.orbitalBasisSet = orbitalBasisSet
        self.residue = None
        
        self.Va = None
        self.Vo = None
        self.Vt = None
        
        self.optimizeAugmentBasis()
        self.calculateResidue(self.Va)
        self.optimizeOrbitalBasis()
        self.optimizeTotalBasis()


class simpleOptimizer(object):
    def __saveInputQuantities(self):
        M = myMolecule(self.pathToMolecule,self.orbitalBasisSet,augmentBasis=True,labelAtoms=False)

        info = {"Path" : self.pathToMolecule,
                "BasisSet" : self.orbitalBasisSet,
                "Functional" : self.functional,
                "nSphere" : self.nSphere,
                "nRadial" : self.nRadial,
                "minDist" : 0.0,
                "maxDist" : 4.0,
                "fitError" : self.residue}    

        np.savez_compressed(f"{self.path}/input.npz",
                C_V_ANC=self.C_V_ANC,
                V_ANC_B=self.V_ANC_B,
                V_EXT=self.V_EXT,
                NEL = M.nElectrons,
                INFO=info)
        
    def __plotPMatrix(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        im = ax.imshow(self.P_ANC_B,cmap="seismic",
                       norm=SymLogNorm(linthresh=1.0,vmin=-np.max(np.abs(self.P_ANC_B)),
                       vmax=np.max(np.abs(self.P_ANC_B))))
        fig.colorbar(im)
        ax.set_title(r"$C^{v,anc}_{\mu \nu}$")
        ax.set_ylabel(r"$ \mu\ \mathrm{index}$")
        ax.set_xlabel(r"$ \nu\ \mathrm{index}$")
        fig.tight_layout()
        fig.savefig(f"{self.path}/P_anc_b.png",dpi=300)

    def __plotCMatrix(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        im = ax.imshow(self.C_V_ANC,cmap="seismic",
                       norm=SymLogNorm(linthresh=1.0,vmin=-np.max(np.abs(self.C_V_ANC)),
                       vmax=np.max(np.abs(self.C_V_ANC))))
        fig.colorbar(im)
        ax.set_title(r"$C^{v,anc}_{\mu \nu}$")
        ax.set_ylabel(r"$ \mu\ \mathrm{index}$")
        ax.set_xlabel(r"$ \nu\ \mathrm{index}$")
        fig.tight_layout()
        fig.savefig(f"{self.path}/C_v_anc.png",dpi=300)

    def __plotTotalBasis(self,idx1=0,idx2=1):
        Mt = myMolecule(self.pathToMolecule,self.orbitalBasisSet,augmentBasis=True,labelAtoms=True)
        P1 = Mt.geom[idx1]
        P2 = Mt.geom[idx2]

        v = P2-P1
        r = [x for x in np.arange(-0.5,np.linalg.norm(v)+0.5,0.02) if (abs(x)>0.01) and abs(x-np.linalg.norm(v))>0.01]

        L = np.array([P1 + i*v/np.linalg.norm(v) for i in r])

        GLt = pointGrid(Mt,L)
        VBAS = vBpot(GLt.phi,self.C_V_ANC.diagonal())
        VANC = vpotANC(GLt.mol.geom,GLt.mol.elez,L,self.prec)
        VPOT = vpot(GLt.mol.geom,GLt.mol.elez,L)

        plt.plot(r,VBAS,label=r"$v_{anc,B}$",color="red")
        plt.plot(r,VANC,label=r"$v_{anc}$",color="green")
        plt.plot(r,VPOT,label=r"$v_{ext}$",color="blue")

        plt.legend()
        plt.ylabel("External Potential [a.u.]")
        plt.xlabel("Distance [a.u.]")

        #plt.ylim((1.1*np.min(VANC),10))
        plt.savefig(f"{self.path}/bond_{idx1}_{idx2}.png",dpi=300)
        plt.clf()

    def __plotAllNeigborsTotalBasis(self):
        M = myMolecule(self.pathToMolecule,self.orbitalBasisSet,augmentBasis=True,labelAtoms=True)

        for i,j in M.getCloseNeighbors(thresh=1.8):
            self.__plotTotalBasis(i,j)

    def __optimizeTotalBasis(self):
        Mt = myMolecule(self.pathToMolecule,self.orbitalBasisSet,augmentBasis=True,labelAtoms=True)
        Gs = sphericalGrid(Mt,minDist=0.0,maxDist=4.0,nSphere=self.nSphere,nRadial=self.nRadial,pruningScheme="None") 
        Vs = Gs.optimizeBasis(potentialType="anc",a=2)

        self.residue = Gs.getMSError(Vs)


        Gs.exportErrorVsDistance(Vs,plotPot=False,path=f"{self.path}/error.png")

        self.C_V_ANC = Vs

    def __getVpotMatrices(self):

        M = myMolecule(self.pathToMolecule,self.orbitalBasisSet,augmentBasis=True,labelAtoms=False)
        Gs = sphericalGrid(M,minDist=0.0,maxDist=15.0,nRadial=300,nSphere=590,pruningScheme="None")

        self.V_EXT = M.ao_pot
        self.V_ANC_B = np.einsum("ji,j,jk,j->ik",Gs.phi,vBpot(Gs.phi,self.C_V_ANC.diagonal()),Gs.phi,Gs.weights,optimize=True)


    def __getDensities(self):
        """
        Ok first calculate the densities with exact analytical external potential
        """


        if not self.runMode:
            self.runMode={"GAMMA" : 0.8,
                          "MAXITER" : 150}
                
        M = myMolecule(self.pathToMolecule,self.orbitalBasisSet,augmentBasis=True,labelAtoms=False)
        res1 = DFTGroundStateRKS(M,self.functional,**self.runMode,OUT=f"{self.path}/PSI_V_EXT.out")

        self.P_EXT = 2*res1["D"]
        self.E_EXT = res1["SCF_E"]

        """
        Now get the one with the Basis set expansion
        """

        res2 = DFTGroundStateRKS(M,self.functional,AOPOT=self.V_ANC_B,**self.runMode,OUT=f"{self.path}/PSI_V_ANC.out")

        self.P_ANC_B = 2*res2["D"]
        self.E_ANC_B = res2["SCF_E"]

    def __saveOutputQuantities(self):
        np.savez_compressed(f"{self.path}/output.npz",
                            P_ANC_B = self.P_ANC_B,
                            E_ANC_B = self.E_ANC_B,
                            P_EXT = self.P_EXT,
                            E_EXT = self.E_EXT)
        

    def __init__(self,pathToMolecule,orbitalBasisSet,functional, runMode=None):
        
        self.nSphere = 590
        self.nRadial = 200
        self.prec = 2

        self.runMode = runMode

        self.pathToMolecule  = pathToMolecule
        self.path= os.path.dirname(pathToMolecule)
        
        self.orbitalBasisSet = orbitalBasisSet
        self.functional = functional
        self.residue = None

        self.C_V_ANC = None
        self.V_ANC_B = None
        self.V_EXT = None

        self.P_EXT = None
        self.P_ANC_B = None

        self.E_EXT = None
        self.E_ANC_B = None

        self.__optimizeTotalBasis()
        self.__plotAllNeigborsTotalBasis()
        self.__plotCMatrix()

        self.__getVpotMatrices()
        self.__saveInputQuantities()

        self.__getDensities()
        self.__saveOutputQuantities()
        self.__plotPMatrix()
        

class softOptimizer(object):
    def __init__(self,pathToMolecule,orbitalBasisSet,functional, runMode=None):
        self.nSphere = 590
        self.nRadial = 300

        self.runMode = runMode

        self.pathToMolecule  = pathToMolecule
        self.path= os.path.dirname(pathToMolecule)
        
        self.orbitalBasisSet = orbitalBasisSet
        self.functional = functional
        self.residue = None

        #Expansion coefficients of the soft part of the external potential
        self.C_V_SOFT = None
        self.C_V_SOFT_ORTH = None
        #external potential matrix, calculated numerically from the basis set expansion of the soft part
        #plus the hard part

        self.V_SOFT_B = None
        self.V_EXT = None
        self.V_MSE = None


        self.P_EXT = None
        self.P_EXT_ORTH = None

        self.P_SOFT_B = None
        self.P_SOFT_B_ORTH = None

        self.P_SAD = None
        self.P_SAD_ORTH = None

        self.E_EXT = None
        self.E_SOFT_B = None

        self.__optimizeTotalBasis()
        
        self.__plotAllNeigborsTotalBasis()
        self.__plotCMatrix()
        
        
        self.__getVpotMatrices()
        self.__saveInputQuantities()

        
        self.__getDensities()
        
        self.__saveOutputQuantities()
        self.__plotPMatrix()
        


    def __optimizeTotalBasis(self):
        M = myMolecule(self.pathToMolecule,self.orbitalBasisSet,augmentBasis=False)
        Gs = sphericalGrid(M,minDist=0.0,maxDist=4.0,nSphere=self.nSphere,nRadial=self.nRadial,pruningScheme="None")

        VpotSoft = generateSoftPotential(M,Gs)
        Vb = Gs.optimizeBasis(potentialType=VpotSoft)

        self.residue = Gs.getMSError(Vb)
        Gs.exportErrorVsDistance(Vb,plotPot=False,path=f"{self.path}/error.png")


        S = M.ao_overlap
        A = M.ao_loewdin
        
        self.C_V_SOFT = Vb
        self.C_V_SOFT_ORTH = S@A.T@ self.C_V_SOFT.diagonal()

    def __plotAllNeigborsTotalBasis(self):
        M = myMolecule(self.pathToMolecule,self.orbitalBasisSet,augmentBasis=False)

        for i,j in M.getCloseNeighbors(thresh=1.8):
            self.__plotTotalBasis(i,j)

    def __plotTotalBasis(self,idx1=0,idx2=1):
        Mt = myMolecule(self.pathToMolecule,self.orbitalBasisSet,augmentBasis=False)
        P1 = Mt.geom[idx1]
        P2 = Mt.geom[idx2]

        v = P2-P1
        r = [x for x in np.arange(-0.5,np.linalg.norm(v)+0.5,0.02) if (abs(x)>0.01) and abs(x-np.linalg.norm(v))>0.01]

        L = np.array([P1 + i*v/np.linalg.norm(v) for i in r])

        GLt = pointGrid(Mt,L)

        VSoftLine = generateSoftPotential(Mt,GLt)
        VBasisLine = vBpot(GLt.phi,self.C_V_SOFT.diagonal())
        VFullBasis = generateFullPotential(Mt,GLt,VBasisLine)
        VFull = vpot(GLt.mol.geom,GLt.mol.elez,L)

        plt.plot(r,VSoftLine,label=r"$v_{soft}$",color="red")
        plt.plot(r,VBasisLine,label=r"$v_{soft,B}$",color="green")
        plt.plot(r,VFull,label=r"$v_{ext}$",color="blue")
        plt.plot(r,VFullBasis,label=r"$v_{ext,B}$",color="orange")


        plt.legend()
        plt.ylabel("External Potential [a.u.]")
        plt.xlabel("Distance [a.u.]")

        #plt.ylim((1.1*np.min(VANC),10))
        plt.savefig(f"{self.path}/bond_{idx1}_{idx2}.png",dpi=300)
        plt.clf()

    def __plotCMatrix(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        im = ax.imshow(self.C_V_SOFT,cmap="seismic",
                       norm=SymLogNorm(linthresh=1.0,vmin=-np.max(np.abs(self.C_V_SOFT)),
                       vmax=np.max(np.abs(self.C_V_SOFT))))
        fig.colorbar(im)
        ax.set_title(r"$C^{v,soft}_{\mu \nu}$")
        ax.set_ylabel(r"$ \mu\ \mathrm{index}$")
        ax.set_xlabel(r"$ \nu\ \mathrm{index}$")
        fig.tight_layout()
        fig.savefig(f"{self.path}/C_V_SOFT.png",dpi=300)

    def __getVpotMatrices(self):

        M = myMolecule(self.pathToMolecule,self.orbitalBasisSet,augmentBasis=False,labelAtoms=False)
        GL = sphericalGrid(M,minDist=0.0,maxDist=15.0,nRadial=300,nSphere=590,pruningScheme="None")


        VpotSoftBasis = vBpot(GL.phi,self.C_V_SOFT.diagonal())
        VpotHardBasis = generateFullPotential(M,GL,VpotSoftBasis)

        self.V_EXT = M.ao_pot
        self.V_SOFT_B = np.einsum("ji,j,jk,j->ik",GL.phi,VpotHardBasis,GL.phi,GL.weights,optimize=True)
        self.V_MSE = np.mean( (self.V_EXT -self.V_SOFT_B)**2)

    def __saveInputQuantities(self):
        M = myMolecule(self.pathToMolecule,self.orbitalBasisSet,augmentBasis=False,labelAtoms=False)

        info = {"Path" : self.pathToMolecule,
                "BasisSet" : self.orbitalBasisSet,
                "Functional" : self.functional,
                "nSphere" : self.nSphere,
                "nRadial" : self.nRadial,
                "minDist" : 0.0,
                "maxDist" : 4.0,
                "fitError" : self.residue,
                "V_MSE"    : self.V_MSE}  


        np.savez_compressed(f"{self.path}/input.npz",
                C_V_SOFT=self.C_V_SOFT,
                C_V_SOFT_ORTH = self.C_V_SOFT_ORTH,
                V_SOFT_B=self.V_SOFT_B,
                V_EXT=self.V_EXT,
                NEL = M.nElectrons,
                INFO=info)



    def __getDensities(self):
        """
        Ok first calculate the densities with exact analytical external potential
        """


        if not self.runMode:
            self.runMode={"GAMMA" : 0.8,
                          "MAXITER" : 150}
                
        M = myMolecule(self.pathToMolecule,self.orbitalBasisSet,augmentBasis=False,labelAtoms=False)
        res1 = DFTGroundStateRKS(M,self.functional,**self.runMode,OUT=f"{self.path}/PSI_V_EXT.out")

        S = M.ao_overlap
        A = M.ao_loewdin

        self.P_EXT = res1["D"]
        self.P_EXT_ORTH = S@A.T @ self.P_EXT @A @S.T
        self.E_EXT = res1["SCF_E"]


        """
        Now get the one with the Basis set expansion
        """

        res2 = DFTGroundStateRKS(M,self.functional,AOPOT=self.V_SOFT_B,**self.runMode,OUT=f"{self.path}/PSI_V_SOFT.out")

        self.P_SOFT_B = res2["D"]
        self.P_SOFT_B_ORTH =  S@A.T@ self.P_SOFT_B @A @S.T
        self.E_SOFT_B = res2["SCF_E"]


        self.P_SAD = getSADGuess(M).Da().np
        self.P_SAD_ORTH = S@A.T @ self.P_SAD @A @S.T


    def __saveOutputQuantities(self):
        np.savez_compressed(f"{self.path}/output.npz",
                            P_SOFT_B = self.P_SOFT_B,
                            P_SOFT_B_ORTH = self.P_SOFT_B_ORTH,
                            E_SOFT_B = self.E_SOFT_B,
                            P_SAD = self.P_SAD,
                            P_SAD_ORTH = self.P_SAD_ORTH,
                            P_EXT = self.P_EXT,
                            P_EXT_ORTH = self.P_EXT_ORTH,
                            E_EXT = self.E_EXT)
    def __plotPMatrix(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        im = ax.imshow(self.P_SOFT_B,cmap="seismic",
                       norm=SymLogNorm(linthresh=1.0,vmin=-np.max(np.abs(self.P_SOFT_B)),
                       vmax=np.max(np.abs(self.P_SOFT_B))))
        fig.colorbar(im)
        ax.set_title(r"$P^{v,soft}_{\mu \nu}$")
        ax.set_ylabel(r"$ \mu\ \mathrm{index}$")
        ax.set_xlabel(r"$ \nu\ \mathrm{index}$")
        fig.tight_layout()
        fig.savefig(f"{self.path}/P_SOFT_B.png",dpi=300)


if __name__ == "__main__":
    from vpot.calc import DFTGroundState
    
    # opti = threeStepOptimizer("tests/CH3Cl.xyz","def2-TZVP")      
    # M = myMolecule("tests/CH3Cl.xyz","def2-TZVP")

    # Gs = sphericalGrid(M,minDist=0.0,maxDist=4.0,nRadial=opti.nRadial,nSphere=opti.nSphere,pruningScheme="None")
    # VPOT = np.einsum("ji,j,jk,j->ik",Gs.phi,vpot(Gs.mol.geom,Gs.mol.elez,Gs.points),Gs.phi,Gs.weights)

    # MyDFT,_ = DFTGroundState(M,"PBE",AOPOT=VPOT, GAMMA=0.8)
    # EPsi,_ = DFTGroundState(M,"PBE", GAMMA=0.8)

    #opti = simpleOptimizer("tests/6-QM7/1/1.xyz","def2-TZVP","PBE")
    opti = simpleOptimizer("tests/6-QM7/503/503.xyz","def2-TZVP","PBE")

    #print(f"VBAS: {MyDFT} VPOT: {EPsi}")



    #opti.plotAllNeigborsTotalBasis()