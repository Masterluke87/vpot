from vpot.calc import myMolecule
from vpot.calc.grids import sphericalGrid, blockGrid, pointGrid,sphericalAtomicGrid,blockAtomicGrid,sphericalIndexGrid
from vpot.calc.potential import vpot,vBpot, vpotANC
from vpot.calc import DFTGroundState
from matplotlib import pyplot as plt
import numpy as np

class Optimizer(object):
    
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


if __name__ == "__main__":
    from vpot.calc import DFTGroundState
    
    opti = Optimizer("tests/CH3Cl.xyz","def2-TZVP")      
    M = myMolecule("tests/CH3Cl.xyz","def2-TZVP")

    Gs = sphericalGrid(M,minDist=0.0,maxDist=4.0,nRadial=opti.nRadial,nSphere=opti.nSphere,pruningScheme="None")
    VPOT = np.einsum("ji,j,jk,j->ik",Gs.phi,vpot(Gs.mol.geom,Gs.mol.elez,Gs.points),Gs.phi,Gs.weights)

    MyDFT,_ = DFTGroundState(M,"PBE",AOPOT=VPOT, GAMMA=0.8)
    EPsi,_ = DFTGroundState(M,"PBE", GAMMA=0.8)


    print(f"VBAS: {MyDFT} VPOT: {EPsi}")



    #opti.plotAllNeigborsTotalBasis()