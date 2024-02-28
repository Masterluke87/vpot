import psi4
from ase.io import read 
from ase.atoms import Atoms
import itertools
from ase.units import Bohr
import numpy as np
from psi4.driver import qcdb
from copy import deepcopy
from vpot.calc.basis.augment import augmented_functions



class myMolecule(object):
    def __init__(self, xyzFile: str, basisString : str = "def2-TZVP",augmentBasis=True, labelAtoms=False):
        """
        Initialize molecule from an XYZ file
        """
        mol = psi4.geometry(f"""
            {self._readXYZFile(xyzFile,labelAtoms)}
            symmetry c1
            nocom
            noreorient
            """)

                      
        #wfn = psi4.core.Wavefunction.build(mol, basisString)
        self.basisSet = None
        self.basisDict = None
        self.orbitalDict = None
        self.augmentDict = None

        self.xyzFile = xyzFile
        self.basisString = basisString
        self.geom, self.mass, self.elem, self.elez, self.uniq = mol.to_arrays()

        #get center of mass, good for plotting
        self.com = self.mass @ self.geom / self.mass.sum()
        self.nElectrons = np.sum(self.elez)


        self.psi4Mol  = mol 
        self.augmentBasis = augmentBasis
        
        if basisString=="":
            print("Basis string is empty")
            """
            We want to create just the augmentation basis as an input
            """
            a,basDict =qcdb.BasisSet.pyconstruct(mol.to_dict(),'BASIS',
                                                     "def2-SVP",fitrole='ORBITAL',
                                                     other=None,return_dict=True,return_atomlist=False)
            for i in basDict["shell_map"]:
                del i[2:]
                
            for i in basDict["shell_map"]:
                    elem = ''.join([k for k in i[0] if not k.isdigit()])
                    for j in reversed(augmented_functions[elem]):
                        i.insert(2,j)
            
            
            self.setBasisDict(augmentDict=basDict)
            

        else:
            if augmentBasis == False:
                a,orbDict = qcdb.BasisSet.pyconstruct(mol.to_dict(),'BASIS',
                                                     basisString,fitrole='ORBITAL',
                                                     other=None,return_dict=True,return_atomlist=False)
                self.setBasisDict(orbitalDict=orbDict,augmentDict=None)

            elif augmentBasis == True:
                a,augmentDict = qcdb.BasisSet.pyconstruct(mol.to_dict(),'BASIS',
                                                     "def2-SVP",fitrole='ORBITAL',
                                                     other=None,return_dict=True,return_atomlist=False)
                for i in augmentDict["shell_map"]:
                    del i[2:]
                for i in augmentDict["shell_map"]:
                    elem = ''.join([k for k in i[0] if not k.isdigit()])
                    for j in reversed(augmented_functions[elem]):
                        i.insert(2,j)

                a,orbDict = qcdb.BasisSet.pyconstruct(mol.to_dict(),'BASIS',
                                                     basisString,fitrole='ORBITAL',
                                                     other=None,return_dict=True,return_atomlist=False)
                
                self.setBasisDict(orbitalDict=orbDict,augmentDict=augmentDict)


            elif type(augmentBasis) == dict:
                a,orbDict = qcdb.BasisSet.pyconstruct(mol.to_dict(),'BASIS',
                                                     basisString,fitrole='ORBITAL',
                                                     other=None,return_dict=True,return_atomlist=False)
                self.setBasisDict(orbitalDict=orbDict,augmentDict=augmentBasis)
                
                
        mints = psi4.core.MintsHelper(self.basisSet)
        self.xyzFile = xyzFile
        self.basisString = basisString
        self.geom, self.mass, self.elem, self.elez, self.uniq = mol.to_arrays()

        #get center of mass, good for plotting
        self.com = self.mass @ self.geom / self.mass.sum()
        self.nElectrons = np.sum(self.elez)
        self.ao_pot = mints.ao_potential().np
        self.ao_overlap = mints.ao_overlap().np

        A = psi4.core.Matrix.from_array(self.ao_overlap)
        A.power(-0.5,1e-16)
        A = np.asarray(A)
        self.ao_loewdin = A

        self.psi4Mol  = mol 



    def getBasisDict(self):
        return deepcopy(self.basisDict)

    def getAugmentDict(self):
        return deepcopy(self.augmentDict)
    
    def getOrbitalDict(self):
        return deepcopy(self.orbitalDict)
    

    def keepAugmentBasisForIndex(self,atomIdx=0):
        a,newAug = qcdb.BasisSet.pyconstruct(self.psi4Mol.to_dict(),'BASIS',
                                            "def2-SVP",fitrole='ORBITAL',
                                            other=None,return_dict=True,return_atomlist=False)

        for i in newAug["shell_map"]:
            del i[2:]

        for c,i in enumerate(self.augmentDict["shell_map"]):
            if c == atomIdx:
                newAug["shell_map"][c] += i[2:]

        self.setBasisDict(orbitalDict=self.orbitalDict,augmentDict=newAug)

    def keepAugmentBasisForAtomType(self,atomType="C"):
        a,newAug = qcdb.BasisSet.pyconstruct(self.psi4Mol.to_dict(),'BASIS',
                                            "def2-SVP",fitrole='ORBITAL',
                                            other=None,return_dict=True,return_atomlist=False)

        for i in newAug["shell_map"]:
            del i[2:]

        for c,i in enumerate(self.augmentDict["shell_map"]):
            if i[0] == atomType:
                newAug["shell_map"][c] += i[2:]

        self.setBasisDict(orbitalDict=self.orbitalDict,augmentDict=newAug)

    def getAngmomAndContraction(self):
        res = {}
        for i in self.augmentDict["shell_map"]:
            if i[0] not in res:
                res[i[0]] = {}
                res[i[0]]["angMom"] = [] 
                res[i[0]]["Contraction"]= [] 
                for j in i[2:]:
                    res[i[0]]["angMom"].append(j[0])
                    res[i[0]]["Contraction"].append(len(j)-1)
        return res
    
    def setBasisDict(self,orbitalDict=None,augmentDict=None,quiet=True):

        """
        CASE 0: Neither an orbital basis is provided nor an augmentation basis is provided
        we return set an empty basis set
        """
        mol = self.psi4Mol

        if (orbitalDict is None) and (augmentDict is None):
            a,basisDict = qcdb.BasisSet.pyconstruct(mol.to_dict(),'BASIS',
                                                "def2-SVP",fitrole='ORBITAL',
                                                other=None,return_dict=True,return_atomlist=False)
            for i in basisDict["shell_map"]:
                del i[2:] 

            self.augmentDict = augmentDict
            self.orbitalDict = orbitalDict
            self.basisDict = basisDict
    
        """
        CASE 1: Orbital Dict provided but no augmentation dict 
        """

        if (orbitalDict is not None) and (augmentDict is None):
            basisDict = orbitalDict

            self.augmentDict = augmentDict
            self.orbitalDict = orbitalDict
            self.basisDict = orbitalDict
            self.basisDict["additionalMessage"] = "\nBasis set not augmented!! \n\n" 

        """
        CASE 2: orbital dict is none, augemtation dict is provided
        """
        if (orbitalDict is None) and (augmentDict is not None):
            self.augmentDict = augmentDict
            self.orbitalDict = orbitalDict
            self.basisDict = augmentDict
            self.basisDict["additionalMessage"] = "\nBasis set only augmented!!\n\n" 
        """
        CASE 3: both basis sets are provided
        """
        if (orbitalDict is not None) and (augmentDict is not None):
            self.augmentDict = augmentDict
            self.orbitalDict = orbitalDict
            a,basisDict = qcdb.BasisSet.pyconstruct(mol.to_dict(),'BASIS',
                                                orbitalDict["name"],fitrole='ORBITAL',
                                                other=None,return_dict=True,return_atomlist=False)
            for i in basisDict["shell_map"]:
                del i[2:]

            for i,j,k in zip(basisDict["shell_map"],orbitalDict["shell_map"],augmentDict["shell_map"]):
                i+=k[2:]
                i+=j[2:] 

            self.basisDict = basisDict
            self.basisDict["additionalMessage"] = "\nBasis is augmented!\n\n" 
        

        self.basisSet = psi4.core.BasisSet.construct_from_pydict(self.psi4Mol,self.basisDict,-1)

        mints = psi4.core.MintsHelper(self.basisSet)
        self.ao_pot = mints.ao_potential().np
        
        self.basisDict["additionalMessage"] += "Basis set has "+str(self.basisSet.nbf())+" functions\n"
            
        if quiet==False:
            psi4.core.print_out(self.basisDict['message'])
            psi4.core.print_out(self.basisDict['additionalMessage'])

    def runPSI4(self,method : str):
        
        psi4.geometry(f"""
            {self._readXYZFile(self.xyzFile,labelAtoms=False)}
            symmetry c1
            nocom
            noreorient
            """)
        E,wfn = psi4.energy(f"{method}/{self.basisString}", return_wfn=True)

        return E,wfn

    def getCloseNeighbors(self,thresh=2.0,printDist=False):
        """
        NOTE: We do Angstrom conversion
        """
        neighbors = [] 

        for (i,j) in itertools.combinations(range(len(self.geom)),2):
            if (np.linalg.norm(self.geom[i]-self.geom[j])*Bohr)<thresh:
                neighbors.append((i,j))
            if printDist:
                print(f"({i},{j}: {(np.linalg.norm(self.geom[i]-self.geom[j])*Bohr)}")

        return neighbors




    def _readXYZFile(self,xyzFile: str,labelAtoms):
        mol = read(xyzFile)
        S = ""
        if labelAtoms==False:
            for i in mol:
                S += f"{i.symbol} {i.position[0]} {i.position[1]} {i.position[2]} \n"
        else:
            for c,i in enumerate(mol):
                S += f"{i.symbol}{c+1} {i.position[0]} {i.position[1]} {i.position[2]} \n"
        return S
    

if __name__ == "__main__":
    """
    perform a test 
    """
    M = myMolecule("6-QM7/1.xyz","def2-TZVP")
    M = myMolecule("6-QM7/1.xyz")
    M = myMolecule("6-QM7/1.xyz","def2-SVP")

