import psi4
from ase.io import read 
from ase.atoms import Atoms
import numpy as np


class myMolecule(object):
    def __init__(self, xyzFile: str, basisString : str = "def2-TZVP"):
        """
        Initialize molecule from an XYZ file
        """

        mol = psi4.geometry(f"""
            {self._readXYZFile(xyzFile)}
            symmetry c1
            nocom
            noreorient
            """)

        wfn = psi4.core.Wavefunction.build(mol, basisString)
        mints = psi4.core.MintsHelper(wfn.basisset())
        self.xyzFile = xyzFile
        self.basisString = basisString
        self.geom, self.mass, self.elem, self.elez, self.uniq = mol.to_arrays()

        #get center of mass, good for plotting
        self.com = self.mass @ self.geom / self.mass.sum()
        self.nElectrons = np.sum(self.elez)
        self.ao_pot = mints.ao_potential().np



        self.basisSet = wfn.basisset()


    def runPSI4(self,method : str):
        
        psi4.geometry(f"""
            {self._readXYZFile(self.xyzFile)}
            symmetry c1
            nocom
            noreorient
            """)
        E,wfn = psi4.energy(f"{method}/{self.basisString}", return_wfn=True)

        return E,wfn




    def _readXYZFile(self,xyzFile: str):
        mol = read(xyzFile)
        S = ""
        for i in mol:
            S += f"{i.symbol} {i.position[0]} {i.position[1]} {i.position[2]} \n"
        return S
    

if __name__ == "__main__":
    """
    perform a test 
    """
    M = myMolecule("6-QM7/1.xyz","def2-TZVP")
    M = myMolecule("6-QM7/1.xyz")
    M = myMolecule("6-QM7/1.xyz","def2-SVP")

