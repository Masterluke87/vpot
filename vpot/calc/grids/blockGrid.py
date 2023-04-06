import psi4
import logging
import numpy as np
from matplotlib import pyplot as plt
import copy
import time
from vpot.calc import myMolecule
from vpot.calc.potential import vpot,vBpot,vpotANC
from ase.data.colors import jmol_colors
from ase.data import chemical_symbols
from vpot.calc.grids import myGrid


def genCube(mol, centeredAroundOrigin=False, thresh=5.0):
    """
    Place the molecule in a cube and return the coordinates
    """
    geom = mol.geom
    elez = mol.elez

    
    xmin,xmax = np.min(geom[:,0])-thresh,np.max(geom[:,0])+thresh
    ymin,ymax = np.min(geom[:,1])-thresh,np.max(geom[:,1])+thresh
    zmin,zmax = np.min(geom[:,2])-thresh,np.max(geom[:,2])+thresh
    
    logging.info("Geometry:")
    logging.info(geom)
    
    
    logging.info(f"xmin: {xmin:8.4f} a.u.; xmax: {xmax:8.4f} a.u.")
    logging.info(f"ymin: {ymin:8.4f} a.u.; ymax: {ymax:8.4f} a.u.")
    logging.info(f"zmin: {zmin:8.4f} a.u.; zmax: {zmax:8.4f} a.u.")
    
    logging.info(f"--- {xmin:8.4f} {ymin:8.4f} {zmin:8.4f} VPOT: {vpot(geom,elez,np.array([[xmin,ymin,zmin]]))}")
    logging.info(f"+++ {xmax:8.4f} {ymax:8.4f} {zmax:8.4f} VPOT: {vpot(geom,elez,np.array([[xmax,ymax,zmax]]))}")
    
    logging.info(f"--+ {xmin:8.4f} {ymin:8.4f} {zmax:8.4f} VPOT: {vpot(geom,elez,np.array([[xmin,ymin,zmax]]))}")
    logging.info(f"-+- {xmin:8.4f} {ymax:8.4f} {zmin:8.4f} VPOT: {vpot(geom,elez,np.array([[xmin,ymax,zmin]]))}")
    logging.info(f"+-- {xmax:8.4f} {ymin:8.4f} {zmin:8.4f} VPOT: {vpot(geom,elez,np.array([[xmax,ymin,zmin]]))}")
    
    logging.info(f"-++ {xmin:8.4f} {ymax:8.4f} {zmax:8.4f} VPOT: {vpot(geom,elez,np.array([[xmin,ymax,zmax]]))}")
    logging.info(f"++- {xmax:8.4f} {ymax:8.4f} {zmin:8.4f} VPOT: {vpot(geom,elez,np.array([[xmax,ymax,zmin]]))}")
    logging.info(f"+-+ {xmax:8.4f} {ymin:8.4f} {zmax:8.4f} VPOT: {vpot(geom,elez,np.array([[xmax,ymin,zmax]]))}")
    
    
    if centeredAroundOrigin:
        cube = {"xmin" : -np.max(np.abs([xmin,xmax])),
                 "xmax" :  np.max(np.abs([xmin,xmax])),
                "ymin" : -np.max(np.abs([ymin,ymax])),
                "ymax" :  np.max(np.abs([ymin,ymax])),
                "zmin" : -np.max(np.abs([zmin,zmax])),
                "zmax" :  np.max(np.abs([zmin,zmax])),
                 }
    else:
        cube = {"xmin" : xmin,
            "xmax" : xmax,
            "ymin" : ymin,
            "ymax" : ymax,
            "zmin" : zmin,
            "zmax" : zmax
           }

    return cube

        
        
        
class pointGrid(myGrid):
    def __init__(self,
                 mol: myMolecule,
                 points: np.array):
        self.points = copy.deepcopy(points)
        self.mol = mol
        
        
        basis_extents = psi4.core.BasisExtents(mol.basisSet, 0.0)
        xs = psi4.core.Vector.from_array(self.points[:,0])
        ys = psi4.core.Vector.from_array(self.points[:,1])
        zs = psi4.core.Vector.from_array(self.points[:,2])
        ws = psi4.core.Vector.from_array(np.ones(len(self.points)))

        blockopoints = psi4.core.BlockOPoints(xs, ys, zs, ws, basis_extents)
        max_points = blockopoints.npoints()
        max_functions = mol.basisSet.nbf()
        funcs = psi4.core.BasisFunctions(mol.basisSet, max_points, max_functions)
        

        funcs.compute_functions(blockopoints)
        self.phi = np.array(funcs.basis_values()['PHI'])
        

class blockGrid(myGrid):

    def __init__(self, 
                 mol: myMolecule,
                 gridSpacing: float=0.2,
                 minDist: float=0.25,
                 maxDist: float=7.5,
                 centeredAroundOrigin=False,
                 cube=None
                 ):
        myGrid.__init__(self,mol)

        if not cube:
            self.cube = genCube(self.mol,centeredAroundOrigin=False,thresh=maxDist)
        else:
            self.cube = cube
        self._genBlockGrid(self.cube,gridSpacing,minDist,maxDist)

    def genBlockPoints(self,
                       cube: dict,
                       gridSpacing: float = 0.2,
                       minDist: float = 0.25,
                       maxDist: float = 7.5):
        mol = self.mol


        delta = 0.001
        basis_extents = psi4.core.BasisExtents(mol.basisSet, delta)


        xdim = np.arange(cube["xmin"],cube["xmax"],gridSpacing)
        ydim = np.arange(cube["ymin"],cube["ymax"],gridSpacing)
        zdim = np.arange(cube["zmin"],cube["zmax"],gridSpacing)

        logging.info(f"xdim: {len(xdim)}, ydim: {len(ydim)}, zdim: {len(zdim)}")

        P = np.array([[i,j,k] for i in xdim
                                for j in ydim 
                                for k in zdim])

        Dist    = np.array([np.linalg.norm(P[:,:3] - x,axis=1) for x in mol.geom]).transpose()
        Dmin    = np.min(Dist,axis=1)
        Pts     = P[np.where((Dmin>minDist) & (Dmin < maxDist))]
        
        self.points = Pts[:,:3]
        self.gridInfo["type"] = "block"
        self.gridInfo["minDist"] = minDist
        self.gridInfo["nPoints"] = len(self.points)
        self.gridInfo["block"] = cube
        self.gridInfo["xdim"] = len(xdim)
        self.gridInfo["ydim"] = len(ydim)
        self.gridInfo["zdim"] = len(zdim)

    def projectBasis(self):

        mol = self.mol
        basis_extents = psi4.core.BasisExtents(mol.basisSet, 0.0)


        xs = psi4.core.Vector.from_array(self.points[:,0])
        ys = psi4.core.Vector.from_array(self.points[:,1])
        zs = psi4.core.Vector.from_array(self.points[:,2])
        ws = psi4.core.Vector.from_array(np.ones(len(self.points)))

        blockopoints = psi4.core.BlockOPoints(xs, ys, zs, ws, basis_extents)
        max_points = blockopoints.npoints()
        max_functions = mol.basisSet.nbf()
        funcs = psi4.core.BasisFunctions(mol.basisSet, max_points, max_functions)
        lpos = np.array(blockopoints.functions_local_to_global())
        npoints = blockopoints.npoints()

        funcs.compute_functions(blockopoints)
        
        vals = np.array(funcs.basis_values()['PHI'])
        self.phi = vals


    def _genBlockGrid(self, 
                     cube: dict,
                     gridSpacing: float = 0.2,
                     minDist: float = 0.25,
                     maxDist: float = 7.5
                     ):
        self.genBlockPoints(cube,gridSpacing,minDist,maxDist)
        self.projectBasis()
        

class blockAtomicGrid(myGrid):
    def __init__(self, 
                 mol: myMolecule,
                 atomType: str = "C",
                 gridSpacing: float=0.2,
                 minDist: float=0.25,
                 maxDist: float=7.5,
                 centeredAroundOrigin=False,
                 cube=None
                 ):
        myGrid.__init__(self,mol)

        if not cube:
            self.cube = genCube(self.mol,centeredAroundOrigin=False,thresh=maxDist)
        else:
            self.cube = cube
        self._genBlockGrid(self.cube,atomType,gridSpacing,minDist,maxDist)

    def genBlockPoints(self,
                       cube: dict,
                       atomType: str,
                       gridSpacing: float = 0.2,
                       minDist: float = 0.25,
                       maxDist: float = 7.5):
        mol = self.mol


        delta = 0.001
        basis_extents = psi4.core.BasisExtents(mol.basisSet, delta)


        xdim = np.arange(cube["xmin"],cube["xmax"],gridSpacing)
        ydim = np.arange(cube["ymin"],cube["ymax"],gridSpacing)
        zdim = np.arange(cube["zmin"],cube["zmax"],gridSpacing)

        logging.info(f"xdim: {len(xdim)}, ydim: {len(ydim)}, zdim: {len(zdim)}")

        atomLabels = [chemical_symbols[int(x)].upper() for x in mol.elez]   
        
        idx = [c for c,x in enumerate(atomLabels) if x==atomType]

        P = np.array([[i,j,k] for i in xdim
                                for j in ydim 
                                for k in zdim])

        Dist    = np.array([np.linalg.norm(P[:,:3] - x,axis=1) for x in mol.geom]).transpose()
        filterDist = Dist[:,idx]
        Dmin    = np.min(filterDist,axis=1)
        Pts     = P[np.where((Dmin>minDist) & (Dmin < maxDist))]
        
        self.points = Pts[:,:3]
        self.gridInfo["type"] = "block"
        self.gridInfo["minDist"] = minDist
        self.gridInfo["nPoints"] = len(self.points)
        self.gridInfo["block"] = cube
        self.gridInfo["xdim"] = len(xdim)
        self.gridInfo["ydim"] = len(ydim)
        self.gridInfo["zdim"] = len(zdim)

    def projectBasis(self):

        mol = self.mol
        basis_extents = psi4.core.BasisExtents(mol.basisSet, 0.0)


        xs = psi4.core.Vector.from_array(self.points[:,0])
        ys = psi4.core.Vector.from_array(self.points[:,1])
        zs = psi4.core.Vector.from_array(self.points[:,2])
        ws = psi4.core.Vector.from_array(np.ones(len(self.points)))

        blockopoints = psi4.core.BlockOPoints(xs, ys, zs, ws, basis_extents)
        max_points = blockopoints.npoints()
        max_functions = mol.basisSet.nbf()
        funcs = psi4.core.BasisFunctions(mol.basisSet, max_points, max_functions)
        lpos = np.array(blockopoints.functions_local_to_global())
        npoints = blockopoints.npoints()

        funcs.compute_functions(blockopoints)
        
        vals = np.array(funcs.basis_values()['PHI'])
        self.phi = vals


    def _genBlockGrid(self, 
                     cube: dict,
                     atomType: str,
                     gridSpacing: float = 0.2,
                     minDist: float = 0.25,
                     maxDist: float = 7.5
                     ):
        self.genBlockPoints(cube,atomType,gridSpacing,minDist,maxDist)
        self.projectBasis()


if __name__ == "__main__":
    
    logging.basicConfig(filename='grids.log', level=logging.DEBUG,filemode="w")
    
    """
    M = myMolecule("tests/6-QM7/1.xyz","def2-TZVP")
    G = sphericalGrid(M,maxDist=5.0,nSphere=2702)
    logging.info(f"{G.gridInfo}")
    V1 = G.optimizeBasis()
    logging.info(f"{G.gridInfo}")
    logging.info(V1)

    G2 = blockGrid(M,minDist=0.2,gridSpacing=0.15)
    V2 = G2.optimizeBasis()
    logging.info(V2)

    G3 = blockGrid(M,minDist=0.2,gridSpacing=0.15,centeredAroundOrigin=True)
    V3 = G3.optimizeBasis()
    logging.info(V3)
    """
    M = myMolecule("tests/6-QM7/1.xyz","def2-TZVP")
    G = sphericalGrid(M)
    E,wfn = G.mol.runPSI4("HF")
    Da = wfn.Da().np
    
