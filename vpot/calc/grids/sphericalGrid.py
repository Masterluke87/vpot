import psi4
import logging
import numpy as np
from matplotlib import pyplot as plt
import time
from vpot.calc.mol import myMolecule
from vpot.calc.potential import vpot,vBpot,vpotANC
from ase.data.colors import jmol_colors
from ase.data import chemical_symbols
from vpot.calc.grids import myGrid

class sphericalGrid(myGrid):
    def __init__(self,mol: myMolecule,
                minDist: float=0.25,
                maxDist: float=7.5,
                nRadial: int=75, 
                nSphere: int=302,
                radialScheme: str  = "BECKE",
                pruningScheme: str = "TREUTLER"):
        myGrid.__init__(self,mol)
        self._genSphericalGrid(minDist,maxDist,nRadial,nSphere,radialScheme,pruningScheme)

    def _genSphericalPoints(self,
                    minDist: float=0.25,
                    maxDist: float=7.5,
                    nRadial: int=75, 
                    nSphere: int=302,
                    radialScheme: str  = "BECKE",
                    pruningScheme: str = "TREUTLER"):
        
        start = time.perf_counter()
        mol = self.mol

        delta = 0.0
        psi4.set_options({"DFT_SPHERICAL_POINTS"   : nSphere,
                            "DFT_RADIAL_POINTS"    : nRadial,
                            "DFT_RADIAL_SCHEME"    : radialScheme,
                            "DFT_PRUNING_SCHEME"   : pruningScheme,
                            "DFT_REMOVE_DISTANT_POINTS" : True})  

        
        functional = psi4.driver.dft.build_superfunctional("svwn", True)[0]
        Vpot       = psi4.core.VBase.build(mol.basisSet, functional, "RV")
        Vpot.initialize()

        P       = np.array(Vpot.get_np_xyzw()).transpose()
        Dist    = np.array([np.linalg.norm(P[:,:3] - x,axis=1) for x in mol.geom]).transpose()
        Dmin    = np.min(Dist,axis=1)
        Pts     = P[np.where((Dmin>minDist) & (Dmin < maxDist))]

        print(f"Filter: {Pts.shape}")
        logging.info(f"genSphereTime T2: {time.perf_counter()-start:10.2f} s")


        for c,i in enumerate(mol.geom):
            logging.info(f"radii  : {sorted(list(set(np.round(np.linalg.norm(Pts[:,:3]-i,axis=1),decimals=2))))}); {len(set(np.round(np.linalg.norm(Pts[:,:3]-i,axis=1),decimals=2)))}")

        logging.info(f"genSphereTime T3: {time.perf_counter()-start:10.2f} s")

        self.points  = Pts[:,:3]
        self.weights = Pts[:,3]
        self.gridInfo["type"] = "spherical"
        self.gridInfo["minDist"] = minDist
        self.gridInfo["maxDist"] = maxDist
        self.gridInfo["nRadical"] = nRadial
        self.gridInfo["nSphere"] = nSphere
        self.gridInfo["nPoints"] = len(self.points)
        self.gridInfo["radialScheme"]  = radialScheme
        self.gridInfo["pruningScheme"] = pruningScheme

    def _projectBasis(self):
        mol = self.mol
        basis_extents = psi4.core.BasisExtents(mol.basisSet, 0.0)
        
        xs = psi4.core.Vector.from_array(self.points[:,0])
        ys = psi4.core.Vector.from_array(self.points[:,1])
        zs = psi4.core.Vector.from_array(self.points[:,2])
        ws = psi4.core.Vector.from_array(self.weights)

        blockopoints = psi4.core.BlockOPoints(xs, ys, zs, ws, basis_extents)
        max_points = blockopoints.npoints()
        max_functions = mol.basisSet.nbf()
        funcs = psi4.core.BasisFunctions(mol.basisSet, max_points, max_functions)
        lpos = np.array(blockopoints.functions_local_to_global())
        npoints = blockopoints.npoints()

        funcs.compute_functions(blockopoints)
        vals = np.array(funcs.basis_values()['PHI'])

        self.phi = vals
        





    def _genSphericalGrid(self,
                    minDist: float=0.25,
                    maxDist: float=7.5,
                    nRadial: int=75, 
                    nSphere: int=302,
                    radialScheme: str  = "BECKE",
                    pruningScheme: str = "TREUTLER"):
        
        self._genSphericalPoints(minDist,maxDist,nRadial,nSphere,radialScheme,pruningScheme)
        self._projectBasis()
        """
        Some docs
        """
        """
        start = time.perf_counter()
        mol = self.mol

        delta = 0.0
        psi4.set_options({"DFT_SPHERICAL_POINTS"   : nSphere,
                            "DFT_RADIAL_POINTS"    : nRadial,
                            "DFT_RADIAL_SCHEME"    : radialScheme,
                            "DFT_PRUNING_SCHEME"   : pruningScheme,
                            "DFT_REMOVE_DISTANT_POINTS" : True})  

        basis_extents = psi4.core.BasisExtents(mol.basisSet, delta)
        functional = psi4.driver.dft.build_superfunctional("svwn", True)[0]
        Vpot       = psi4.core.VBase.build(mol.basisSet, functional, "RV")
        Vpot.initialize()
        """
        
        """
        x, y, z, w = Vpot.get_np_xyzw()

        logging.info(f"genSphereTime -T1.1: {time.perf_counter()-start:10.2f} s")
        Pts = np.array([[i,j,k,l] for i,j,k,l in zip(x,y,z,w) ])
        logging.info(f"genSphereTime -T1.2: {time.perf_counter()-start:10.2f} s")
        
        tmpPts = []
        for i in Pts:
            if (np.min([np.linalg.norm(i[:3]-x) for x in mol.geom]) > minDist) and (np.min([np.linalg.norm(i[:3]-x) for x in mol.geom]) < maxDist):
                tmpPts.append(i)

        logging.info(f"Before filter: {len(Pts)} after filter: {len(tmpPts)}")
        Pts = np.array(tmpPts)
        logging.info(f"genSphereTime T2: {time.perf_counter()-start:10.2f} s")
        """
        
        
        """
        P       = np.array(Vpot.get_np_xyzw()).transpose()
        Dist    = np.array([np.linalg.norm(P[:,:3] - x,axis=1) for x in mol.geom]).transpose()
        Dmin    = np.min(Dist,axis=1)
        Pts     = P[np.where((Dmin>minDist) & (Dmin < maxDist))]

        print(f"Filter: {Pts.shape}")
        logging.info(f"genSphereTime T2: {time.perf_counter()-start:10.2f} s")


        for c,i in enumerate(mol.geom):
            logging.info(f"radii  : {sorted(list(set(np.round(np.linalg.norm(Pts[:,:3]-i,axis=1),decimals=2))))}); {len(set(np.round(np.linalg.norm(Pts[:,:3]-i,axis=1),decimals=2)))}")

        logging.info(f"genSphereTime T3: {time.perf_counter()-start:10.2f} s")

        xs = psi4.core.Vector.from_array(Pts[:,0])
        ys = psi4.core.Vector.from_array(Pts[:,1])
        zs = psi4.core.Vector.from_array(Pts[:,2])
        ws = psi4.core.Vector.from_array(Pts[:,3])

        blockopoints = psi4.core.BlockOPoints(xs, ys, zs, ws, basis_extents)
        max_points = blockopoints.npoints()
        max_functions = mol.basisSet.nbf()
        funcs = psi4.core.BasisFunctions(mol.basisSet, max_points, max_functions)
        lpos = np.array(blockopoints.functions_local_to_global())
        npoints = blockopoints.npoints()

        funcs.compute_functions(blockopoints)
        phi = np.array(funcs.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

        vals = np.array(funcs.basis_values()['PHI'])

        all_zeros = []
        for col_idx in range(vals.shape[1]):
            if np.allclose(vals[:, col_idx], 0.0):
                all_zeros.append(col_idx)

        logging.info(f'basis fcns that are all zeros: {all_zeros}')
        self.points = Pts[:,:3]
        self.weights = Pts[:,3]
        self.phi = vals
        self.gridInfo["type"] = "spherical"
        self.gridInfo["minDist"] = minDist
        self.gridInfo["maxDist"] = maxDist
        self.gridInfo["nRadical"] = nRadial
        self.gridInfo["nSphere"] = nSphere
        self.gridInfo["nPoints"] = len(self.points)
        self.gridInfo["radialScheme"]  = radialScheme
        self.gridInfo["pruningScheme"] = pruningScheme
        logging.info(f"genSphereTime: {time.perf_counter()-start:10.2f} s")
        """
        
    def optimizeBasisWeights(self,potentialType:str="anc",a:int=2):
        start = time.perf_counter()
        
        """
        we try to solve
        A.X = B
        phi.C = Vpot
        
        
        we try to use the the weights from 
        """
        
        Dist    = np.array([np.linalg.norm(self.points[:,:3] - x,axis=1) for x in self.mol.geom]).transpose()
        Dmin    = np.min(Dist,axis=1)
        
        W = np.sqrt(0.5*(1-np.exp(-(Dmin/2)**5))+0.5)
        
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

class sphericalAtomicGrid(myGrid):
    def __init__(self,
                 mol: myMolecule,
                 atomType : str="C",
                 minDist: float=0.25,
                 maxDist: float=7.5,
                 nRadial: int=75, 
                 nSphere: int=302,
                 radialScheme: str  = "BECKE",
                 pruningScheme: str = "TREUTLER"):
        myGrid.__init__(self,mol)
        self._genSphericalGrid(atomType,minDist,maxDist,nRadial,nSphere,radialScheme,pruningScheme)
        
    def _genSphericalGrid(self,atomType,minDist,maxDist,nRadial,nSphere,radialScheme,pruningScheme):
        print(f"Creating a spherical grid for atomtype {atomType}")
        start = time.perf_counter()
        mol = self.mol

        delta = 0.0
        psi4.set_options({"DFT_SPHERICAL_POINTS"   : nSphere,
                            "DFT_RADIAL_POINTS"    : nRadial,
                            "DFT_RADIAL_SCHEME"    : radialScheme,
                            "DFT_PRUNING_SCHEME"   : pruningScheme,
                            "DFT_REMOVE_DISTANT_POINTS" : True})  

        basis_extents = psi4.core.BasisExtents(mol.basisSet, delta)
        functional = psi4.driver.dft.build_superfunctional("svwn", True)[0]
        Vpot       = psi4.core.VBase.build(mol.basisSet, functional, "RV")
        Vpot.initialize()
        
        atomLabels = [chemical_symbols[int(x)].upper() for x in mol.elez]
        #print(f"atomLabels: {atomLabels}")
        
        
        idx = [c for c,x in enumerate(atomLabels) if x==atomType]
        
        
        
        P       = np.array(Vpot.get_np_xyzw()).transpose()
        #print(f"Total {P.shape}")
        Dist    = np.array([np.linalg.norm(P[:,:3] - x,axis=1) for x in mol.geom]).transpose()
        filterDist = Dist[:,idx]
        Dmin    = np.min(filterDist,axis=1)
        Pts     = P[np.where((Dmin>minDist) & (Dmin < maxDist))]

        #print(f"Filter {Pts.shape}")

        logging.info(f"genSphereTime T2: {time.perf_counter()-start:10.2f} s")


        for c,i in enumerate(mol.geom):
            logging.info(f"radii  : {sorted(list(set(np.round(np.linalg.norm(Pts[:,:3]-i,axis=1),decimals=2))))}); {len(set(np.round(np.linalg.norm(Pts[:,:3]-i,axis=1),decimals=2)))}")

        logging.info(f"genSphereTime T3: {time.perf_counter()-start:10.2f} s")

        xs = psi4.core.Vector.from_array(Pts[:,0])
        ys = psi4.core.Vector.from_array(Pts[:,1])
        zs = psi4.core.Vector.from_array(Pts[:,2])
        ws = psi4.core.Vector.from_array(Pts[:,3])

        blockopoints = psi4.core.BlockOPoints(xs, ys, zs, ws, basis_extents)
        max_points = blockopoints.npoints()
        max_functions = mol.basisSet.nbf()
        funcs = psi4.core.BasisFunctions(mol.basisSet, max_points, max_functions)
        lpos = np.array(blockopoints.functions_local_to_global())
        npoints = blockopoints.npoints()

        funcs.compute_functions(blockopoints)
        phi = np.array(funcs.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

        vals = np.array(funcs.basis_values()['PHI'])

        all_zeros = []
        for col_idx in range(vals.shape[1]):
            if np.allclose(vals[:, col_idx], 0.0):
                all_zeros.append(col_idx)

        logging.info(f'basis fcns that are all zeros: {all_zeros}')
        self.points = Pts[:,:3]
        self.weights = Pts[:,3]
        self.phi = vals
        self.gridInfo["type"] = "spherical"
        self.gridInfo["minDist"] = minDist
        self.gridInfo["maxDist"] = maxDist
        self.gridInfo["nRadical"] = nRadial
        self.gridInfo["nSphere"] = nSphere
        self.gridInfo["nPoints"] = len(self.points)
        self.gridInfo["radialScheme"]  = radialScheme
        self.gridInfo["pruningScheme"] = pruningScheme
        logging.info(f"genSphereTime: {time.perf_counter()-start:10.2f} s")
        
        
