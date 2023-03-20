import psi4
import logging
import numpy as np
from matplotlib import pyplot as plt
import time
from .mol import myMolecule
from .potential import vpot,vBpot,vpotANC
from ase.data.colors import jmol_colors
from ase.data import chemical_symbols



colors = {"H" : "grey",
          "C" : "steelblue",
          "N" : ""}


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

    def getMSError(self,V):
        return np.mean(np.square((vBpot(self.phi,V.diagonal()) - self.vpot )))
        
    def optimizeBasis(self,potentialType:str = "exact",a:int = 2):
        start = time.perf_counter()
        
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

        else:
            raise(f"Unknown potential {potentialType}")

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
            plt.savefig(f"{path}")
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

    def _genSphericalGrid(self,
                    minDist: float=0.25,
                    maxDist: float=7.5,
                    nRadial: int=75, 
                    nSphere: int=302,
                    radialScheme: str  = "BECKE",
                    pruningScheme: str = "TREUTLER"):
        """
        Some docs
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
        
        
        
        
        
class pointGrid(myGrid):
    def __init__(self,
                 mol: myMolecule,
                 points: np.array):
        self.points = points
        self.mol = mol
        
        delta = 0.001
        basis_extents = psi4.core.BasisExtents(mol.basisSet, delta)
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
        phi = np.array(funcs.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

        vals = np.array(funcs.basis_values()['PHI'])
        self.phi = vals
        

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

    def _genBlockGrid(self, 
                     cube: dict,
                     gridSpacing: float = 0.2,
                     minDist: float = 0.25,
                     maxDist: float = 7.5
                     ):
        
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

        """
        tmpPts = []
        for i in Pts:
            if (np.min([np.linalg.norm(i[:3]-x) for x in mol.geom]) > minDist):
                tmpPts.append(i)

        logging.info(f"Before filter: {len(Pts)} after filter: {len(tmpPts)}")
        Pts = np.array(tmpPts)
        """
        xs = psi4.core.Vector.from_array(Pts[:,0])
        ys = psi4.core.Vector.from_array(Pts[:,1])
        zs = psi4.core.Vector.from_array(Pts[:,2])
        ws = psi4.core.Vector.from_array(np.ones(len(Pts)))

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

        logging.info(f'basis fcns that are all zeros {all_zeros}')
        self.points = Pts[:,:3]
        self.phi = vals

        self.gridInfo["type"] = "block"
        self.gridInfo["minDist"] = minDist
        self.gridInfo["nPoints"] = len(self.points)
        self.gridInfo["block"] = cube
        self.gridInfo["xdim"] = len(xdim)
        self.gridInfo["ydim"] = len(ydim)
        self.gridInfo["zdim"] = len(zdim)
        




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
    
