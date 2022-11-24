import psi4
import logging
import numpy as np
from matplotlib import pyplot as plt
import time
from .mol import myMolecule
from .potential import vpot,vBpot


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

    def optimizeBasis(self):
        start = time.perf_counter()
        VMat, resis,_,_ = np.linalg.lstsq(self.phi, -vpot(self.mol.geom,
                                                      self.mol.elez,
                                                      self.points),rcond=-1)

        mints = psi4.core.MintsHelper(self.mol.basisSet)
        X =mints.ao_overlap().np
        for c1,i in enumerate(VMat):
            X[c1,c1] = i

        logging.info(f"resis : {resis} ")
        logging.info(f"optimizeBasis time: {time.perf_counter()-start:10.2f} s")
        return X

    def exportErrorVsDistance(self,V,mode :str ="min", pltLabel: str= "", path : str ="",plotPot:bool=False):
        
        Error = vBpot(self.phi,V.diagonal()) - vpot(self.mol.geom,self.mol.elez,self.points)
        if mode=="com":
            Dists = np.array([np.linalg.norm(self.points - self.mol.com,axis=1)]).transpose()
        if mode=="min":
            Dists = np.min(np.array([np.linalg.norm(self.points - x,axis=1) for x in self.mol.geom]).transpose(),axis=1)
        plt.plot(Dists,Error,"o",markersize=0.6,label=pltLabel)
        plt.ylabel("Error wrt. exact potential [a.u.]")
        plt.xlabel("Distance to neareast nuclei [a.u.]")
      
        if plotPot:
            ax2 = plt.gca().twinx()
            ax2.plot(Dists,vpot(self.mol.geom,self.mol.elez,self.points),"o",color="red",markersize=0.6,label="Potential")           
            ax2.set_ylabel("Potential [a.u.]")
        
        if path:
            plt.savefig(f"{path}")
            plt.clf()
        
        



    def printStats(self, V):
        logging.info(f"GridInfo : {self.gridInfo} ")
        logging.info(f"residue  : {np.sum(np.square((vBpot(self.phi,V.diagonal()) - vpot(self.mol.geom ,self.mol.elez,self.points))))}")
        logging.info(f"MeanError: {np.mean(np.square((vBpot(self.phi,V.diagonal()) - vpot(self.mol.geom,self.mol.elez,self.points))))}")
        logging.info(f"MaxError : {np.max(np.square((vBpot(self.phi,V.diagonal()) - vpot(self.mol.geom ,self.mol.elez,self.points))))}")
        logging.info(f"MinError : {np.min(np.square((vBpot(self.phi,V.diagonal()) - vpot(self.mol.geom ,self.mol.elez,self.points))))}")





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

        delta = 0.01
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

    

class blockGrid(myGrid):

    def __init__(self, 
                 mol: myMolecule,
                 gridSpacing: float=0.2,
                 minDist: float=0.25,
                 maxDist: float=7.5,
                 centeredAroundOrigin=False
                 ):
        myGrid.__init__(self,mol)

        self.cube = genCube(self.mol,centeredAroundOrigin=False,thresh=maxDist)
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
