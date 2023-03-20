import psi4
from ase.io import read 
from ase.atoms import Atoms
import numpy as np
from psi4.driver import qcdb



class myMolecule(object):
    
    
    def getBasisDict(self):
        return self.basisDict
    
    def setBasisDict(self,newBasis,quiet=False):
        self.basisDict = newBasis
        self.basisSet = psi4.core.BasisSet.construct_from_pydict(self.psi4Mol,self.basisDict,-1)
        self.basisDict["additionalMessage"] = "\nBASIS SET WAS AUGMENTED!! In Total now "+str(self.basisSet.nbf())+" functions\n\n" 
        if quiet==False:
            psi4.core.print_out(self.basisDict['message'])
            psi4.core.print_out(self.basisDict['additionalMessage'])
        
        mints = psi4.core.MintsHelper(self.basisSet)
        self.ao_pot = mints.ao_potential().np
    
        
        

    def __init__(self, xyzFile: str, basisString : str = "def2-TZVP",augmentBasis=True):
        """
        Initialize molecule from an XYZ file
        """
        
        augmented_functions = {"H" : [[0,
                                 (5.405316891299325,0.6820129648915216),
                                 (4.888708126704825,0.5266237600075427),
                                 (0.4225473780551931,0.5074699312426245)
                                     ],
                                     [2,
                                 (5.405316891299325,0.6820129648915216),
                                 (4.888708126704825,0.5266237600075427),
                                 (0.4225473780551931,0.5074699312426245)
                                     ]
                                     ],
                           "C": [[0,
                                  (172.96261138256966,-0.5214371382885968),
                                  (29.314946661375846,-0.39513913610629037),
                                  (185.99626466742993,-0.7351102294708934),
                                  (183.42014380299744,-0.6888203785681367),
                                  (3.1759813492302214,-0.40781260095499905),
                                 ],
                                 [1,
                                  (114.40612202857618,63.76591878887223),
                                  (94.21886928828486,-35.49369196905149),
                                  (90.24482250171442,-56.31530581107358),
                                  (92.67776068886113,-43.53466026766574),
                                  (107.97793044475458,33.43572608100986),
                                 ],
                                 [2,
                                  (0.6959100051760453,-1.3057675363246695),
                                  (56.55606458424013,-4.721212130352486),
                                  (52.48985683705824,-6.360235268472501),
                                  (39.94626677179216,-5.2953449886893535),
                                  (87.77817707938222,-3.6712027665137428),
                                 ]
                               ],
                           "N": [[0,
                                  (251.51870646639483,0.9560118363676076),
                                  (36.737639718300116,0.24648479132601472),
                                  (2.916437729034275,0.15901766056009572)
                                ]],
                           "O": [[0,
                                  (43.48891578194754,0.24672926006761906),
                                  (325.1856769113162,0.9580250597686155),
                                  (3.203782667107184,0.14598855120120136)
                                ]],
                           "S": [[0,
                                  (1221.6467961997446,0.9668870867462587),
                                  (105.15458456551536,0.23770984486133562),
                                  (4.961913991444419,0.09286221588640114)
                                ]],
                           "CL": [[0,
                                   (20.84926858428245, -438.89923983799537), 
                                   (1472.3411687771475, -167.66996116542745), 
                                   (190.119815661342, 547.189643416135), 
                                   (190.12040864298228, -748.3011941633538), 
                                   (0.87699139849954, -1944.6646517547522)
                                   ],
                                   [1,
                                    (762.7206245300981, -51.54800226065295),
                                    (755.7840552457565, -52.59792973416408),
                                    (759.9439787477676, -52.48372010038438),
                                    (768.2069454879173, -51.52732911423492),
                                    (776.0888431628953, -51.64974821742614)
                                   ],
                                    [2,
                                     (81.19427345532856, -34.1371162227081),
                                     (11.307847069950917, -23.38016826238459),
                                     (179.91558069913086, -29.754957722641464), 
                                     (1.492788554159369, -2.049927073387957), 
                                     (40.53240785945691, -20.2212452238696)
                                    ]]
                          }

        mol = psi4.geometry(f"""
            {self._readXYZFile(xyzFile)}
            symmetry c1
            nocom
            noreorient
            """)

                      
        #wfn = psi4.core.Wavefunction.build(mol, basisString)
        self.basisSet = None
        self.basisDict = None
        self.xyzFile = xyzFile
        self.basisString = basisString
        self.geom, self.mass, self.elem, self.elez, self.uniq = mol.to_arrays()

        #get center of mass, good for plotting
        self.com = self.mass @ self.geom / self.mass.sum()
        self.nElectrons = np.sum(self.elez)


        self.psi4Mol  = mol 
        
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
                    for j in reversed(augmented_functions[i[0]]):
                        i.insert(2,j)
            
            
            self.setBasisDict(basDict)
            

        else:
            if augmentBasis == False:
                self.basisSet= psi4.core.BasisSet.build(mol,"BASIS",basisString,'ORBITAL',None,-1,False)                   
            elif augmentBasis == True:
                a,basDict =qcdb.BasisSet.pyconstruct(mol.to_dict(),'BASIS',
                                                     basisString,fitrole='ORBITAL',
                                                     other=None,return_dict=True,return_atomlist=False)
            
                for i in basDict["shell_map"]:
                    for j in reversed(augmented_functions[i[0]]):
                        i.insert(2,j)

                
                self.setBasisDict(basDict)
                
                
        mints = psi4.core.MintsHelper(self.basisSet)
        self.xyzFile = xyzFile
        self.basisString = basisString
        self.geom, self.mass, self.elem, self.elez, self.uniq = mol.to_arrays()

        #get center of mass, good for plotting
        self.com = self.mass @ self.geom / self.mass.sum()
        self.nElectrons = np.sum(self.elez)
        self.ao_pot = mints.ao_potential().np


        self.psi4Mol  = mol 


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

