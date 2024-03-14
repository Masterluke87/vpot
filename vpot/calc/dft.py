# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 01:57:59 2018

@author: luke
"""
import psi4
import numpy as np
from vpot.calc.kshelper import diag_H,ACDIIS,Timer,printHeader,ACDIISRKS
from vpot.calc.kshelper import DIIS_helper
from vpot.calc.mol import myMolecule
import os
import time
import logging
import pdb

def calcEnergyWithPerturbedDensity(M, Pinit, fac, diagFock=True, perturb=False,func="PBE0",**kwargs):
    """
    This function should calculate the energy of a given electron density
    The density cab be perturbed prior to the energy calculation
    Further a FOCK Matrix can be constructed and diagonalized
    """
   
    mints = psi4.core.MintsHelper(M.basisSet)
    wfn   = psi4.core.Wavefunction.build(M.psi4Mol,M.basisSet)
    aux   = psi4.core.BasisSet.build(M.psi4Mol, "DF_BASIS_SCF", "", "JKFIT", M.basisString, puream=1 if wfn.basisset().has_puream() else 0)
    
    S = np.asarray(mints.ao_overlap())
    T = np.asarray(mints.ao_kinetic())

    if "AOPOT" in kwargs:
        V = kwargs["AOPOT"]
        psi4.core.print_out("\nChanged AO-potential!\n")
    else:
        V = np.asarray(mints.ao_potential())
    
    H = T + V
    
    A = mints.ao_overlap()
    A.power(-0.5, 1.e-16)
    A = np.asarray(A)
    
    nbf    = wfn.nso()
    ndocc  = wfn.nalpha()
                
    Va   = psi4.core.Matrix(nbf,nbf)
    Cocc = psi4.core.Matrix(nbf, ndocc)
    
    sup = psi4.driver.dft.build_superfunctional(f"{func}", True)[0]
    
    D_m = psi4.core.Matrix(nbf,nbf)
    Vpot = psi4.core.VBase.build(wfn.basisset(), sup, "RV")
    Vpot.initialize()
    
    jk = psi4.core.JK.build(wfn.basisset(),aux=aux,jk_type="MEM_DF")
    glob_mem = psi4.core.get_memory()/8
    jk.set_memory(int(glob_mem*0.6))
    if (sup.is_x_hybrid()):
        jk.set_do_K(True)
    if (sup.is_x_lrc()):
        jk.set_omega(sup.x_omega())
        jk.set_do_wK(True)
    jk.initialize()
    
    
    
    noise = np.zeros((nbf,nbf))
    if perturb:
        noise[np.triu_indices(nbf)]=(np.random.random(int(nbf*(nbf+1)/2))-0.5)/0.5 
        noise = (np.tril(noise.T) + np.triu(noise))/fac
        #trace should be zero
        noise[np.diag_indices_from(noise)] -= np.trace(noise)/nbf

    
    nTrace = np.trace(Pinit)
    if np.isclose(ndocc,nTrace):
        print("Density seems already in the orthogonal basis \n")
        Porth = Pinit
    else:
        print("Transform to orthogonal basis \n")
        Porth = S@A.T@Pinit@S.T@A
    
    PorthPert = Porth + noise
    mse = np.square(Porth-PorthPert).mean()
    print(f"MSE: {mse}")
                
   
    
    
    """
    Obtain a Cocc
    """
    
    print(f"Trace of othogonalized density matrix: {np.trace(PorthPert)}")
    U,Sigma,V = np.linalg.svd(Pinit)
    idx = Sigma>=1E-6
    print(f"S: {Sigma[idx]}")
    print(f"Consistency check (-->0) #1: {np.max(U[:,:ndocc]@U[:,:ndocc].T - PorthPert)}")
    
    CoccOrth = U[:,:ndocc]
    Cocc.np[:] = A@CoccOrth

    D_m.np[:] = Cocc.np@Cocc.np.T
    
    """
    End obtain Cocc
    """
    
    
    
    """
    Build Fock
    """
    if diagFock==True:
        Vpot.set_D([D_m])
        Vpot.compute_V([Va])
        jk.C_left_add(Cocc)
        jk.compute()
    
        J = np.asarray(jk.J()[0])

        F = H + 2*J + Va
        if sup.is_x_hybrid():
            F -= sup.x_alpha()*np.asarray(jk.K()[0])
        if sup.is_x_lrc(): 
            F -= sup.x_beta()*np.asarray(jk.wK()[0]) 
            
        """
        END BUILD FOCK
        """
        
        """
        DIAG FOCK
        """
    
        C,eps = diag_H(F, A)
        
        Cocc.np[:]  = C[:, :ndocc]
        D_m.np[:]   = (Cocc.np @ Cocc.np.T)

    
    D =  Cocc.np@Cocc.np.T
    """
    Energy Evaluation
    """
    Vpot.set_D([D_m])
    Vpot.compute_V([Va])
    jk.C_left_add(Cocc)
    jk.compute()
    
    one_electron_E  =   np.sum(D * 2*H)
    coulomb_E       =   np.sum(D * 2*np.asarray(jk.J()[0]))
    exchange_E  = 0.0
    if sup.is_x_hybrid():
        exchange_E -=  sup.x_alpha() * np.sum(D * np.asarray(jk.K()[0]))
    if sup.is_x_lrc():
        exchange_E -=  sup.x_beta() * np.sum(D * np.asarray(jk.wK()[0]))
    
    
    XC_E = Vpot.quadrature_values()["FUNCTIONAL"]
    
    SCF_E = 0.0
    SCF_E += M.psi4Mol.nuclear_repulsion_energy()
    SCF_E += one_electron_E
    SCF_E += coulomb_E
    SCF_E += exchange_E
    SCF_E += XC_E
    return {"E" : SCF_E,
            "mse": mse,
            "fac" : fac,
            "D" : D,
           }

def constructSADGuess(M,func="PBE0",returnEnergies=False):
    """
    Constructs a SAD guess and used the natural orbitals to 
    create an idempotentent matrix. This should give a variational initial guess.
    """
    from vpot.calc.kshelper import atomicOccupations
    from scipy.linalg import block_diag
    
    uniqueElem = list(set(M.elem))
    atomicDensities = {}
    atomicEnergies = {}
    currentOutput = psi4.core.get_output_file()
    for atom in uniqueElem:
        fname = f"{os.path.dirname(M.xyzFile)}/{np.random.randint(1E4,1E5)}.xyz"
        with open(fname,"w") as f:
            f.write("1\n\n")
            f.write(f"{atom} 0.0 0.0 0.0")
        A = myMolecule(fname,M.basisString,M.augmentBasis)    
        res = DFTGroundState(A,func,GAMMA=0.25,OCCA=atomicOccupations[atom],OCCB=atomicOccupations[atom],OUT="/dev/null")
        os.remove(fname)
        atomicDensities[atom] = (res['Da']+res['Db'])/2.0
        atomicEnergies[atom] = res['SCF_E']

   

    DGuess = block_diag(*[atomicDensities[x] for x in M.elem])
    EAtoms = [atomicEnergies[x] for x in M.elem]

    DguessOrth = M.ao_overlap@M.ao_loewdin.T@DGuess@M.ao_loewdin@M.ao_overlap.T
    vals,vecs = np.linalg.eigh(-1.0*DguessOrth)

    vecsOcc = vecs[:,:int(M.nElectrons/2.0)]
    DNoOrth = vecsOcc @ vecsOcc.T

    if returnEnergies:
        EAtoms = [atomicEnergies[x] for x in M.elem]
        ESad   =  calcEnergyWithPerturbedDensity(M, DNoOrth, 0.0, diagFock=False, perturb=False,func=func)
        psi4.core.set_output_file(currentOutput,True)
        return (DNoOrth,EAtoms,ESad["E"])
    else:
        psi4.core.set_output_file(currentOutput,True)
        return DNoOrth

def getSADGuess(M):
    Mtmp = myMolecule(M.xyzFile,M.basisString,augmentBasis=M.augmentBasis,labelAtoms=False)
    a,basisDict = psi4.driver.qcdb.BasisSet.pyconstruct(Mtmp.psi4Mol.to_dict(),'BASIS',Mtmp.orbitalDict["name"],fitrole='ORBITAL',other=None,return_dict=True,return_atomlist=True)

    for i in range(len(basisDict)):
        basisDict[i]["shell_map"] = [Mtmp.basisDict["shell_map"][i]]

    sad_basis_list = [psi4.core.BasisSet.construct_from_pydict(psi4.core.Molecule.from_dict(basisDict[i]["molecule"]),basisDict[i],-1) for i in range(len(basisDict))]
    sad_fitting_list = psi4.core.BasisSet.build(M.psi4Mol, "DF_BASIS_SAD", psi4.core.get_option("SCF", "DF_BASIS_SAD"), puream=M.basisSet.has_puream(), return_atomlist=True)
    SAD = psi4.core.SADGuess.build_SAD(Mtmp.basisSet, sad_basis_list)

    SAD.set_atomic_fit_bases(sad_fitting_list)
    SAD.compute_guess()
    return SAD


def DFTGroundStateRKS(mol,func,**kwargs):
    """
    Perform restrictred Kohn-Sham
    """

    if "OUT" in kwargs:
        psi4.core.set_output_file(kwargs["OUT"])
    else:
        psi4.core.set_output_file("stdout")
    psi4.core.reopen_outfile()


    printHeader("Entering Ground State Restricted Kohn-Sham")
    options = {
        "PREFIX"    : "VPOT",
        "E_CONV"    : 1E-8,
        "D_CONV"    : 1E-6,
        "MAXITER"   : 150,
        "BASIS"     : mol.basisString,
        "GAMMA"     : 0.95,
        "GUESS"     : "SAD",
        "VSHIFT"    : 0.0, #mEh
        "DIIS_LEN"  : 6,
        "DIIS_MODE" : "ADIIS+CDIIS",
        "DIIS_EPS"  : 0.1,
        "MIXMODE"   : "DAMP",
        "RESTART"   : False}
    
    for i in options.keys():
        if i in kwargs:
            options[i] = kwargs[i]
            
    printHeader("Options Run:",2)
    for key,value in options.items():
        psi4.core.print_out(f"{key:20s} {str(value):20s} \n")

    printHeader("Basis Set:",2)
    wfn   = psi4.core.Wavefunction.build(mol.psi4Mol,mol.basisSet)
    aux   = psi4.core.BasisSet.build(mol.psi4Mol, "DF_BASIS_SCF", "", "JKFIT", mol.basisString, puream=1 if wfn.basisset().has_puream() else 0)
    sup = psi4.driver.dft.build_superfunctional(func, True)[0]

    psi4.core.be_quiet()
    mints = psi4.core.MintsHelper(mol.basisSet)    
    sup.allocate()

    psi4.core.reopen_outfile()
    
    S = np.asarray(mints.ao_overlap())
    T = np.asarray(mints.ao_kinetic())
    
    H = np.zeros((mints.nbf(),mints.nbf()))

    if "AOPOT" in kwargs:
        V = kwargs["AOPOT"]
        psi4.core.print_out("\nChanged AO-potential!\n")
    else:
        V = np.asarray(mints.ao_potential())
    H = T+V

    if wfn.basisset().has_ECP():
        ECP = mints.ao_ecp()
        H += ECP


    A = mints.ao_overlap()
    A.power(-0.5,1.e-16)
    A = np.asarray(A)


    Enuc = mol.psi4Mol.nuclear_repulsion_energy()
    Eold = 0.0

    nbf    = wfn.nso()
    ndocc  = wfn.nalpha()
    

    Va = psi4.core.Matrix(nbf,nbf)
    Cocc = psi4.core.Matrix(nbf, ndocc)
    
    Vpot = psi4.core.VBase.build(wfn.basisset(), sup, "RV")
    Vpot.initialize()

    """
    INITIAL DENSITY OR CINP or CoreGuess
    """

    if "Pinp" in kwargs:
        psi4.core.print_out("\n\nTaking Density from Input \n\n")
        Pinit = kwargs["Pinp"]
        assert Pinit.shape == (nbf,nbf)
        #From here it is assumed that the density matrix is in the ordinary non-orthogonal basis
        #One should check if the trace of the Matrix is sufficiently close to the number of electron/2
        nTrace = np.trace(Pinit)
        if np.isclose(ndocc,nTrace):
            psi4.core.print_out("Density seems already in the orthogonal basis \n")
            Porth = Pinit
        else:
            psi4.core.print_out("Transform to orthogonal basis \n")
            Porth = S@A.T@Pinit@S.T@A

        assert np.isclose(ndocc,np.trace(Porth),rtol=1E-4,atol=1E-4)

        psi4.core.print_out(f"Trace of orthogonalized density matrix: {np.trace(Porth)}")
        U,Sigma,V = np.linalg.svd(Porth)
        CoccOrth = U[:,:ndocc]
        Cocc.np[:] = A@CoccOrth
        D = Cocc.np @ Cocc.np.T

    elif "Cinp" in kwargs:
        psi4.core.print_out("\n\n Taking Coefficients from kwargs!\n\n")
        C = kwargs["Cinp"]
        Cocc.np[:]  = C[:, :ndocc]
        D      = Cocc.np @ Cocc.np.T

    elif options["GUESS"] == "SAD":
        psi4.core.print_out("Doing a SAD guess\n")
        Pinit = constructSADGuess(mol)
        assert Pinit.shape == (nbf,nbf)
        #From here it is assumed that the density matrix is in the ordinary non-orthogonal basis
        #One should check if the trace of the Matrix is sufficiently close to the number of electron/2
        nTrace = np.trace(Pinit)
        if np.isclose(ndocc,nTrace):
            psi4.core.print_out("Density seems already in the orthogonal basis \n")
            Porth = Pinit
        else:
            psi4.core.print_out("Transform to orthogonal basis \n")
            Porth = S@A.T@Pinit@S.T@A

        assert np.isclose(mol.nElectrons/2.0,np.trace(Porth))

        psi4.core.print_out(f"Trace of orthogonalized density matrix: {np.trace(Porth)}")
        U,Sigma,V = np.linalg.svd(Porth)
        CoccOrth = U[:,:ndocc]
        Cocc.np[:] = A@CoccOrth
        D = Cocc.np @ Cocc.np.T

    else:
        psi4.core.print_out("\n\n Doing a core guess!\n\n")
        C,eps = diag_H(H, A)
        Cocc.np[:]  = C[:, :ndocc]
        D        = Cocc.np @ Cocc.np.T



    HLgap = 1000

    printHeader("Molecule:",2)
    mol.psi4Mol.print_out()
    printHeader("XC & JK-Info:",2)

    jk = psi4.core.JK.build(wfn.basisset(),aux=aux,jk_type="MEM_DF")
    glob_mem = psi4.core.get_memory()/8
    jk.set_memory(int(glob_mem*0.6))
    
    if (sup.is_x_hybrid()):
        jk.set_do_K(True)
    if (sup.is_x_lrc()):
        jk.set_omega(sup.x_omega())
        jk.set_do_wK(True)
    jk.initialize()
    jk.C_left_add(Cocc)

    D_m = psi4.core.Matrix(nbf,nbf)
    
    sup.print_out()
    psi4.core.print_out("\n\n")
    mol.basisSet.print_out()

    jk.print_header()

    diis = ACDIISRKS(max_vec=options["DIIS_LEN"],diismode=options["DIIS_MODE"])
    diis_e = 1000.0

    printHeader("Starting SCF:",2)    
    psi4.core.print_out("""{:>10} {:8.2E}
{:>10} {:8.2E}
{:>10} {:8.4f}
{:>10} {:8.4f}
{:>10} {:8.2E}
{:>10} {:8d}
{:>10} {:8d}
{:>10} {:^11}""".format(
    "E_CONV:",options["E_CONV"],
    "D_CONV:",options["D_CONV"],
    "VSHIFT:",options["VSHIFT"],
    "DAMP:",options["GAMMA"],
    "DIIS_EPS:",options["DIIS_EPS"],
    "MAXITER:", options["MAXITER"],
    "DIIS_LEN:",options["DIIS_LEN"],
    "DIIS_MODE:",options["DIIS_MODE"]))

    myTimer = Timer()

    psi4.core.print_out("\n\n{:^4} {:^14} {:^11} {:^11} {:^11} {:^11} {:^6} {:^6} {:^3} \n".format("# IT", "Escf", "dEscf","Derror","DIIS-E","MIX","HL-Gap","Time","DL"))
    psi4.core.print_out("="*80+"\n")
    diis_counter = 0

    for SCF_ITER in range(1, options["MAXITER"] + 1):
        myTimer.addStart("SCF")     
        jk.compute()

        """
        Build Fock
        """
        D_m.np[:] = D
        Vpot.set_D([D_m])
        Vpot.compute_V([Va])

        J = np.asarray(jk.J()[0])

        if SCF_ITER>1 :
            FOld = np.copy(F)

        F = H + 2*J + Va
        if sup.is_x_hybrid():
            F -= sup.x_alpha()*np.asarray(jk.K()[0])
        if sup.is_x_lrc(): 
            F -= sup.x_beta()*np.asarray(jk.wK()[0]) 
        """
        END BUILD FOCK
        """


        if options["VSHIFT"] > 0.0:
            if HLgap < options["VSHIFT"]:
                FMO = C.T @ F @ C
                Cinv = np.linalg.inv(C)
                idxs = range(ndocc,nbf)

                FMO[idxs,idxs] += (options["VSHIFT"] / 1000) 
                F = Cinv.T @ FMO @ Cinv

        """
        CALC E
        """
        one_electron_E  =   np.sum(D * 2*H)
        coulomb_E       =   np.sum(D * 2*J)
        exchange_E  = 0.0
        if sup.is_x_hybrid():
            exchange_E -=  sup.x_alpha() * np.sum(D * np.asarray(jk.K()[0]))
        if sup.is_x_lrc():
            exchange_E -=  sup.x_beta() * np.sum(D * np.asarray(jk.wK()[0]))


        XC_E = Vpot.quadrature_values()["FUNCTIONAL"]


        SCF_E = 0.0
        SCF_E += Enuc
        SCF_E += one_electron_E
        SCF_E += coulomb_E
        SCF_E += exchange_E
        SCF_E += XC_E
        """
        END CALCE
        """

        """
        DIIS/MIXING
        """
        diis_e = np.ravel(A.T@(F@D@S - S@D@F)@A)
        diis.add(F,D,diis_e)


        if ("DIIS" in options["MIXMODE"]) and (SCF_ITER>1):
            # Extrapolate alpha & beta Fock matrices separately
            F = diis.extrapolate(DIISError)
            diis_counter += 1

            if (diis_counter >= 2*options["DIIS_LEN"]):
                diis.reset()
                diis_counter = 0
                psi4.core.print_out("Resetting DIIS\n")

        elif (options["MIXMODE"] == "DAMP") and (SCF_ITER>1):
            #...but use damping to obtain the new Fock matrices
            F = (1-options["GAMMA"]) * np.copy(F) + (options["GAMMA"]) * FOld
            

        """
        END DIIS/MIXING
        """

        """
        DIAG F-tilde -> get D
        """
        DOld = np.copy(D)
        
        C,eps = diag_H(F, A)
        Cocc.np[:]  = C[:, :ndocc]
        D      = (Cocc.np @ Cocc.np.T)

        """
        END DIAG F + BUILD D
        """

        DError = np.linalg.norm(DOld-D)
        EError = (SCF_E - Eold)
        DIISError = (np.sum(diis_e**2)**0.5)


        HLgap = (eps[ndocc]-eps[ndocc-1])*1000
     
        """
        OUTPUT
        """
        myTimer.addEnd("SCF")
        psi4.core.print_out(" {:3d} {:14.8f} {:11.3E} {:11.3E} {:11.3E} {:^11} {:6.2f} {:6.2f} {:3d} \n".format(SCF_ITER,
             SCF_E,
             EError,
             DError,
             DIISError,
             options["MIXMODE"],
             HLgap,
             myTimer.getTime("SCF"),
             len(diis.F)))
                  
        psi4.core.flush_outfile()
        if (abs(DIISError) < options["DIIS_EPS"]):
            options["MIXMODE"] = options["DIIS_MODE"]
        else:
            options["MIXMODE"] = "DAMP"        
        
        if (abs(EError) < options["E_CONV"]) and (abs(DError)<options["D_CONV"]):
            break

        Eold = SCF_E

        if SCF_ITER == options["MAXITER"]:
            psi4.core.print_out("\n\nMaximum number of SCF cycles exceeded.")
            results = {"SCF_E" : SCF_E, "D":D,"C":C,"S":S,"X":A}
            return results

    psi4.core.print_out("\n\nFINAL GS SCF ENERGY: {:12.8f} [Ha] \n\n".format(SCF_E))

    results = {"SCF_E" : SCF_E, "D":D,"C" : C}

    return results




def DFTGroundState(mol,func,**kwargs):
    """
    Perform unrestrictred Kohn-Sham
    """

    if "OUT" in kwargs:
        psi4.core.set_output_file(kwargs["OUT"])
    else:
        psi4.core.set_output_file("stdout")
    psi4.core.reopen_outfile()

    printHeader("Entering Ground State Kohn-Sham")

    options = {
        "PREFIX"    : "VPOT",
        "E_CONV"    : 1E-8,
        "D_CONV"    : 1E-6,
        "MAXITER"   : 150,
        "BASIS"     : mol.basisString,
        "GAMMA"     : 0.95,
        "DIIS_LEN"  : 6,
        "DIIS_MODE" : "ADIIS+CDIIS",
        "DIIS_EPS"  : 0.1,
        "OCCA"      : None,
        "OCCB"      : None,
        "MIXMODE"   : "DAMP",
        "RESTART"   : False}
    
    for i in options.keys():
        if i in kwargs:
            options[i] = kwargs[i]

    printHeader("Options Run:",2)
    for key,value in options.items():
        psi4.core.print_out(f"{key:20s} {str(value):20s} \n")


   

    
    

    printHeader("Basis Set:",2)
    wfn   = psi4.core.Wavefunction.build(mol.psi4Mol,mol.basisSet)
    aux   = psi4.core.BasisSet.build(mol.psi4Mol, "DF_BASIS_SCF", "", "JKFIT", mol.basisString, puream=1 if wfn.basisset().has_puream() else 0)

    sup = psi4.driver.dft.build_superfunctional(func, False)[0]

    psi4.core.be_quiet()
    mints = psi4.core.MintsHelper(mol.basisSet)
    
    
    sup.allocate()

    uhf   = psi4.core.UHF(wfn,sup)
    psi4.core.reopen_outfile()
    
    S = np.asarray(mints.ao_overlap())
    T = np.asarray(mints.ao_kinetic())
    
    H = np.zeros((mints.nbf(),mints.nbf()))

    if "AOPOT" in kwargs:
        V = kwargs["AOPOT"]
        psi4.core.print_out("\nChanged AO-potential!\n")
    else:
        V = np.asarray(mints.ao_potential())
    H = T+V

    if wfn.basisset().has_ECP():
        ECP = mints.ao_ecp()
        H += ECP

    A = mints.ao_overlap()
    A.power(-0.5,1.e-16)
    A = np.asarray(A)


    Enuc = mol.psi4Mol.nuclear_repulsion_energy()
    Eold = 0.0

    nbf    = wfn.nso()
    nalpha = wfn.nalpha()
    nbeta  = wfn.nbeta()

    Va = psi4.core.Matrix(nbf,nbf)
    Vb = psi4.core.Matrix(nbf,nbf)

    Vpot = psi4.core.VBase.build(wfn.basisset(), sup, "UV")
    Vpot.initialize()

    Cocca       = psi4.core.Matrix(nbf, nalpha)
    Coccb       = psi4.core.Matrix(nbf, nbeta)

    if not(options["OCCA"] is None) and not(options["OCCB"] is None):
        occa = np.pad(options['OCCA'],(0,nbf-len(options['OCCA'])))
        occb = np.pad(options['OCCB'],(0,nbf-len(options['OCCB'])))
        
        Cocca       = psi4.core.Matrix(nbf, np.count_nonzero(occa))
        Coccb       = psi4.core.Matrix(nbf, np.count_nonzero(occb))

        nalpha = np.count_nonzero(occa)
        nbeta  = np.count_nonzero(occb)
    else:
        occa = np.zeros(nbf)
        occa[:nalpha] = 1.0
        occb = np.zeros(nbf)
        occb[:nbeta] = 1.0
        
    psi4.core.print_out(f"\nPadded Occupation Vectors: {occa}")
    psi4.core.print_out(f"\nNon-zerop alpha orbs: {np.count_nonzero(occa)}")
        
        
        
    psi4.core.print_out(f"\nPadded Occupation Vectors: {occb} \n\n")
    psi4.core.print_out(f"\nNon-zerop alpha orbs: {np.count_nonzero(occb)}")
            



    """
    Read or Core Guess
    """    
   
    if "Cinp" in kwargs:
        psi4.core.print_out("Taking Coefficients from kwargs\n\n")
        Ca = kwargs["Cinp"][0]
        Cb = kwargs["Cinp"][1]
        Cocca.np[:]  = Ca[:, :nalpha]*np.sqrt(occa[:nalpha])
        Da     = Ca[:, :nalpha] @ Ca[:, :nalpha].T
        Coccb.np[:]  = Cb[:, :nbeta]*np.sqrt(occb[:nbeta])
        Db     = Cb[:, :nbeta] @ Cb[:, :nbeta].T
    else:
        """
        Just do a Core Guess
        """
        psi4.core.print_out("Creating CORE guess\n\n")
        Ca,_ = diag_H(H, A)
        Cb = np.copy(Ca)

        Cocca.np[:] = Ca[:, :nalpha]
        Coccb.np[:] = Cb[:, :nbeta]
     
        #This is the guess!
        Da  = Cocca.np @ Cocca.np.T
        Db  = Coccb.np @ Coccb.np.T


    printHeader("Molecule:",2)
    mol.psi4Mol.print_out()
    printHeader("XC & JK-Info:",2)

    jk = psi4.core.JK.build(wfn.basisset(),aux=aux,jk_type="MEM_DF")
    glob_mem = psi4.core.get_memory()/8
    jk.set_memory(int(glob_mem*0.6))
    
    if (sup.is_x_hybrid()):
        jk.set_do_K(True)
    if (sup.is_x_lrc()):
        jk.set_omega(sup.x_omega())
        jk.set_do_wK(True)
    jk.initialize()
    jk.C_left_add(Cocca)
    jk.C_left_add(Coccb)

    Da_m = psi4.core.Matrix(nbf,nbf)
    Db_m = psi4.core.Matrix(nbf,nbf)
    
    sup.print_out()
    psi4.core.print_out("\n\n")
    mol.basisSet.print_out()
    jk.print_header()
        
    diis = ACDIIS(max_vec=options["DIIS_LEN"],diismode=options["DIIS_MODE"])
    diisa_e = 1000.0
    diisb_e = 1000.0

    printHeader("Starting SCF:",2)    
    psi4.core.print_out("""{:>10} {:8.2E}
{:>10} {:8.2E}
{:>10} {:8.4f}
{:>10} {:8.2E}
{:>10} {:8d}
{:>10} {:8d}
{:>10} {:^11}""".format(
    "E_CONV:",options["E_CONV"],
    "D_CONV:",options["D_CONV"],
    "DAMP:",options["GAMMA"],
    "DIIS_EPS:",options["DIIS_EPS"],
    "MAXITER:", options["MAXITER"],
    "DIIS_LEN:",options["DIIS_LEN"],
    "DIIS_MODE:",options["DIIS_MODE"]))

    myTimer = Timer()

    psi4.core.print_out("\n\n{:^4} {:^14} {:^11} {:^11} {:^11} {:^11} {:^6} \n".format("# IT", "Escf", "dEscf","Derror","DIIS-E","MIX","Time"))
    psi4.core.print_out("="*80+"\n")
    diis_counter = 0

    for SCF_ITER in range(1, options["MAXITER"] + 1):
        myTimer.addStart("SCF")     
        jk.compute()
        
        """
        Build Fock
        """
        Da_m.np[:] = Da
        Db_m.np[:] = Db
        Vpot.set_D([Da_m,Db_m])
        Vpot.compute_V([Va,Vb])

        Ja = np.asarray(jk.J()[0])
        Jb = np.asarray(jk.J()[1])

        if SCF_ITER>1 :
            FaOld = np.copy(Fa)
            FbOld = np.copy(Fb)

        Fa = H + (Ja + Jb) + Va
        Fb = H + (Ja + Jb) + Vb
        if sup.is_x_hybrid():
            Fa -= sup.x_alpha()*np.asarray(jk.K()[0]) 
            Fb -= sup.x_alpha()*np.asarray(jk.K()[1])
        if sup.is_x_lrc(): 
            Fa -= sup.x_beta()*np.asarray(jk.wK()[0]) 
            Fb -= sup.x_beta()*np.asarray(jk.wK()[1])
        """
        END BUILD FOCK
        """

        """
        CALC E
        """
        one_electron_E  = np.sum(Da * H)
        one_electron_E += np.sum(Db * H)
        coulomb_E       = np.sum(Da * (Ja+Jb))
        coulomb_E      += np.sum(Db * (Ja+Jb))

        exchange_E  = 0.0
        if sup.is_x_hybrid():
            exchange_E -=  sup.x_alpha() * np.sum(Da * np.asarray(jk.K()[0]))
            exchange_E -=  sup.x_alpha() * np.sum(Db * np.asarray(jk.K()[1]))
        if sup.is_x_lrc():
            exchange_E -= sup.x_beta() * np.sum(Da * np.asarray(jk.wK()[0]))
            exchange_E -= sup.x_beta() * np.sum(Db * np.asarray(jk.wK()[1]))


        XC_E = Vpot.quadrature_values()["FUNCTIONAL"]


        SCF_E = 0.0
        SCF_E += Enuc
        SCF_E += one_electron_E
        SCF_E += 0.5 * coulomb_E
        SCF_E += 0.5 * exchange_E
        SCF_E += XC_E
        """
        END CALCE
        """

        """
        DIIS/MIXING
        """
        diisa_e = np.ravel(A.T@(Fa@Da@S - S@Da@Fa)@A)
        diisb_e = np.ravel(A.T@(Fb@Db@S - S@Db@Fb)@A)
        diis.add(Fa,Fb,Da,Db,np.concatenate((diisa_e,diisb_e)))


        if ("DIIS" in options["MIXMODE"]) and (SCF_ITER>1):
            # Extrapolate alpha & beta Fock matrices separately
            (Fa,Fb) = diis.extrapolate(DIISError)
            diis_counter += 1

            if (diis_counter >= 2*options["DIIS_LEN"]):
                diis.reset()
                diis_counter = 0
                psi4.core.print_out("Resetting DIIS\n")

        elif (options["MIXMODE"] == "DAMP") and (SCF_ITER>1):
            #...but use damping to obtain the new Fock matrices
            Fa = (1-options["GAMMA"]) * np.copy(Fa) + (options["GAMMA"]) * FaOld
            Fb = (1-options["GAMMA"]) * np.copy(Fb) + (options["GAMMA"]) * FbOld

        """
        END DIIS/MIXING
        """
       

        """
        DIAG F-tilde -> get D
        """
        DaOld = np.copy(Da)
        DbOld = np.copy(Db)

        Ca,epsa = diag_H(Fa, A)
        
        Cocca.np[:]  = Ca[:, :nalpha]*np.sqrt(occa[:nalpha])
        Da      = Cocca.np @ Cocca.np.T


        Cb,epsb = diag_H(Fb, A)
        Coccb.np[:]  = Cb[:, :nbeta]*np.sqrt(occb[:nbeta])
        Db      = Coccb.np @ Coccb.np.T
        """
        END DIAG F + BUILD D
        """

        DError = (np.sum((DaOld-Da)**2)**0.5 + np.sum((DbOld-Db)**2)**0.5)
        EError = (SCF_E - Eold)
        DIISError = (np.sum(diisa_e**2)**0.5 + np.sum(diisb_e**2)**0.5)
     
        """
        OUTPUT
        """
        myTimer.addEnd("SCF")
        psi4.core.print_out(" {:3d} {:14.8f} {:11.3E} {:11.3E} {:11.3E} {:^11} {:6.2f} {:2d} \n".format(SCF_ITER,
             SCF_E,
             EError,
             DError,
             DIISError,
             options["MIXMODE"],
             myTimer.getTime("SCF"),
             len(diis.Fa)))
                  
        psi4.core.flush_outfile()
        if (abs(DIISError) < options["DIIS_EPS"]):
            options["MIXMODE"] = options["DIIS_MODE"]
        else:
            options["MIXMODE"] = "DAMP"        
        
        if (abs(EError) < options["E_CONV"]) and (abs(DError)<options["D_CONV"]):
            break

        Eold = SCF_E

        if SCF_ITER == options["MAXITER"]:
            psi4.core.print_out("\n\nMaximum number of SCF cycles exceeded.")
            raise Exception("Maximum number of SCF cycles exceeded.")

    psi4.core.print_out("\n\nFINAL GS SCF ENERGY: {:12.8f} [Ha] \n\n".format(SCF_E))

    results = {"SCF_E" : SCF_E, "Da":Da,"Db":Db,"Ca" : Ca, "Cb":Cb}

    return results
