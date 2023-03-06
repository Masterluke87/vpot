# File description

- The archive qm7.tar.gz contains 7211 folders (from 0 to 7210). 
- Each folder contains 5 files:
    1. ``*.xyz``: file representing the structure
    2. ``error.svg``: containing some statistics of the fitting procedure 
    3. ``psi.out``: output of the PSI4 calculation 
    4. ``input.npz`` : A numpy archive containing information and input matrices (see below)
    5. ``output.npz`` : A numpy archive containing output matrices (see below)

## Structure of ``input.npz``
- Load the file with:
```python
import numpy as np
I = np.load("input.npz",allow_pickle=True)
```
The file contains the following fields (K is the number of basis functions):
- ``I['Vfit']``: KxK matrix. The K diagonal elements are fitted to represent the external potential:

```math
    \sum^{K}_{\mu} V^{fit}_{\mu\mu} \phi_\mu (r) \approx V(r) = -\sum_A \dfrac{Z_A}{|R_A -r|}
```

The non-diagonal elements $\mu \nu$ are overlap integrals:
```math
    V^{fit}_{\mu \nu} = S_{\mu \nu} = \langle \phi_\mu \mid \phi_\nu \rangle
```
- ``I['Vpot']``: KxK matrix, representing the ao_potential matrix (might also be useful):
```math
 V_{\mu \nu}^{pot} = \langle \phi_\mu \mid -\sum_A \dfrac{Z_A}{|R_A -r|} \mid\phi_\nu \rangle
``` 
- ``I['Nel'].item()`` contains the number of electrons
- ``I['basis'].item()`` string containing the basis set
- ``I['gridInfo'].item()`` dict with some information about the grid used in the fitting scheme.


## Structure of ``output.npz``
- Load the file with:
```python
import numpy as np
O = np.load("output.npz",allow_pickle=True)
```
The file contains the following fields:
- ``I['Fa']``: KxK converged Fock-matrix:
```math
F_{\mu \nu } = \langle \phi_\mu \mid \hat{f} \mid\phi_\nu \rangle
```
- ``I['Da']``: KxK Density-matrix.
