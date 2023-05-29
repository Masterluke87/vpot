# File description

- The archive qm7.tar.gz contains 7211 folders (from 0 to 7210). 
- Each folder contains 5 files:
    1. ``*.xyz``: file representing the structure
    2. ``*.png``: Figures of the fitted potential + the total error
    3. ``PSI_V_EXT.out``: output of the PSI4 calculation using the exact external potential
    4. ``PSI_V_ANC.out``: output of the PSI4 calculation using the basis set expanded ANC potential
    5. ``input.npz`` : A numpy archive containing information and input matrices (see below)
    6. ``output.npz`` : A numpy archive containing output matrices (see below)

## Structure of ``input.npz``
- Load the file with:
```python
import numpy as np
I = np.load("input.npz",allow_pickle=True)
```
The fields that are available in that file can be inspected as:
```python
print(list(I.keys()))
```
The file contains the following fields (K is the number of basis functions):
- ``I['C_V_ANC']``: KxK matrix. The K diagonal elements are fitted to represent the external ANC potential:

```math
    \sum^{K}_{\mu} C^{v,ANC}_{\mu\mu} \phi_\mu (r) \approx V(r) = -\sum_A \dfrac{Z_A}{|R_A -r|}
```

The non-diagonal elements $\mu \nu$ are overlap integrals:
```math
    C^{v,ANC}_{\mu \nu} = S_{\mu \nu} = \langle \phi_\mu \mid \phi_\nu \rangle
```

- ``I['V_ANC_B']``: External potential matrix used for the DFT calculation, calculated from the basis set expanded potential:
```math
   \underline{\underline{V}}^{ANC} = \langle \phi_mu\mid \sum^{K}_{\sigma} C^{v,ANC}_{\sigma\sigma} \phi_\sigma \mid \phi_nu \rangle
```
- ``I['V_EXT']``: External potential matrix, calculated from the exact potential.
- ``I['NEL'].item()``: Number of electrons in the system.
- ``I['INFO'].item()``: A dictonary which holds more information on the actual calcaultion.


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
