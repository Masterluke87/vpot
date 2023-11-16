# File description
- You should work on two files:
    1. ``qm7-pbe.tar.gz`` : Structures calculated with  PBE/def2-TZVP, 116 calculations did *not* converge, i.e. there is no output.npz.
    2. ``qm7-pbe0.tar.gz`` : Structures calculated with PBE0/def2-TZVP, all calculations converged
- Each folder contains 5 files:
    1. ``*.xyz``: file representing the structure
    2. ``*.png``: Figures of the fitted potential + the total error
    3. ``PSI_V_EXT.out``: output of the PSI4 calculation using the exact external potential
    4. ``PSI_V_ANC.out``: output of the PSI4 calculation using the basis set expanded ANC potential
    5. ``input*.npz`` : A numpy archive containing information and input matrices (see below). There are different versions. Try to use the one with the largest number.
    6. ``output*.npz`` : A numpy archive containing output matrices (see below), There are different version. Take the one with the largest number.

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
    \sum^{K}_{\mu} C^{v,ANC}_{\mu\mu} \phi_\mu (\underline{r}) \approx V(\underline{r}) = -\sum_A \dfrac{Z_A}{|\underline{R}_A -\underline{r}|}
```

The non-diagonal elements $\mu \nu$ are overlap integrals:
```math
    C^{v,ANC}_{\mu \nu} = S_{\mu \nu} = \langle \phi_\mu \mid \phi_\nu \rangle
```
- ``I['C_V_ANC']``: K dimensional vector. Contains the expansion of the ANC potential in the orthogonal basis:
```math
\underline{\underline{C}}^{v,ANC,o} = \underline{\underline{X}}^{-1} \cdot \underline{\underline{C}}^{v,ANC}
```
with $\underline{\underline{X}}$ beeing the Loewdin Matrix ($\underline{\underline{S}}^{-1/2}$).

- ``I['V_ANC_B']``: External potential matrix used for the DFT calculation, calculated from the basis set expanded potential:
```math
   {V}_{\mu \nu}^{ANC} = \langle \phi_\mu\mid \sum^{K}_{\sigma} C^{v,ANC}_{\sigma\sigma} \phi_\sigma \mid \phi_\nu \rangle
```
- ``I['V_EXT']``: External potential matrix, calculated from the exact potential.
```math
 V^{exact}_{\mu\nu} = \langle \phi_\mu \mid -\sum_A \dfrac{Z_A}{|\underline{R}_A - \underline{r}|} \mid \phi_\nu \rangle
```
- ``I['NEL'].item()``: Number of electrons in the system.
- ``I['INFO'].item()``: A dictonary which holds more information on the actual calcaultion.


## Structure of ``output.npz``
- Load the file with:
```python
import numpy as np
O = np.load("output.npz",allow_pickle=True)
```
The file contains the following fields:
- ``O['P_ANC_B']``: KxK matrix holding the density matrix, calculated with the basis set expanded ANC potential $\underline{\underline{V}}^{ANC}$

- ``O['E_ANC'].item()``: Converged electronic energy calculated with with the basis set expanded ANC potential $\underline{\underline{V}}^{ANC}$.

- ``O['P_EXT']``: KxK matrix holding the density matrix, calculated with the exact external potential $\underline{\underline{V}}^{exact}$.

- ``O['E_EXT'].item()``: Converged electronic energy calculated with the exact external potential $\underline{V}^{exact}$.
