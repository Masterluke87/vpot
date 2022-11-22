## General program structure

1. Provide an XYZ file + a basis set string.
2. Generate several objects
3. Either choose a spherical grid or a block grid.
4. Perform least square algorithm.


## Requirements
- Psi4
- ASE
- numpy
- logging 


## Details

### Spherical Grid
That grid should depend on:
1. A **minimum distance** that grid points need to be away from nuclei
2. A **maximum distance** that grid points should not exceed to be away from any nucleus
3. The number of spherical and radial points
4. The radial scheme ("TREUTLER","BECKE")
5. Pruning scheme ("TREUTLER","ROBUST")


