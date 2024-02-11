#!/usr/bin/env python

from pyscf import gto, scf, dft

mol = gto.Mole()
mol.build(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'cc-pvtz',
    symmetry = True,
    verbose=5,
)

mf = dft.RKS(mol)
mf.xc = 'pbe0'
mf.kernel()

from rttddft.rttdbase import RTTDSCF
myrtd = RTTDSCF(mf)

myrtd.kernel(10.0, 0.1)
